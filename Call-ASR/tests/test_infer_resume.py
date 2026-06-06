import json

import pandas as pd

from call_asr.infer import AsrResult, FakeAsrBackend, run_inference


def test_run_inference_writes_predictions_and_jsonl_log(tmp_path):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    (audio_dir / "RSP_001_audio.wav").write_bytes(b"fake")
    sample_df = pd.DataFrame({"file_name": ["RSP_001_audio.wav"], "text": [""]})
    output_csv = tmp_path / "predictions.csv"
    log_jsonl = tmp_path / "run.jsonl"
    backend = FakeAsrBackend(
        {"RSP_001_audio.wav": AsrResult(text=" สวัสดีค่ะ ", avg_logprob=-0.1, compression_ratio=1.2, no_speech_prob=0.01)}
    )

    predictions = run_inference(
        sample_df=sample_df,
        audio_dir=audio_dir,
        output_csv=output_csv,
        log_jsonl=log_jsonl,
        backend=backend,
        normalization_policy="single_space",
        resume=True,
    )

    assert predictions.to_dict("records") == [
        {
            "file_name": "RSP_001_audio.wav",
            "raw_text": " สวัสดีค่ะ ",
            "normalized_text": "สวัสดีค่ะ",
            "model_name": "fake",
            "avg_logprob": -0.1,
            "compression_ratio": 1.2,
            "no_speech_prob": 0.01,
            "error": "",
        }
    ]
    log_row = json.loads(log_jsonl.read_text(encoding="utf-8").strip())
    assert log_row["file_name"] == "RSP_001_audio.wav"
    assert log_row["normalized_text"] == "สวัสดีค่ะ"


def test_run_inference_resumes_existing_rows(tmp_path):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    (audio_dir / "RSP_001_audio.wav").write_bytes(b"fake")
    (audio_dir / "SDB_001_audio.wav").write_bytes(b"fake")
    sample_df = pd.DataFrame({"file_name": ["RSP_001_audio.wav", "SDB_001_audio.wav"], "text": ["", ""]})
    output_csv = tmp_path / "predictions.csv"
    log_jsonl = tmp_path / "run.jsonl"
    pd.DataFrame(
        [
            {
                "file_name": "RSP_001_audio.wav",
                "raw_text": "เดิม",
                "normalized_text": "เดิม",
                "model_name": "fake",
                "avg_logprob": -0.2,
                "compression_ratio": 1.0,
                "no_speech_prob": 0.0,
                "error": "",
            }
        ]
    ).to_csv(output_csv, index=False)
    backend = FakeAsrBackend({"SDB_001_audio.wav": AsrResult(text="ใหม่", avg_logprob=-0.3, compression_ratio=1.1, no_speech_prob=0.02)})

    predictions = run_inference(
        sample_df=sample_df,
        audio_dir=audio_dir,
        output_csv=output_csv,
        log_jsonl=log_jsonl,
        backend=backend,
        normalization_policy="raw",
        resume=True,
    )

    assert predictions["file_name"].tolist() == ["RSP_001_audio.wav", "SDB_001_audio.wav"]
    assert backend.calls == ["SDB_001_audio.wav"]
