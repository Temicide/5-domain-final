import json

import pandas as pd

from call_asr.audit_audio import audit_audio_directory


def test_audit_audio_directory_writes_inventory_and_failures(tmp_path):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    bad_file = audio_dir / "RSP_bad_audio.wav"
    bad_file.write_text("not a wav", encoding="utf-8")
    output_csv = tmp_path / "audio_inventory.csv"
    failures_jsonl = tmp_path / "decode_failures.jsonl"

    inventory = audit_audio_directory(audio_dir, output_csv, failures_jsonl)

    assert inventory.loc[0, "file_name"] == "RSP_bad_audio.wav"
    assert inventory.loc[0, "decode_ok"] is False or inventory.loc[0, "decode_ok"] == False
    assert output_csv.is_file()
    assert failures_jsonl.is_file()
    failure = json.loads(failures_jsonl.read_text(encoding="utf-8").strip())
    assert failure["file_name"] == "RSP_bad_audio.wav"
    assert "error" in failure


def test_audit_audio_directory_sorts_by_file_name(tmp_path):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    for name in ["SDB_002_audio.wav", "SDB_001_audio.wav"]:
        (audio_dir / name).write_text("not a wav", encoding="utf-8")
    output_csv = tmp_path / "audio_inventory.csv"
    failures_jsonl = tmp_path / "decode_failures.jsonl"

    audit_audio_directory(audio_dir, output_csv, failures_jsonl)

    written = pd.read_csv(output_csv)
    assert written["file_name"].tolist() == ["SDB_001_audio.wav", "SDB_002_audio.wav"]
