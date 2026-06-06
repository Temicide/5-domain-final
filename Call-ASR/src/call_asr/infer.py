from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pandas as pd

from call_asr.normalize_text import normalize_text
from call_asr.paths import resolve_competition_paths
from call_asr.submission import load_sample_submission, validate_audio_coverage


PREDICTION_COLUMNS = [
    "file_name",
    "raw_text",
    "normalized_text",
    "model_name",
    "avg_logprob",
    "compression_ratio",
    "no_speech_prob",
    "error",
]


@dataclass(frozen=True)
class AsrResult:
    text: str
    avg_logprob: float
    compression_ratio: float
    no_speech_prob: float


class AsrBackend(Protocol):
    model_name: str

    def transcribe(self, audio_path: Path) -> AsrResult:
        ...


class FakeAsrBackend:
    model_name = "fake"

    def __init__(self, outputs: dict[str, AsrResult]):
        self.outputs = outputs
        self.calls: list[str] = []

    def transcribe(self, audio_path: Path) -> AsrResult:
        self.calls.append(audio_path.name)
        return self.outputs[audio_path.name]


class WhisperPipelineBackend:
    def __init__(
        self,
        model_name: str,
        chunk_length_s: int,
        batch_size: int,
        condition_on_previous_text: bool,
        device: str,
    ):
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

        self.model_name = model_name
        torch_dtype = torch.float16 if device == "cuda" else torch.float32
        device_index = 0 if device == "cuda" else -1
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        processor = AutoProcessor.from_pretrained(model_name)
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            chunk_length_s=chunk_length_s,
            batch_size=batch_size,
            torch_dtype=torch_dtype,
            device=device_index,
            return_timestamps=False,
        )
        self.generate_kwargs = {
            "language": "thai",
            "task": "transcribe",
            "condition_on_prev_tokens": condition_on_previous_text,
        }

    def transcribe(self, audio_path: Path) -> AsrResult:
        output = self.pipe(str(audio_path), generate_kwargs=self.generate_kwargs)
        text = str(output.get("text", ""))
        return AsrResult(text=text, avg_logprob=0.0, compression_ratio=0.0, no_speech_prob=0.0)


def _load_existing(output_csv: Path) -> pd.DataFrame:
    if output_csv.is_file():
        return pd.read_csv(output_csv, keep_default_na=False)
    return pd.DataFrame(columns=PREDICTION_COLUMNS)


def _write_log(log_jsonl: Path, row: dict) -> None:
    log_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with log_jsonl.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_inference(
    sample_df: pd.DataFrame,
    audio_dir: Path,
    output_csv: Path,
    log_jsonl: Path,
    backend: AsrBackend,
    normalization_policy: str,
    resume: bool,
) -> pd.DataFrame:
    existing = _load_existing(output_csv) if resume else pd.DataFrame(columns=PREDICTION_COLUMNS)
    completed = set(existing["file_name"].tolist()) if not existing.empty else set()
    rows = existing.to_dict("records")

    for file_name in sample_df["file_name"].tolist():
        if file_name in completed:
            continue
        started = time.time()
        audio_path = audio_dir / file_name
        try:
            result = backend.transcribe(audio_path)
            raw_text = result.text
            normalized_text = normalize_text(raw_text, normalization_policy)
            row = {
                "file_name": file_name,
                "raw_text": raw_text,
                "normalized_text": normalized_text,
                "model_name": backend.model_name,
                "avg_logprob": result.avg_logprob,
                "compression_ratio": result.compression_ratio,
                "no_speech_prob": result.no_speech_prob,
                "error": "",
            }
        except Exception as exc:
            row = {
                "file_name": file_name,
                "raw_text": "",
                "normalized_text": "",
                "model_name": backend.model_name,
                "avg_logprob": 0.0,
                "compression_ratio": 0.0,
                "no_speech_prob": 0.0,
                "error": f"{type(exc).__name__}: {exc}",
            }
        log_row = dict(row)
        log_row["runtime_seconds"] = round(time.time() - started, 4)
        _write_log(log_jsonl, log_row)
        rows.append(row)
        pd.DataFrame(rows, columns=PREDICTION_COLUMNS).to_csv(output_csv, index=False)

    output = pd.DataFrame(rows, columns=PREDICTION_COLUMNS)
    order = {file_name: position for position, file_name in enumerate(sample_df["file_name"].tolist())}
    output["_order"] = output["file_name"].map(order)
    output = output.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)
    output.to_csv(output_csv, index=False)
    return output


def _default_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Thai ASR inference.")
    parser.add_argument("--model-name", default="typhoon-ai/typhoon-whisper-large-v3")
    parser.add_argument("--normalization-policy", default="single_space")
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--log-jsonl", type=Path, default=None)
    parser.add_argument("--chunk-length-s", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--condition-on-previous-text", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--device", choices=["cuda", "cpu"], default=_default_device())
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    paths = resolve_competition_paths()
    sample_df = load_sample_submission(paths.sample_submission)
    validate_audio_coverage(sample_df, paths.audio_dir)
    safe_model_name = args.model_name.replace("/", "__")
    output_csv = args.output_csv or (paths.working_dir / f"predictions_{safe_model_name}_{args.normalization_policy}.csv")
    log_jsonl = args.log_jsonl or (paths.working_dir / f"run_{safe_model_name}_{args.normalization_policy}.jsonl")
    backend = WhisperPipelineBackend(
        model_name=args.model_name,
        chunk_length_s=args.chunk_length_s,
        batch_size=args.batch_size,
        condition_on_previous_text=args.condition_on_previous_text,
        device=args.device,
    )
    predictions = run_inference(
        sample_df=sample_df,
        audio_dir=paths.audio_dir,
        output_csv=output_csv,
        log_jsonl=log_jsonl,
        backend=backend,
        normalization_policy=args.normalization_policy,
        resume=not args.no_resume,
    )
    errors = int((predictions["error"] != "").sum())
    print(f"Wrote predictions: {output_csv}")
    print(f"Wrote log: {log_jsonl}")
    print(f"Rows: {len(predictions)}")
    print(f"Errors: {errors}")


if __name__ == "__main__":
    main()
