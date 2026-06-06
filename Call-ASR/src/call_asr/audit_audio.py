from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from call_asr.audio import probe_audio_file
from call_asr.paths import resolve_competition_paths


def audit_audio_directory(audio_dir: Path, output_csv: Path, failures_jsonl: Path) -> pd.DataFrame:
    rows = [asdict(probe_audio_file(path)) for path in sorted(audio_dir.glob("*.wav"))]
    inventory = pd.DataFrame(rows)
    if inventory.empty:
        inventory = pd.DataFrame(
            columns=[
                "file_name",
                "prefix",
                "sample_rate",
                "channels",
                "frames",
                "duration_seconds",
                "decode_ok",
                "error",
            ]
        )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    failures_jsonl.parent.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(output_csv, index=False)
    failures = inventory[inventory["decode_ok"] == False]
    with failures_jsonl.open("w", encoding="utf-8") as handle:
        for row in failures.to_dict("records"):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return inventory


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit Thai call-center WAV files.")
    parser.add_argument("--audio-dir", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--failures-jsonl", type=Path, default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    paths = resolve_competition_paths()
    audio_dir = args.audio_dir or paths.audio_dir
    output_csv = args.output_csv or (paths.working_dir / "audio_inventory.csv")
    failures_jsonl = args.failures_jsonl or (paths.working_dir / "decode_failures.jsonl")
    inventory = audit_audio_directory(audio_dir, output_csv, failures_jsonl)
    prefix_counts = inventory["prefix"].value_counts().sort_index().to_dict()
    print(f"Audited {len(inventory)} WAV files")
    print(f"Decode failures: {int((inventory['decode_ok'] == False).sum())}")
    print(f"Prefix counts: {prefix_counts}")
    print(f"Wrote inventory: {output_csv}")
    print(f"Wrote failures: {failures_jsonl}")


if __name__ == "__main__":
    main()
