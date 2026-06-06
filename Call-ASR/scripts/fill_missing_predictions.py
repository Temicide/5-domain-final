#!/usr/bin/env python3
"""
Fill missing predictions in unfinished CSV with temporary placeholders.

Usage:
    python3 scripts/fill_missing_predictions.py \
        --unfinished outputs/predictions_typhoon-ai__typhoon-whisper-large-v3_single_space_unfinished_2.csv \
        --sample data/individual-test-thai-call-center-asr/sample_submission.csv \
        --output outputs/predictions_typhoon-ai__typhoon-whisper-large-v3_single_space_complete.csv
"""

import argparse
import csv
from pathlib import Path


PLACEHOLDER_TEXT = "[TEMPORARY_PLACEHOLDER]"
MODEL_NAME = "typhoon-ai/typhoon-whisper-large-v3"


def load_unfinished_predictions(path: Path) -> dict[str, dict]:
    """Load existing predictions into a dict keyed by file_name."""
    predictions = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            predictions[row["file_name"]] = row
    return predictions


def load_sample_submission(path: Path) -> list[str]:
    """Load the list of expected file names from sample submission."""
    file_names = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            file_names.append(row["file_name"])
    return file_names


def create_placeholder_row(file_name: str) -> dict:
    """Create a placeholder row for a missing prediction."""
    return {
        "file_name": file_name,
        "raw_text": PLACEHOLDER_TEXT,
        "normalized_text": PLACEHOLDER_TEXT,
        "model_name": MODEL_NAME,
        "avg_logprob": "0.0",
        "compression_ratio": "0.0",
        "no_speech_prob": "0.0",
        "error": "",
    }


def main():
    parser = argparse.ArgumentParser(description="Fill missing predictions with placeholders")
    parser.add_argument(
        "--unfinished",
        type=Path,
        required=True,
        help="Path to unfinished predictions CSV",
    )
    parser.add_argument(
        "--sample",
        type=Path,
        required=True,
        help="Path to sample submission CSV",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to output submission CSV (file_name,text only)",
    )
    args = parser.parse_args()

    print(f"Loading unfinished predictions from: {args.unfinished}")
    existing = load_unfinished_predictions(args.unfinished)
    print(f"  Found {len(existing)} existing predictions")

    print(f"Loading sample submission from: {args.sample}")
    expected_files = load_sample_submission(args.sample)
    print(f"  Expected {len(expected_files)} files")

    missing_count = 0
    rows = []

    for file_name in expected_files:
        if file_name in existing:
            rows.append({"file_name": file_name, "text": existing[file_name]["normalized_text"]})
        else:
            rows.append({"file_name": file_name, "text": PLACEHOLDER_TEXT})
            missing_count += 1

    print(f"Missing files filled with placeholders: {missing_count}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file_name", "text"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved submission to: {args.output}")
    print(f"Total rows: {len(rows)}")


if __name__ == "__main__":
    main()
