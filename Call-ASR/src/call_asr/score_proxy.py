from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from call_asr.audio import audio_prefix


def _edit_distance(reference: str, prediction: str) -> int:
    previous = list(range(len(prediction) + 1))
    for i, ref_char in enumerate(reference, start=1):
        current = [i]
        for j, pred_char in enumerate(prediction, start=1):
            substitution_cost = 0 if ref_char == pred_char else 1
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + substitution_cost,
                )
            )
        previous = current
    return previous[-1]


def character_error_rate(reference: str, prediction: str) -> float:
    reference = str(reference)
    prediction = str(prediction)
    if len(reference) == 0:
        return 0.0 if prediction == "" else 1.0
    return _edit_distance(reference, prediction) / len(reference)


def score_predictions(reference_df: pd.DataFrame, predictions_df: pd.DataFrame) -> dict:
    merged = reference_df[["file_name", "text"]].merge(
        predictions_df[["file_name", "normalized_text"]],
        on="file_name",
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(reference_df):
        missing = sorted(set(reference_df["file_name"]) - set(merged["file_name"]))
        raise ValueError(f"Missing prediction rows: {', '.join(missing[:20])}")

    total_distance = 0
    total_reference_chars = 0
    by_prefix: dict[str, dict[str, float | int]] = {}
    for row in merged.itertuples(index=False):
        reference = str(row.text)
        prediction = str(row.normalized_text)
        distance = _edit_distance(reference, prediction)
        ref_len = len(reference)
        prefix = audio_prefix(row.file_name)
        total_distance += distance
        total_reference_chars += ref_len
        stats = by_prefix.setdefault(prefix, {"distance": 0, "reference_chars": 0, "count": 0, "cer": 0.0})
        stats["distance"] += distance
        stats["reference_chars"] += ref_len
        stats["count"] += 1

    for stats in by_prefix.values():
        stats["cer"] = 0.0 if stats["reference_chars"] == 0 else stats["distance"] / stats["reference_chars"]

    return {
        "overall_cer": 0.0 if total_reference_chars == 0 else total_distance / total_reference_chars,
        "count": int(len(merged)),
        "by_prefix": by_prefix,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Score proxy ASR predictions.")
    parser.add_argument("--reference-csv", type=Path, required=True)
    parser.add_argument("--predictions-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    reference_df = pd.read_csv(args.reference_csv, keep_default_na=False)
    predictions_df = pd.read_csv(args.predictions_csv, keep_default_na=False)
    scores = score_predictions(reference_df, predictions_df)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(scores, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(scores, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
