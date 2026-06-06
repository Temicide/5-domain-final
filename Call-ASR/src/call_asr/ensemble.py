from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from call_asr.audio import audio_prefix


OUTPUT_COLUMNS = ["file_name", "normalized_text", "selected_model"]


def _candidate_row(frame: pd.DataFrame, file_name: str) -> pd.Series:
    matches = frame.loc[frame["file_name"] == file_name]
    if len(matches) != 1:
        raise ValueError(f"Expected one row for {file_name}, got {len(matches)}")
    return matches.iloc[0]


def prefix_route(
    candidates: dict[str, pd.DataFrame],
    route_by_prefix: dict[str, str],
    default_model: str,
) -> pd.DataFrame:
    if default_model not in candidates:
        raise ValueError(f"Default model not found in candidates: {default_model}")
    base = candidates[default_model]
    rows = []
    for file_name in base["file_name"].tolist():
        selected_model = route_by_prefix.get(audio_prefix(file_name), default_model)
        if selected_model not in candidates:
            raise ValueError(f"Route references missing model {selected_model} for {file_name}")
        candidate = _candidate_row(candidates[selected_model], file_name)
        rows.append(
            {
                "file_name": file_name,
                "normalized_text": candidate["normalized_text"],
                "selected_model": selected_model,
            }
        )
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def confidence_select(candidates: dict[str, pd.DataFrame]) -> pd.DataFrame:
    first_model = next(iter(candidates))
    file_names = candidates[first_model]["file_name"].tolist()
    rows = []
    for file_name in file_names:
        scored_rows = []
        for model_name, frame in candidates.items():
            row = _candidate_row(frame, file_name).copy()
            row["selected_model"] = model_name
            scored_rows.append(row)
        selected = sorted(
            scored_rows,
            key=lambda row: (
                float(row.get("avg_logprob", 0.0)),
                -float(row.get("no_speech_prob", 0.0)),
                -float(row.get("compression_ratio", 0.0)),
            ),
            reverse=True,
        )[0]
        rows.append(
            {
                "file_name": file_name,
                "normalized_text": selected["normalized_text"],
                "selected_model": selected["selected_model"],
            }
        )
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine candidate ASR transcripts.")
    parser.add_argument("--method", choices=["confidence"], required=True)
    parser.add_argument("--candidate", action="append", nargs=2, metavar=("MODEL_NAME", "CSV_PATH"), required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()
    candidates = {name: pd.read_csv(path, keep_default_na=False) for name, path in args.candidate}
    if args.method == "confidence":
        result = confidence_select(candidates)
    else:
        raise ValueError(f"Unsupported method: {args.method}")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_csv, index=False)
    print(f"Wrote ensemble: {args.output_csv}")
    print(f"Rows: {len(result)}")


if __name__ == "__main__":
    main()
