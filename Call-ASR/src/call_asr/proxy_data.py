from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REQUIRED_PROXY_COLUMNS = ["file_name", "audio_path", "text", "source", "split"]


def validate_proxy_manifest(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, keep_default_na=False)
    missing = [column for column in REQUIRED_PROXY_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Proxy manifest missing columns: {', '.join(missing)}")
    empty_text = df.loc[df["text"].astype(str) == "", "file_name"].tolist()
    if empty_text:
        raise ValueError(f"Proxy manifest has empty text: {', '.join(empty_text[:20])}")
    return df[REQUIRED_PROXY_COLUMNS].copy()


def degrade_manifest_for_call_center(df: pd.DataFrame) -> pd.DataFrame:
    degraded = df.copy()
    degraded["transform"] = "call_center_degraded"
    degraded["target_sample_rate"] = 16000
    degraded["bandpass_hz"] = "300-3400"
    degraded["mono"] = True
    degraded["gain_db"] = "-6,0,6"
    return degraded


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare proxy validation manifest metadata.")
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--clean-output", type=Path, required=True)
    parser.add_argument("--degraded-output", type=Path, required=True)
    args = parser.parse_args()
    clean = validate_proxy_manifest(args.input_manifest)
    degraded = degrade_manifest_for_call_center(clean)
    args.clean_output.parent.mkdir(parents=True, exist_ok=True)
    args.degraded_output.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(args.clean_output, index=False)
    degraded.to_csv(args.degraded_output, index=False)
    print(f"Wrote clean manifest: {args.clean_output}")
    print(f"Wrote degraded manifest: {args.degraded_output}")
    print(f"Rows: {len(clean)}")


if __name__ == "__main__":
    main()
