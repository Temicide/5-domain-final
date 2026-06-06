from __future__ import annotations

import pandas as pd

METADATA_COLUMNS = ["id", "recording_id", "epoch_index", "label"]


def feature_columns(table: pd.DataFrame) -> list[str]:
    return [column for column in table.columns if column not in METADATA_COLUMNS]


def add_context_features(
    table: pd.DataFrame,
    lags: tuple[int, ...] = (1, 2, 3),
    rolling_windows: tuple[int, ...] = (3, 5, 9, 15),
) -> pd.DataFrame:
    if "recording_id" not in table.columns:
        raise ValueError("table must include recording_id for grouped context features")

    output = table.sort_values(["recording_id", "epoch_index"]).reset_index(drop=True).copy()
    base_columns = feature_columns(output)
    grouped = output.groupby("recording_id", sort=False, group_keys=False)
    additions: dict[str, pd.Series] = {}

    for column in base_columns:
        current = output[column]
        for lag in lags:
            prev = grouped[column].shift(lag).fillna(current)
            next_ = grouped[column].shift(-lag).fillna(current)
            additions[f"{column}_prev{lag}"] = prev
            additions[f"{column}_next{lag}"] = next_
            additions[f"{column}_delta_prev{lag}"] = current - prev
        for window in rolling_windows:
            rolled = grouped[column].rolling(window=window, center=True, min_periods=1)
            additions[f"{column}_roll{window}_mean"] = rolled.mean().reset_index(level=0, drop=True)
            additions[f"{column}_roll{window}_std"] = rolled.std().reset_index(level=0, drop=True).fillna(0.0)

    sizes = grouped["epoch_index"].transform("size")
    ranks = grouped.cumcount()
    additions["relative_position"] = (ranks / (sizes - 1).where(sizes > 1, 1)).astype(float)
    return pd.concat([output, pd.DataFrame(additions, index=output.index)], axis=1)
