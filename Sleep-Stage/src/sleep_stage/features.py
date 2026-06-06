from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from sleep_stage.config import SAMPLING_HZ, SIGNAL_COLUMNS
from sleep_stage.data import as_numeric_signal_array


def _safe_float(value: float) -> float:
    if np.isfinite(value):
        return float(value)
    return 0.0


def _channel_stats(values: np.ndarray, prefix: str) -> dict[str, float]:
    diffs = np.diff(values)
    x = np.arange(len(values), dtype=float)
    is_constant = np.allclose(values, values[0])
    if is_constant:
        slope = 0.0
    else:
        slope = np.polyfit(x, values, 1)[0]
    centered = values - np.nanmean(values)
    zero_crossings = np.mean(np.diff(np.signbit(centered)) != 0) if len(values) > 1 else 0.0
    q25, q75 = np.nanquantile(values, [0.25, 0.75])
    return {
        f"{prefix}_mean": _safe_float(np.nanmean(values)),
        f"{prefix}_std": _safe_float(np.nanstd(values)),
        f"{prefix}_min": _safe_float(np.nanmin(values)),
        f"{prefix}_max": _safe_float(np.nanmax(values)),
        f"{prefix}_median": _safe_float(np.nanmedian(values)),
        f"{prefix}_q25": _safe_float(q25),
        f"{prefix}_q75": _safe_float(q75),
        f"{prefix}_iqr": _safe_float(q75 - q25),
        f"{prefix}_skew": 0.0 if is_constant else _safe_float(stats.skew(values, nan_policy="omit")),
        f"{prefix}_kurtosis": 0.0 if is_constant else _safe_float(stats.kurtosis(values, nan_policy="omit")),
        f"{prefix}_range": _safe_float(np.nanmax(values) - np.nanmin(values)),
        f"{prefix}_slope": _safe_float(slope),
        f"{prefix}_madiff_mean": _safe_float(np.nanmean(np.abs(diffs))) if len(diffs) else 0.0,
        f"{prefix}_madiff_max": _safe_float(np.nanmax(np.abs(diffs))) if len(diffs) else 0.0,
        f"{prefix}_zero_crossing_rate": _safe_float(zero_crossings),
    }


def _band_power(freqs: np.ndarray, power: np.ndarray, low: float, high: float) -> float:
    mask = (freqs >= low) & (freqs < high)
    if not np.any(mask):
        return 0.0
    return _safe_float(power[mask].sum())


def extract_epoch_features(signals: object) -> dict[str, float]:
    array = as_numeric_signal_array(signals)
    features: dict[str, float] = {}
    by_name = {column: array[:, index] for index, column in enumerate(SIGNAL_COLUMNS)}
    for column in SIGNAL_COLUMNS:
        features.update(_channel_stats(by_name[column], column))

    acc = np.sqrt(by_name["ACC_X"] ** 2 + by_name["ACC_Y"] ** 2 + by_name["ACC_Z"] ** 2)
    jerk = np.abs(np.diff(acc))
    features.update(_channel_stats(acc, "ACC_mag"))
    features["ACC_jerk_mean"] = _safe_float(np.mean(jerk)) if len(jerk) else 0.0
    features["ACC_jerk_max"] = _safe_float(np.max(jerk)) if len(jerk) else 0.0

    bvp = by_name["BVP"] - np.mean(by_name["BVP"])
    freqs = np.fft.rfftfreq(len(bvp), d=1.0 / SAMPLING_HZ)
    power = np.abs(np.fft.rfft(bvp)) ** 2
    total_power = power[1:].sum() + 1e-12
    nonzero = power.copy()
    if len(nonzero):
        nonzero[0] = 0.0
    features["BVP_fft_peak_hz"] = _safe_float(freqs[int(np.argmax(nonzero))])
    for low, high, name in [(0.04, 0.15, "low"), (0.15, 0.4, "mid"), (0.4, 2.0, "high"), (2.0, 8.0, "very_high")]:
        features[f"BVP_power_{name}_ratio"] = _safe_float(_band_power(freqs, power, low, high) / total_power)

    hr = by_name["HR"]
    ibi = by_name["IBI"]
    hr_from_ibi = 60.0 / np.clip(ibi, 1e-6, None)
    features["IBI_rmssd"] = _safe_float(np.sqrt(np.mean(np.diff(ibi) ** 2))) if len(ibi) > 1 else 0.0
    features["HR_rmssd"] = _safe_float(np.sqrt(np.mean(np.diff(hr) ** 2))) if len(hr) > 1 else 0.0
    features["HR_IBI_consistency"] = _safe_float(np.mean(np.abs(hr - hr_from_ibi)))
    features["ACC_mag_x_HR"] = _safe_float(np.mean(acc) * np.mean(hr))
    features["EDA_x_TEMP"] = _safe_float(np.mean(by_name["EDA"]) * np.mean(by_name["TEMP"]))

    return {key: features[key] for key in sorted(features)}


def extract_feature_table(epoch_table: pd.DataFrame) -> pd.DataFrame:
    metadata_columns = [column for column in ["id", "recording_id", "epoch_index", "label"] if column in epoch_table.columns]
    metadata = epoch_table[metadata_columns].reset_index(drop=True).copy()
    feature_rows = [extract_epoch_features(signals) for signals in epoch_table["signals"]]
    features = pd.DataFrame(feature_rows)
    ordered_features = sorted(features.columns)
    return pd.concat([metadata, features[ordered_features]], axis=1)
