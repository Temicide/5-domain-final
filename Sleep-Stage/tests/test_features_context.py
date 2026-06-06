import numpy as np
import pandas as pd

from sleep_stage.config import EPOCH_ROWS, SIGNAL_COLUMNS
from sleep_stage.context import add_context_features
from sleep_stage.features import extract_epoch_features, extract_feature_table


def _signal_array(offset: float = 0.0) -> np.ndarray:
    x = np.linspace(0, 2 * np.pi, EPOCH_ROWS)
    columns = [
        np.sin(x) + offset,
        np.cos(x),
        np.sin(2 * x),
        np.cos(2 * x),
        np.full(EPOCH_ROWS, 32.0 + offset),
        np.linspace(0.1, 0.3, EPOCH_ROWS),
        np.full(EPOCH_ROWS, 60.0 + offset),
        np.full(EPOCH_ROWS, 1.0),
    ]
    return np.column_stack(columns)


def test_extract_epoch_features_has_deterministic_finite_values():
    features = extract_epoch_features(_signal_array())
    keys = list(features)

    assert keys == sorted(keys)
    assert np.isfinite(list(features.values())).all()
    assert "BVP_fft_peak_hz" in features
    assert "ACC_mag_mean" in features
    assert "HR_IBI_consistency" in features


def test_extract_feature_table_returns_metadata_and_sorted_feature_columns():
    epochs = pd.DataFrame(
        {
            "id": ["test001_00000"],
            "recording_id": ["test001"],
            "epoch_index": [0],
            "label": ["W"],
            "signals": [_signal_array()],
        }
    )

    table = extract_feature_table(epochs)
    feature_columns = [column for column in table.columns if column not in {"id", "recording_id", "epoch_index", "label"}]

    assert table.loc[0, "id"] == "test001_00000"
    assert feature_columns == sorted(feature_columns)
    assert set(SIGNAL_COLUMNS).isdisjoint(feature_columns)


def test_add_context_features_preserves_rows_and_adds_temporal_columns():
    table = pd.DataFrame(
        {
            "recording_id": ["a", "a", "a", "b"],
            "epoch_index": [0, 1, 2, 0],
            "label": ["W", "N1", "N2", "R"],
            "feat": [1.0, 3.0, 5.0, 10.0],
        }
    )

    context = add_context_features(table, lags=(1,), rolling_windows=(3,))

    assert len(context) == len(table)
    assert "feat_prev1" in context.columns
    assert "feat_next1" in context.columns
    assert "feat_roll3_mean" in context.columns
    assert "feat_roll3_std" in context.columns
    assert context.loc[0, "feat_prev1"] == context.loc[0, "feat"]
    assert context.loc[2, "feat_next1"] == context.loc[2, "feat"]
    assert context.loc[3, "feat_prev1"] == context.loc[3, "feat"]
    assert context.loc[0, "relative_position"] == 0.0
    assert context.loc[2, "relative_position"] == 1.0
