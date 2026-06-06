import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sleep_stage.config import EPOCH_ROWS, LABELS, SIGNAL_COLUMNS
from sleep_stage.deep import (
    ConvTransformerSleepNet,
    DeepTrainingConfig,
    RawEpochArrays,
    RawWindowDataset,
    build_window_indices,
    class_weights_from_labels,
    normalize_per_recording,
)


def _arrays() -> RawEpochArrays:
    signals = np.zeros((6, EPOCH_ROWS, len(SIGNAL_COLUMNS)), dtype=np.float32)
    for index in range(len(signals)):
        signals[index] = index + np.linspace(0, 1, EPOCH_ROWS, dtype=np.float32)[:, None]
    return RawEpochArrays(
        signals=signals,
        recording_ids=np.array(["a", "a", "a", "b", "b", "b"]),
        epoch_indices=np.array([0, 1, 2, 0, 1, 2]),
        labels=np.array([0, 1, 2, 3, 4, 0]),
    )


def test_build_window_indices_pads_within_recording_boundaries():
    arrays = _arrays()

    windows = build_window_indices(arrays.recording_ids, arrays.epoch_indices, context_epochs=3)

    assert windows[0].tolist() == [0, 0, 1]
    assert windows[1].tolist() == [0, 1, 2]
    assert windows[2].tolist() == [1, 2, 2]
    assert windows[3].tolist() == [3, 3, 4]


def test_normalize_per_recording_returns_finite_zero_centered_values():
    arrays = _arrays()

    normalized = normalize_per_recording(arrays.signals, arrays.recording_ids)

    assert np.isfinite(normalized).all()
    for recording_id in np.unique(arrays.recording_ids):
        flat = normalized[arrays.recording_ids == recording_id].reshape(-1, len(SIGNAL_COLUMNS))
        assert np.allclose(flat.mean(axis=0), 0.0, atol=1e-5)


def test_raw_window_dataset_returns_channels_by_time_tensor():
    arrays = _arrays()
    dataset = RawWindowDataset(arrays, context_epochs=3, config=DeepTrainingConfig(context_epochs=3))

    x, y = dataset[1]

    assert x.shape == (len(SIGNAL_COLUMNS), EPOCH_ROWS * 3)
    assert y.item() == 1


def test_conv_transformer_forward_shape():
    model = ConvTransformerSleepNet(width=32, transformer_layers=1, transformer_heads=4, dropout=0.0)
    x = torch.randn(2, len(SIGNAL_COLUMNS), EPOCH_ROWS * 3)

    logits = model(x)

    assert logits.shape == (2, len(LABELS))


def test_class_weights_from_labels_returns_one_weight_per_class():
    weights = class_weights_from_labels(np.array([0, 1, 1, 2, 2, 2]))

    assert weights.shape == (len(LABELS),)
    assert torch.isfinite(weights).all()
