from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

from sleep_stage.config import EPOCH_ROWS, INDEX_TO_LABEL, LABELS, LABEL_TO_INDEX, ProjectPaths, SIGNAL_COLUMNS
from sleep_stage.data import (
    as_numeric_signal_array,
    iter_train_recordings,
    list_test_segment_paths,
    read_test_segments,
    validate_submission,
    write_submission_from_labels,
)
from sleep_stage.smoothing import build_transition_log_probs, viterbi_decode

try:  # Optional locally; Colab H100 runtimes already provide PyTorch.
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
except Exception:  # pragma: no cover - exercised only when torch is absent.
    torch = None
    nn = None
    F = None
    DataLoader = object
    Dataset = object


class TorchUnavailableError(RuntimeError):
    pass


def require_torch() -> Any:
    if torch is None:
        raise TorchUnavailableError("PyTorch is required for the H100 raw-signal pipeline.")
    return torch


@dataclass(frozen=True)
class RawEpochArrays:
    signals: np.ndarray
    recording_ids: np.ndarray
    epoch_indices: np.ndarray
    labels: np.ndarray | None = None
    ids: np.ndarray | None = None

    def subset(self, indices: np.ndarray) -> "RawEpochArrays":
        return RawEpochArrays(
            signals=self.signals[indices],
            recording_ids=self.recording_ids[indices],
            epoch_indices=self.epoch_indices[indices],
            labels=None if self.labels is None else self.labels[indices],
            ids=None if self.ids is None else self.ids[indices],
        )

    def with_signals(self, signals: np.ndarray) -> "RawEpochArrays":
        return replace(self, signals=signals)


@dataclass(frozen=True)
class DeepTrainingConfig:
    context_epochs: int = 11
    batch_size: int = 512
    epochs: int = 18
    lr: float = 3e-4
    weight_decay: float = 1e-2
    width: int = 256
    transformer_layers: int = 4
    transformer_heads: int = 8
    dropout: float = 0.15
    label_smoothing: float = 0.03
    focal_gamma: float = 1.0
    noise_std: float = 0.015
    time_mask_prob: float = 0.15
    channel_dropout_prob: float = 0.08
    num_workers: int = 2
    use_compile: bool = True
    amp_dtype: str = "bf16"
    random_state: int = 42
    device: str = "auto"


@dataclass(frozen=True)
class DeepFoldResult:
    fold: int
    best_weighted_f1: float
    history: pd.DataFrame
    valid_probabilities: np.ndarray
    valid_labels: np.ndarray
    valid_indices: np.ndarray
    checkpoint_path: Path | None


def configure_h100_runtime() -> None:
    torch_mod = require_torch()
    if torch_mod.cuda.is_available():
        torch_mod.backends.cuda.matmul.allow_tf32 = True
        torch_mod.backends.cudnn.allow_tf32 = True
        torch_mod.backends.cudnn.benchmark = True
    torch_mod.set_float32_matmul_precision("high")


def epoch_table_to_raw_arrays(table: pd.DataFrame) -> RawEpochArrays:
    signals = np.stack([as_numeric_signal_array(value).astype(np.float32) for value in table["signals"]])
    labels = None
    if "label" in table.columns:
        labels = table["label"].astype(str).map(LABEL_TO_INDEX).astype(np.int64).to_numpy()
    ids = table["id"].astype(str).to_numpy() if "id" in table.columns else None
    return RawEpochArrays(
        signals=signals,
        recording_ids=table["recording_id"].astype(str).to_numpy(),
        epoch_indices=table["epoch_index"].astype(np.int64).to_numpy(),
        labels=labels,
        ids=ids,
    )


def load_or_build_raw_arrays(paths: ProjectPaths, force: bool = False) -> tuple[RawEpochArrays, RawEpochArrays]:
    paths.cache_dir.mkdir(parents=True, exist_ok=True)
    train_path = paths.cache_dir / "train_raw_epochs.joblib"
    test_path = paths.cache_dir / "test_raw_epochs.joblib"
    if not force and train_path.exists() and test_path.exists():
        return joblib.load(train_path), joblib.load(test_path)

    train_epochs = pd.concat(list(iter_train_recordings(paths.train_dir)), ignore_index=True)
    test_segments = read_test_segments(paths.sample_submission, list_test_segment_paths(paths.test_dir))
    train_arrays = epoch_table_to_raw_arrays(train_epochs)
    test_arrays = epoch_table_to_raw_arrays(test_segments)
    joblib.dump(train_arrays, train_path, compress=0)
    joblib.dump(test_arrays, test_path, compress=0)
    return train_arrays, test_arrays


def normalize_per_recording(signals: np.ndarray, recording_ids: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    normalized = signals.astype(np.float32, copy=True)
    for recording_id in np.unique(recording_ids):
        mask = recording_ids == recording_id
        flat = normalized[mask].reshape(-1, len(SIGNAL_COLUMNS))
        mean = flat.mean(axis=0, keepdims=True)
        std = flat.std(axis=0, keepdims=True)
        normalized[mask] = ((flat - mean) / np.maximum(std, eps)).reshape(normalized[mask].shape)
    return normalized


def build_window_indices(recording_ids: np.ndarray, epoch_indices: np.ndarray, context_epochs: int) -> np.ndarray:
    if context_epochs % 2 != 1:
        raise ValueError("context_epochs must be odd so each window has a center epoch")
    radius = context_epochs // 2
    order = np.lexsort((epoch_indices, recording_ids))
    windows = np.empty((len(recording_ids), context_epochs), dtype=np.int64)
    for recording_id in np.unique(recording_ids):
        group = order[recording_ids[order] == recording_id]
        group = group[np.argsort(epoch_indices[group])]
        for position, center_index in enumerate(group):
            padded_positions = np.clip(np.arange(position - radius, position + radius + 1), 0, len(group) - 1)
            windows[center_index] = group[padded_positions]
    return windows


class RawWindowDataset(Dataset):
    def __init__(self, arrays: RawEpochArrays, context_epochs: int, augment: bool = False, config: DeepTrainingConfig | None = None):
        require_torch()
        self.arrays = arrays
        self.window_indices = build_window_indices(arrays.recording_ids, arrays.epoch_indices, context_epochs)
        self.augment = augment
        self.config = config or DeepTrainingConfig(context_epochs=context_epochs)

    def __len__(self) -> int:
        return len(self.arrays.signals)

    def __getitem__(self, index: int):
        window = self.arrays.signals[self.window_indices[index]].reshape(-1, len(SIGNAL_COLUMNS))
        x = torch.from_numpy(window.T.astype(np.float32, copy=False))
        if self.augment:
            x = self._augment(x)
        if self.arrays.labels is None:
            return x, index
        return x, torch.tensor(int(self.arrays.labels[index]), dtype=torch.long)

    def _augment(self, x):
        if self.config.noise_std > 0:
            x = x + torch.randn_like(x) * self.config.noise_std
        if self.config.channel_dropout_prob > 0 and torch.rand(()) < self.config.channel_dropout_prob:
            channel = torch.randint(0, x.shape[0], (1,)).item()
            x[channel] = 0.0
        if self.config.time_mask_prob > 0 and torch.rand(()) < self.config.time_mask_prob:
            width = max(8, x.shape[1] // 32)
            start = torch.randint(0, max(1, x.shape[1] - width + 1), (1,)).item()
            x[:, start : start + width] = 0.0
        return x


def _sinusoidal_positions(length: int, dim: int, device):
    position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, dim, 2, device=device, dtype=torch.float32) * (-np.log(10000.0) / dim))
    pe = torch.zeros(length, dim, device=device, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(position * div)
    pe[:, 1::2] = torch.cos(position * div)
    return pe.unsqueeze(0)


class ConvTransformerSleepNet(nn.Module):
    def __init__(
        self,
        in_channels: int = len(SIGNAL_COLUMNS),
        n_classes: int = len(LABELS),
        width: int = 256,
        transformer_layers: int = 4,
        transformer_heads: int = 8,
        dropout: float = 0.15,
    ):
        require_torch()
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, width // 2, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(width // 2),
            nn.GELU(),
            nn.Conv1d(width // 2, width, kernel_size=9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(width),
            nn.GELU(),
            nn.Conv1d(width, width, kernel_size=7, stride=2, padding=3, groups=4, bias=False),
            nn.BatchNorm1d(width),
            nn.GELU(),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=transformer_heads,
            dim_feedforward=width * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=transformer_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(width * 2),
            nn.Dropout(dropout),
            nn.Linear(width * 2, width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, n_classes),
        )

    def forward(self, x):
        tokens = self.stem(x).transpose(1, 2)
        tokens = tokens + _sinusoidal_positions(tokens.shape[1], tokens.shape[2], tokens.device).to(tokens.dtype)
        encoded = self.encoder(tokens)
        center = encoded[:, encoded.shape[1] // 2]
        pooled = encoded.mean(dim=1)
        return self.classifier(torch.cat([center, pooled], dim=1))


class WeightedFocalLoss(nn.Module):
    def __init__(self, class_weights, gamma: float = 1.0, label_smoothing: float = 0.0):
        require_torch()
        super().__init__()
        self.register_buffer("class_weights", class_weights)
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        ce = F.cross_entropy(
            logits,
            targets,
            weight=self.class_weights,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        if self.gamma <= 0:
            return ce.mean()
        pt = torch.exp(-ce.detach())
        return ((1.0 - pt) ** self.gamma * ce).mean()


def class_weights_from_labels(labels: np.ndarray):
    require_torch()
    counts = np.bincount(labels.astype(int), minlength=len(LABELS)).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def _resolve_device(config: DeepTrainingConfig):
    torch_mod = require_torch()
    if config.device != "auto":
        return torch_mod.device(config.device)
    return torch_mod.device("cuda" if torch_mod.cuda.is_available() else "cpu")


def _amp_dtype(config: DeepTrainingConfig):
    return torch.bfloat16 if config.amp_dtype == "bf16" else torch.float16


def _make_loader(dataset: Dataset, config: DeepTrainingConfig, shuffle: bool) -> DataLoader:
    device = _resolve_device(config)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=config.num_workers > 0,
        drop_last=shuffle,
    )


def _predict(model, arrays: RawEpochArrays, config: DeepTrainingConfig) -> tuple[np.ndarray, np.ndarray | None]:
    device = _resolve_device(config)
    dataset = RawWindowDataset(arrays, context_epochs=config.context_epochs, augment=False, config=config)
    loader = _make_loader(dataset, config, shuffle=False)
    probabilities: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for x, y_or_index in loader:
            x = x.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=_amp_dtype(config), enabled=device.type == "cuda"):
                logits = model(x)
            probabilities.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
            if arrays.labels is not None:
                labels.append(y_or_index.numpy())
    y_true = np.concatenate(labels) if labels else None
    return np.concatenate(probabilities), y_true


def fit_deep_model(
    train_arrays: RawEpochArrays,
    config: DeepTrainingConfig,
    valid_arrays: RawEpochArrays | None = None,
    output_dir: Path | None = None,
    fold: int | None = None,
):
    configure_h100_runtime()
    if train_arrays.labels is None:
        raise ValueError("train_arrays must include labels")
    device = _resolve_device(config)
    torch.manual_seed(config.random_state + (fold or 0))
    train_dataset = RawWindowDataset(train_arrays, context_epochs=config.context_epochs, augment=True, config=config)
    train_loader = _make_loader(train_dataset, config, shuffle=True)

    model = ConvTransformerSleepNet(
        width=config.width,
        transformer_layers=config.transformer_layers,
        transformer_heads=config.transformer_heads,
        dropout=config.dropout,
    ).to(device)
    if config.use_compile and device.type == "cuda" and hasattr(torch, "compile"):
        model = torch.compile(model)

    criterion = WeightedFocalLoss(
        class_weights_from_labels(train_arrays.labels).to(device),
        gamma=config.focal_gamma,
        label_smoothing=config.label_smoothing,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, config.epochs))
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda" and config.amp_dtype == "fp16")

    output_dir = Path(output_dir) if output_dir is not None else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = None if output_dir is None else output_dir / (f"fold{fold}_best.pt" if fold is not None else "final.pt")
    best_metric = -np.inf
    history: list[dict[str, float]] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        losses: list[float] = []
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=_amp_dtype(config), enabled=device.type == "cuda"):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()

        row = {"epoch": float(epoch), "train_loss": float(np.mean(losses))}
        metric = -row["train_loss"]
        if valid_arrays is not None:
            valid_probabilities, valid_labels = _predict(model, valid_arrays, config)
            predictions = valid_probabilities.argmax(axis=1)
            metric = float(f1_score(valid_labels, predictions, average="weighted", labels=list(range(len(LABELS))), zero_division=0))
            row["valid_weighted_f1"] = metric
        history.append(row)

        if metric > best_metric:
            best_metric = metric
            if checkpoint_path is not None:
                torch.save({"model": model.state_dict(), "config": config}, checkpoint_path)

    return model, pd.DataFrame(history), checkpoint_path, float(best_metric)


def train_deep_fold(
    arrays: RawEpochArrays,
    train_indices: np.ndarray,
    valid_indices: np.ndarray,
    config: DeepTrainingConfig,
    output_dir: Path,
    fold: int,
) -> DeepFoldResult:
    train_arrays = arrays.subset(train_indices)
    valid_arrays = arrays.subset(valid_indices)
    train_arrays = train_arrays.with_signals(normalize_per_recording(train_arrays.signals, train_arrays.recording_ids))
    valid_arrays = valid_arrays.with_signals(normalize_per_recording(valid_arrays.signals, valid_arrays.recording_ids))
    model, history, checkpoint_path, best_metric = fit_deep_model(
        train_arrays=train_arrays,
        valid_arrays=valid_arrays,
        config=config,
        output_dir=output_dir,
        fold=fold,
    )
    valid_probabilities, valid_labels = _predict(model, valid_arrays, config)
    return DeepFoldResult(
        fold=fold,
        best_weighted_f1=best_metric,
        history=history,
        valid_probabilities=valid_probabilities,
        valid_labels=valid_labels,
        valid_indices=valid_indices,
        checkpoint_path=checkpoint_path,
    )


def run_deep_grouped_cv(
    paths: ProjectPaths,
    config: DeepTrainingConfig | None = None,
    n_splits: int = 5,
    folds: list[int] | None = None,
    force_raw: bool = False,
) -> pd.DataFrame:
    config = config or DeepTrainingConfig()
    train_arrays, _ = load_or_build_raw_arrays(paths, force=force_raw)
    splitter = GroupKFold(n_splits=min(n_splits, len(np.unique(train_arrays.recording_ids))))
    requested = set(range(n_splits) if folds is None else folds)
    rows: list[dict[str, object]] = []
    output_dir = paths.working_dir / "deep"
    for fold, (train_idx, valid_idx) in enumerate(
        splitter.split(train_arrays.signals, train_arrays.labels, train_arrays.recording_ids)
    ):
        if fold not in requested:
            continue
        result = train_deep_fold(train_arrays, train_idx, valid_idx, config, output_dir=output_dir, fold=fold)
        result.history.to_csv(output_dir / f"fold{fold}_history.csv", index=False)
        np.save(output_dir / f"fold{fold}_valid_probabilities.npy", result.valid_probabilities)
        rows.append(
            {
                "fold": fold,
                "best_weighted_f1": result.best_weighted_f1,
                "checkpoint_path": "" if result.checkpoint_path is None else str(result.checkpoint_path),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "cv_summary.csv", index=False)
    return summary


def train_final_deep_submission(
    paths: ProjectPaths,
    config: DeepTrainingConfig | None = None,
    force_raw: bool = False,
) -> Path:
    config = config or DeepTrainingConfig()
    train_arrays, test_arrays = load_or_build_raw_arrays(paths, force=force_raw)
    normalized_train = train_arrays.with_signals(normalize_per_recording(train_arrays.signals, train_arrays.recording_ids))
    normalized_test = test_arrays.with_signals(normalize_per_recording(test_arrays.signals, test_arrays.recording_ids))
    output_dir = paths.working_dir / "deep"
    model, history, _, _ = fit_deep_model(normalized_train, config=config, valid_arrays=None, output_dir=output_dir)
    history.to_csv(output_dir / "final_history.csv", index=False)
    probabilities, _ = _predict(model, normalized_test, config)

    train_sequences = []
    for recording_id in np.unique(train_arrays.recording_ids):
        indices = np.where(train_arrays.recording_ids == recording_id)[0]
        indices = indices[np.argsort(train_arrays.epoch_indices[indices])]
        train_sequences.append([INDEX_TO_LABEL[int(label)] for label in train_arrays.labels[indices]])
    transition = build_transition_log_probs(train_sequences)

    labels_by_id: dict[str, str] = {}
    for recording_id in np.unique(test_arrays.recording_ids):
        indices = np.where(test_arrays.recording_ids == recording_id)[0]
        indices = indices[np.argsort(test_arrays.epoch_indices[indices])]
        decoded = viterbi_decode(probabilities[indices], transition)
        for segment_id, label in zip(test_arrays.ids[indices].astype(str), decoded):
            labels_by_id[segment_id] = label

    output_path = write_submission_from_labels(labels_by_id, paths.sample_submission, paths.output_path)
    validate_submission(output_path, paths.sample_submission)
    return output_path


__all__ = [
    "ConvTransformerSleepNet",
    "DeepTrainingConfig",
    "RawEpochArrays",
    "RawWindowDataset",
    "TorchUnavailableError",
    "build_window_indices",
    "class_weights_from_labels",
    "configure_h100_runtime",
    "epoch_table_to_raw_arrays",
    "load_or_build_raw_arrays",
    "normalize_per_recording",
    "run_deep_grouped_cv",
    "train_final_deep_submission",
]
