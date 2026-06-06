from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from .config import DISEASE_LABEL_COLUMNS, LABEL_COLUMNS


def tune_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    labels: list[str],
    grid: np.ndarray | None = None,
) -> dict[str, float]:
    thresholds = grid if grid is not None else np.linspace(0.05, 0.95, 91)
    best: dict[str, float] = {}
    for index, label in enumerate(labels):
        scores = []
        for threshold in thresholds:
            pred = (y_prob[:, index] >= threshold).astype(int)
            scores.append(f1_score(y_true[:, index], pred, zero_division=0))
        best[label] = float(thresholds[int(np.argmax(scores))])
    return best


def apply_no_finding_consistency(predictions: pd.DataFrame) -> pd.DataFrame:
    fixed = predictions.copy()
    disease_positive = fixed[DISEASE_LABEL_COLUMNS].sum(axis=1) > 0
    fixed.loc[disease_positive, "No Finding"] = 0
    fixed.loc[~disease_positive, "No Finding"] = 1
    return fixed[LABEL_COLUMNS]


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    labels: list[str],
    thresholds: dict[str, float],
) -> dict[str, object]:
    pred = np.zeros_like(y_prob, dtype=int)
    for index, label in enumerate(labels):
        pred[:, index] = (y_prob[:, index] >= thresholds[label]).astype(int)
    per_label = {}
    for index, label in enumerate(labels):
        try:
            auc = float(roc_auc_score(y_true[:, index], y_prob[:, index]))
        except ValueError:
            auc = float("nan")
        try:
            ap = float(average_precision_score(y_true[:, index], y_prob[:, index]))
        except ValueError:
            ap = float("nan")
        per_label[label] = {
            "f1": float(f1_score(y_true[:, index], pred[:, index], zero_division=0)),
            "roc_auc": auc,
            "average_precision": ap,
        }
    return {
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, pred, average="micro", zero_division=0)),
        "mean_roc_auc": float(np.nanmean([item["roc_auc"] for item in per_label.values()])),
        "macro_average_precision": float(np.nanmean([item["average_precision"] for item in per_label.values()])),
        "per_label": per_label,
    }
