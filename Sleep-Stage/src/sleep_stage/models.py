from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold

from sleep_stage.config import LABELS, LABEL_TO_INDEX
from sleep_stage.context import feature_columns


@dataclass(frozen=True)
class CVReport:
    model_name: str
    feature_columns: list[str]
    fold_scores: list[float]
    weighted_f1_mean: float
    weighted_f1_std: float
    per_class_f1: pd.DataFrame
    confusion: np.ndarray
    oof_probabilities: np.ndarray


def build_model(model_name: str, random_state: int = 42):
    if model_name == "hgb":
        return HistGradientBoostingClassifier(max_iter=15, learning_rate=0.1, l2_regularization=0.05, random_state=random_state)
    if model_name == "extra_trees":
        return ExtraTreesClassifier(n_estimators=60, min_samples_leaf=2, class_weight="balanced", n_jobs=-1, random_state=random_state)
    if model_name == "random_forest":
        return RandomForestClassifier(n_estimators=60, min_samples_leaf=2, class_weight="balanced", n_jobs=-1, random_state=random_state)
    raise ValueError(f"unknown model name: {model_name}")


def _aligned_predict_proba(model, x: pd.DataFrame) -> np.ndarray:
    proba = model.predict_proba(x)
    aligned = np.zeros((len(x), len(LABELS)), dtype=float)
    for source_index, class_index in enumerate(model.classes_):
        aligned[:, int(class_index)] = proba[:, source_index]
    return aligned


def run_grouped_cv(table: pd.DataFrame, model_name: str = "hgb", n_splits: int = 5, random_state: int = 42) -> CVReport:
    columns = feature_columns(table)
    x = table[columns].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    y = table["label"].map(LABEL_TO_INDEX).to_numpy()
    groups = table["recording_id"].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    splits = min(n_splits, len(unique_groups))
    if splits < 2:
        raise ValueError("grouped CV requires at least two recording groups")

    oof = np.zeros((len(table), len(LABELS)), dtype=float)
    fold_scores: list[float] = []
    per_class_rows: list[dict[str, float]] = []
    predictions = np.full(len(table), -1, dtype=int)
    splitter = GroupKFold(n_splits=splits)
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(x, y, groups), start=1):
        model = build_model(model_name, random_state=random_state + fold)
        model.fit(x.iloc[train_idx], y[train_idx])
        probabilities = _aligned_predict_proba(model, x.iloc[valid_idx])
        oof[valid_idx] = probabilities
        pred = probabilities.argmax(axis=1)
        predictions[valid_idx] = pred
        fold_scores.append(float(f1_score(y[valid_idx], pred, average="weighted", labels=list(range(len(LABELS))), zero_division=0)))
        class_scores = f1_score(y[valid_idx], pred, average=None, labels=list(range(len(LABELS))), zero_division=0)
        per_class_rows.append({"fold": fold, **{label: float(class_scores[index]) for index, label in enumerate(LABELS)}})

    confusion = confusion_matrix(y, predictions, labels=list(range(len(LABELS))))
    return CVReport(
        model_name=model_name,
        feature_columns=columns,
        fold_scores=fold_scores,
        weighted_f1_mean=float(np.mean(fold_scores)),
        weighted_f1_std=float(np.std(fold_scores)),
        per_class_f1=pd.DataFrame(per_class_rows),
        confusion=confusion,
        oof_probabilities=oof,
    )


def fit_final_model(table: pd.DataFrame, model_name: str = "hgb", random_state: int = 42):
    columns = feature_columns(table)
    x = table[columns].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    y = table["label"].map(LABEL_TO_INDEX).to_numpy()
    model = build_model(model_name, random_state=random_state)
    model.fit(x, y)
    return model, columns


def predict_probabilities(model, table: pd.DataFrame, columns: list[str]) -> np.ndarray:
    x = table[columns].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return _aligned_predict_proba(model, x)
