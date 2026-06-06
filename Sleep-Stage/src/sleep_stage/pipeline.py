from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sleep_stage.config import INDEX_TO_LABEL, ProjectPaths, build_paths
from sleep_stage.context import add_context_features
from sleep_stage.data import (
    iter_train_recordings,
    list_test_segment_paths,
    read_test_segments,
    validate_submission,
    write_submission_from_labels,
)
from sleep_stage.features import extract_feature_table
from sleep_stage.models import fit_final_model, predict_probabilities, run_grouped_cv
from sleep_stage.smoothing import build_transition_log_probs, mode_filter_labels, viterbi_decode

MODEL_CONTEXT_LAGS = (1,)
MODEL_CONTEXT_WINDOWS = (3,)


def build_train_feature_table(paths: ProjectPaths) -> pd.DataFrame:
    epoch_tables = list(iter_train_recordings(paths.train_dir))
    if not epoch_tables:
        raise FileNotFoundError(f"no training CSV files found in {paths.train_dir}")
    epochs = pd.concat(epoch_tables, ignore_index=True)
    return extract_feature_table(epochs)


def build_test_feature_table(paths: ProjectPaths) -> pd.DataFrame:
    segment_paths = list_test_segment_paths(paths.test_dir)
    if not segment_paths:
        raise FileNotFoundError(f"no test segment CSV files found in {paths.test_dir}")
    segments = read_test_segments(paths.sample_submission, segment_paths)
    return extract_feature_table(segments)


def load_or_build_features(paths: ProjectPaths, force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths.cache_dir.mkdir(parents=True, exist_ok=True)
    train_path = paths.cache_dir / "train_features.joblib"
    test_path = paths.cache_dir / "test_features.joblib"
    if not force and train_path.exists() and test_path.exists():
        return joblib.load(train_path), joblib.load(test_path)

    train_table = build_train_feature_table(paths)
    test_table = build_test_feature_table(paths)
    joblib.dump(train_table, train_path)
    joblib.dump(test_table, test_path)
    return train_table, test_table


def train_sequences_from_table(table: pd.DataFrame) -> list[list[str]]:
    sequences: list[list[str]] = []
    for _, group in table.sort_values(["recording_id", "epoch_index"]).groupby("recording_id", sort=False):
        sequences.append(group["label"].astype(str).tolist())
    return sequences


def _maybe_context(train_table: pd.DataFrame, test_table: pd.DataFrame, use_context: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not use_context:
        return train_table, test_table
    return (
        add_context_features(train_table, lags=MODEL_CONTEXT_LAGS, rolling_windows=MODEL_CONTEXT_WINDOWS),
        add_context_features(test_table, lags=MODEL_CONTEXT_LAGS, rolling_windows=MODEL_CONTEXT_WINDOWS),
    )


def predict_labels_by_segment(
    train_table: pd.DataFrame,
    test_table: pd.DataFrame,
    model_name: str = "hgb",
    use_context: bool = True,
    use_viterbi: bool = True,
) -> dict[str, str]:
    train_model_table, test_model_table = _maybe_context(train_table, test_table, use_context)
    model, columns = fit_final_model(train_model_table, model_name=model_name)
    probabilities = predict_probabilities(model, test_model_table, columns)
    predicted = probabilities.argmax(axis=1)
    test_ordered = test_model_table.reset_index(drop=True).copy()
    test_ordered["predicted_label"] = [INDEX_TO_LABEL[int(index)] for index in predicted]

    labels_by_id: dict[str, str] = {}
    transition = build_transition_log_probs(train_sequences_from_table(train_table))
    for _, group in test_ordered.groupby("recording_id", sort=False):
        group = group.sort_values("epoch_index")
        if use_viterbi:
            decoded = viterbi_decode(probabilities[group.index.to_numpy()], transition)
        else:
            decoded = group["predicted_label"].astype(str).tolist()
        smoothed = mode_filter_labels(decoded, window=3)
        for segment_id, label in zip(group["id"].astype(str), smoothed):
            labels_by_id[segment_id] = label
    return labels_by_id


def run_experiments(paths: ProjectPaths | None = None, force_features: bool = False) -> pd.DataFrame:
    paths = paths or build_paths()
    train_table, test_table = load_or_build_features(paths, force=force_features)
    _ = test_table
    experiment_specs = [
        ("context_hgb", "hgb", True),
        ("context_extra_trees", "extra_trees", True),
        ("static_hgb", "hgb", False),
        ("static_extra_trees", "extra_trees", False),
    ]
    rows: list[dict[str, object]] = []
    paths.cache_dir.mkdir(parents=True, exist_ok=True)
    for experiment, model_name, use_context in experiment_specs:
        model_table = (
            add_context_features(train_table, lags=MODEL_CONTEXT_LAGS, rolling_windows=MODEL_CONTEXT_WINDOWS)
            if use_context
            else train_table
        )
        report = run_grouped_cv(model_table, model_name=model_name)
        joblib.dump(report, paths.cache_dir / f"{experiment}_cv_report.joblib")
        rows.append(
            {
                "experiment": experiment,
                "model_name": model_name,
                "use_context": use_context,
                "n_features": len(report.feature_columns),
                "weighted_f1_mean": report.weighted_f1_mean,
                "weighted_f1_std": report.weighted_f1_std,
            }
        )
    results = pd.DataFrame(rows)
    results.to_csv(paths.cache_dir / "experiment_results.csv", index=False)
    return results


def generate_submission(paths: ProjectPaths | None = None) -> Path:
    paths = paths or build_paths()
    train_table, test_table = load_or_build_features(paths, force=False)
    labels_by_id = predict_labels_by_segment(train_table, test_table, model_name="hgb", use_context=True, use_viterbi=True)
    output_path = write_submission_from_labels(labels_by_id, paths.sample_submission, paths.output_path)
    validation = validate_submission(output_path, paths.sample_submission)
    labels = sorted(set(pd.read_csv(output_path)["labels"].astype(str)))
    print(f"Validated submission: rows={validation.rows}, labels={labels}")
    return output_path


__all__ = [
    "build_train_feature_table",
    "build_test_feature_table",
    "load_or_build_features",
    "train_sequences_from_table",
    "predict_labels_by_segment",
    "write_submission_from_labels",
    "run_experiments",
    "generate_submission",
]
