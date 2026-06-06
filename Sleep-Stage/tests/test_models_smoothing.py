import numpy as np
import pandas as pd

from sleep_stage.config import LABELS
from sleep_stage.models import build_model, run_grouped_cv
from sleep_stage.smoothing import build_transition_log_probs, mode_filter_labels, viterbi_decode


def _model_table() -> pd.DataFrame:
    rows = []
    for group_index in range(5):
        for label_index, label in enumerate(LABELS):
            rows.append(
                {
                    "recording_id": f"g{group_index}",
                    "epoch_index": label_index,
                    "label": label,
                    "f1": float(label_index + group_index / 10),
                    "f2": float(label_index % 2),
                }
            )
    return pd.DataFrame(rows)


def test_build_model_returns_predict_proba_estimator():
    model = build_model("extra_trees")

    assert hasattr(model, "fit")
    assert hasattr(model, "predict_proba")


def test_run_grouped_cv_returns_fold_metrics_and_oof_probabilities():
    table = _model_table()

    report = run_grouped_cv(table, model_name="extra_trees", n_splits=5)

    assert len(report.fold_scores) == 5
    assert set(report.per_class_f1.columns) == {"fold", *LABELS}
    assert report.confusion.shape == (5, 5)
    assert report.feature_columns == ["f1", "f2"]
    assert report.oof_probabilities.shape == (len(table), 5)


def test_mode_filter_labels_removes_single_epoch_spike():
    assert mode_filter_labels(["N2", "N2", "W", "N2", "N2"], window=3) == ["N2", "N2", "N2", "N2", "N2"]


def test_viterbi_decode_uses_transition_counts_and_probability_shape():
    transition = build_transition_log_probs([["W", "N1", "N2", "N2", "R"]])
    probabilities = np.full((3, 5), 0.05)
    probabilities[:, 2] = 0.8
    decoded = viterbi_decode(probabilities, transition)

    assert transition.shape == (5, 5)
    assert len(decoded) == 3
    assert set(decoded).issubset(set(LABELS))
