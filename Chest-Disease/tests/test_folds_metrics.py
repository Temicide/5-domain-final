import numpy as np
import pandas as pd

from chest_disease.config import DISEASE_LABEL_COLUMNS, EXPECTED_COLUMNS, LABEL_COLUMNS
from chest_disease.folds import make_multilabel_folds
from chest_disease.metrics import apply_no_finding_consistency, compute_metrics, tune_thresholds


def make_frame(n: int = 30) -> pd.DataFrame:
    rows = []
    for i in range(n):
        labels = [0] * len(LABEL_COLUMNS)
        if i % 3 == 0:
            labels[LABEL_COLUMNS.index("No Finding")] = 1
        else:
            labels[i % len(DISEASE_LABEL_COLUMNS)] = 1
        rows.append([f"cxr{i:05d}.jpg"] + labels)
    return pd.DataFrame(rows, columns=EXPECTED_COLUMNS)


def test_make_multilabel_folds_is_deterministic_and_complete():
    frame = make_frame(30)
    folded = make_multilabel_folds(frame, num_folds=3, seed=7)
    assert sorted(folded["fold"].unique().tolist()) == [0, 1, 2]
    assert folded["fold"].notna().all()
    assert folded["filename"].tolist() == frame["filename"].tolist()
    again = make_multilabel_folds(frame, num_folds=3, seed=7)
    assert folded["fold"].tolist() == again["fold"].tolist()


def test_tune_thresholds_returns_one_threshold_per_label():
    y_true = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
    y_prob = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.4, 0.6]])
    thresholds = tune_thresholds(y_true, y_prob, labels=["A", "B"])
    assert set(thresholds) == {"A", "B"}
    assert all(0.05 <= value <= 0.95 for value in thresholds.values())


def test_apply_no_finding_consistency_makes_normal_exclusive():
    predictions = pd.DataFrame(
        {
            "Atelectasis": [1, 0],
            "Cardiomegaly": [0, 0],
            "Consolidation": [0, 0],
            "Edema": [0, 0],
            "Enlarged Cardiomediastinum": [0, 0],
            "Fracture": [0, 0],
            "Lung Lesion": [0, 0],
            "Lung Opacity": [0, 0],
            "No Finding": [1, 0],
            "Pleural Effusion": [0, 0],
            "Pleural Other": [0, 0],
            "Pneumonia": [0, 0],
            "Pneumothorax": [0, 0],
        }
    )
    fixed = apply_no_finding_consistency(predictions)
    assert fixed.loc[0, "No Finding"] == 0
    assert fixed.loc[1, "No Finding"] == 1


def test_compute_metrics_reports_macro_f1_and_per_label_f1():
    y_true = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
    y_prob = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.4, 0.6]])
    report = compute_metrics(y_true, y_prob, labels=["A", "B"], thresholds={"A": 0.5, "B": 0.5})
    assert report["macro_f1"] == 1.0
    assert report["per_label"]["A"]["f1"] == 1.0
    assert report["per_label"]["B"]["f1"] == 1.0
