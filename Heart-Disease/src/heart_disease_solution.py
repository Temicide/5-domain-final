from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, MutableMapping, Sequence

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ID_COLUMN = "ID"
TARGET_COLUMN = "History of HeartDisease or Attack"
POSITIVE_LABEL = "Yes"
NEGATIVE_LABEL = "No"
VALID_LABELS = {POSITIVE_LABEL, NEGATIVE_LABEL}

EXPECTED_TRAIN_COLUMNS = [
    "ID",
    "History of HeartDisease or Attack",
    "High Blood Pressure",
    "Told High Cholesterol",
    "Cholesterol Checked",
    "Body Mass Index",
    "Smoked 100+ Cigarettes",
    "Diagnosed Stroke",
    "Diagnosed Diabetes",
    "Leisure Physical Activity",
    "Heavy Alcohol Consumption",
    "Health Care Coverage",
    "Doctor Visit Cost Barrier",
    "General Health",
    "Difficulty Walking",
    "Sex",
    "Education Level",
    "Income Level",
    "Age",
    "Vegetable or Fruit Intake (1+ per Day)",
]
EXPECTED_TEST_COLUMNS = [column for column in EXPECTED_TRAIN_COLUMNS if column != TARGET_COLUMN]
SUBMISSION_COLUMNS = ["ID", "History of HeartDisease or Attack"]

PROJECT_ROOT = "/Users/temicide/Documents/5_domain_final/Heart-Disease"
COMPETITION_SLUG = "super-ai-engineer-ss-6-individual-heart-disease-prediction"

COLAB_ROOT = "/content"
COLAB_INPUT_ROOT = "/content/input"
COLAB_WORKING_DIR = "/content/working"
COLAB_COMPETITION_DIR = f"{COLAB_INPUT_ROOT}/{COMPETITION_SLUG}"
COLAB_ARCHIVE_PATH = f"{COLAB_INPUT_ROOT}/{COMPETITION_SLUG}.zip"
COLAB_SUBMISSION_PATH = "/content/submission.csv"
COLAB_WORKING_SUBMISSION_PATH = "/content/working/submission.csv"

LOCAL_DATA_DIR = f"{PROJECT_ROOT}/data/{COMPETITION_SLUG}"
LOCAL_OUTPUT_DIR = f"{PROJECT_ROOT}/outputs"
LOCAL_SUBMISSION_PATH = f"{LOCAL_OUTPUT_DIR}/submissions/submission.csv"

NUMERIC_COLUMNS = ["Body Mass Index", "Age"]
CATEGORICAL_COLUMNS = [column for column in EXPECTED_TEST_COLUMNS if column not in {ID_COLUMN, *NUMERIC_COLUMNS}]


@dataclass(frozen=True)
class DataPaths:
    train_path: Path
    test_path: Path
    sample_submission_path: Path
    submission_path: Path
    working_submission_path: Path | None


@dataclass
class CompetitionData:
    train: pd.DataFrame
    test: pd.DataFrame
    sample_submission: pd.DataFrame


@dataclass
class ModelResult:
    model_name: str
    oof_probabilities: np.ndarray
    test_probabilities: np.ndarray
    metrics: dict[str, Any]
    threshold_metrics: dict[str, float]


def is_colab_runtime() -> bool:
    return "google.colab" in sys.modules or Path(COLAB_ROOT).exists()


def _required_csvs_exist(directory: Path) -> bool:
    return all((directory / filename).exists() for filename in ("train.csv", "test.csv", "sample_submission.csv"))


def configure_kaggle_credentials(
    kaggle_json_bytes: bytes | None = None,
    kaggle_dir: Path = Path("/root/.kaggle"),
    env: MutableMapping[str, str] | None = None,
) -> str:
    credential_env = os.environ if env is None else env
    if credential_env.get("KAGGLE_USERNAME") and credential_env.get("KAGGLE_KEY"):
        return "environment_variables"
    if kaggle_json_bytes:
        kaggle_dir.mkdir(parents=True, exist_ok=True)
        credential_path = kaggle_dir / "kaggle.json"
        credential_path.write_bytes(kaggle_json_bytes)
        credential_path.chmod(0o600)
        try:
            parsed = json.loads(kaggle_json_bytes.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise RuntimeError("Uploaded kaggle.json is not valid JSON.") from exc
        if not parsed.get("username") or not parsed.get("key"):
            raise RuntimeError("Uploaded kaggle.json must contain username and key.")
        return "uploaded_kaggle_json"
    raise RuntimeError(
        "Kaggle credentials were not found. Set KAGGLE_USERNAME and KAGGLE_KEY, "
        "or upload kaggle.json in Colab and pass its bytes to configure_kaggle_credentials."
    )


def download_competition_archive(input_root: Path = Path(COLAB_INPUT_ROOT)) -> Path:
    input_root.mkdir(parents=True, exist_ok=True)
    archive_path = input_root / f"{COMPETITION_SLUG}.zip"
    command = ["kaggle", "competitions", "download", "-c", COMPETITION_SLUG, "-p", str(input_root)]
    subprocess.run(command, check=True)
    if not archive_path.exists():
        raise RuntimeError(f"Kaggle download completed but archive was not found at {archive_path}.")
    return archive_path


def extract_competition_archive(archive_path: Path, competition_dir: Path) -> Path:
    competition_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as zip_ref:
        zip_ref.extractall(competition_dir)
    if not _required_csvs_exist(competition_dir):
        raise RuntimeError(f"Extracted archive is missing train.csv, test.csv, or sample_submission.csv in {competition_dir}.")
    return competition_dir


def ensure_competition_data_available(force_download: bool = False) -> Path:
    input_root = Path(COLAB_INPUT_ROOT)
    working_dir = Path(COLAB_WORKING_DIR)
    competition_dir = Path(COLAB_COMPETITION_DIR)
    archive_path = Path(COLAB_ARCHIVE_PATH)
    input_root.mkdir(parents=True, exist_ok=True)
    working_dir.mkdir(parents=True, exist_ok=True)

    if _required_csvs_exist(competition_dir) and not force_download:
        return competition_dir

    if force_download or not archive_path.exists():
        archive_path = download_competition_archive(input_root=input_root)
    return extract_competition_archive(archive_path, competition_dir)


def resolve_data_paths() -> DataPaths:
    colab_dir = Path(COLAB_COMPETITION_DIR)
    local_dir = Path(LOCAL_DATA_DIR)
    if _required_csvs_exist(colab_dir):
        return DataPaths(
            train_path=colab_dir / "train.csv",
            test_path=colab_dir / "test.csv",
            sample_submission_path=colab_dir / "sample_submission.csv",
            submission_path=Path(COLAB_SUBMISSION_PATH),
            working_submission_path=Path(COLAB_WORKING_SUBMISSION_PATH),
        )
    if _required_csvs_exist(local_dir):
        return DataPaths(
            train_path=local_dir / "train.csv",
            test_path=local_dir / "test.csv",
            sample_submission_path=local_dir / "sample_submission.csv",
            submission_path=Path(LOCAL_SUBMISSION_PATH),
            working_submission_path=None,
        )
    raise FileNotFoundError(
        "Competition CSVs were not found. In Colab, run ensure_competition_data_available() before loading data. "
        f"Local fallback expected files under {local_dir}."
    )


def _normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [str(column).lstrip("\ufeff") for column in frame.columns]
    return frame


def load_competition_data(paths: DataPaths | None = None) -> CompetitionData:
    resolved = resolve_data_paths() if paths is None else paths
    train = _normalize_columns(pd.read_csv(resolved.train_path, encoding="utf-8-sig"))
    test = _normalize_columns(pd.read_csv(resolved.test_path, encoding="utf-8-sig"))
    sample = _normalize_columns(pd.read_csv(resolved.sample_submission_path, encoding="utf-8-sig"))
    validate_input_frames(train, test, sample)
    return CompetitionData(train=train, test=test, sample_submission=sample)


def validate_input_frames(train: pd.DataFrame, test: pd.DataFrame, sample_submission: pd.DataFrame) -> None:
    if list(train.columns) != EXPECTED_TRAIN_COLUMNS:
        raise ValueError(f"Train columns do not match expected contract: {list(train.columns)}")
    if list(test.columns) != EXPECTED_TEST_COLUMNS:
        raise ValueError(f"Test columns do not match expected contract: {list(test.columns)}")
    if list(sample_submission.columns) != SUBMISSION_COLUMNS:
        raise ValueError(f"Sample submission columns do not match expected contract: {list(sample_submission.columns)}")
    labels = train[TARGET_COLUMN].dropna().astype(str).str.strip()
    invalid = set(labels[labels != ""].unique()) - VALID_LABELS
    if invalid:
        raise ValueError(f"Train target contains invalid labels: {sorted(invalid)}")
    if len(sample_submission) != len(test):
        raise ValueError("Sample submission row count must match test row count.")
    if sample_submission[ID_COLUMN].tolist() != test[ID_COLUMN].tolist():
        raise ValueError("Sample submission IDs must match test IDs exactly and in order.")


def prepare_supervised_training_frame(train: pd.DataFrame) -> pd.DataFrame:
    labels = train[TARGET_COLUMN]
    mask = labels.notna() & labels.astype(str).str.strip().ne("")
    supervised = train.loc[mask].copy()
    supervised[TARGET_COLUMN] = supervised[TARGET_COLUMN].astype(str).str.strip()
    return supervised


def _yes_no_indicator(series: pd.Series) -> pd.Series:
    normalized = series.fillna("Missing").astype(str).str.strip().str.lower()
    return normalized.isin({"yes", "1", "true"}).astype(int)


def _rank_text(series: pd.Series, mapping: dict[str, int]) -> pd.Series:
    return series.fillna("Missing").astype(str).str.strip().map(mapping).fillna(0).astype(int)


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    features = frame.copy()
    for column in CATEGORICAL_COLUMNS:
        if column in features:
            features[column] = features[column].fillna("Missing").astype(str).replace({"": "Missing"})
    features["Body Mass Index"] = pd.to_numeric(features["Body Mass Index"], errors="coerce")
    features["Age"] = pd.to_numeric(features["Age"], errors="coerce")

    features["age_bin"] = pd.cut(
        features["Age"],
        bins=[-np.inf, 34, 44, 54, 64, np.inf],
        labels=["under_35", "35_44", "45_54", "55_64", "65_plus"],
    ).astype("object").fillna("Missing")
    features["bmi_class"] = pd.cut(
        features["Body Mass Index"],
        bins=[-np.inf, 18.5, 25, 30, np.inf],
        labels=["underweight", "normal", "overweight", "obese"],
    ).astype("object").fillna("Missing")

    clinical_cols = [
        "High Blood Pressure",
        "Told High Cholesterol",
        "Diagnosed Stroke",
        "Diagnosed Diabetes",
        "Difficulty Walking",
    ]
    lifestyle_cols = [
        "Leisure Physical Activity",
        "Heavy Alcohol Consumption",
        "Vegetable or Fruit Intake (1+ per Day)",
    ]
    features["clinical_risk_count"] = sum(_yes_no_indicator(features[column]) for column in clinical_cols)
    features["cardiometabolic_cluster"] = (
        _yes_no_indicator(features["High Blood Pressure"])
        + _yes_no_indicator(features["Told High Cholesterol"])
        + _yes_no_indicator(features["Diagnosed Diabetes"])
    )
    features["lifestyle_protective_count"] = (
        _yes_no_indicator(features["Leisure Physical Activity"])
        + (1 - _yes_no_indicator(features["Heavy Alcohol Consumption"]))
        + _yes_no_indicator(features["Vegetable or Fruit Intake (1+ per Day)"])
    )
    features["health_access_friction"] = (
        (1 - _yes_no_indicator(features["Health Care Coverage"]))
        + _yes_no_indicator(features["Doctor Visit Cost Barrier"])
    )
    education_map = {"never attended": 1, "elementary": 2, "some high school": 3, "high school": 4, "college": 5, "graduate": 6}
    income_map = {"less than $10,000": 1, "$10,000 to $15,000": 2, "$15,000 to $20,000": 3, "$20,000 to $25,000": 4, "$25,000 to $35,000": 5, "$35,000 to $50,000": 6, "$50,000 to $75,000": 7, "$75,000 or more": 8}
    features["education_rank"] = _rank_text(features["Education Level"], education_map)
    features["income_rank"] = _rank_text(features["Income Level"], income_map)
    features["age_x_high_bp"] = features["Age"].fillna(features["Age"].median()) * _yes_no_indicator(features["High Blood Pressure"])
    general_health_rank = _rank_text(
        features["General Health"],
        {"excellent": 1, "very good": 2, "good": 3, "fair": 4, "poor": 5},
    )
    features["general_health_x_walking"] = general_health_rank * _yes_no_indicator(features["Difficulty Walking"])
    return features


def calculate_threshold_metrics(y_true: Sequence[int], probabilities: Sequence[float], threshold: float) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    labels = (p >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "f1": float(f1_score(y, labels, zero_division=0)),
        "accuracy": float(accuracy_score(y, labels)),
        "precision": float(precision_score(y, labels, zero_division=0)),
        "recall": float(recall_score(y, labels, zero_division=0)),
        "positive_rate": float(labels.mean()),
    }


def tune_threshold(
    y_true: Sequence[int],
    probabilities: Sequence[float],
    metric: str = "f1",
    thresholds: Sequence[float] | None = None,
) -> dict[str, float]:
    if metric not in {"f1", "accuracy"}:
        raise ValueError("threshold metric must be 'f1' or 'accuracy'.")
    grid = np.asarray(thresholds if thresholds is not None else np.linspace(0.01, 0.99, 99), dtype=float)
    scored = [calculate_threshold_metrics(y_true, probabilities, threshold) for threshold in grid]
    return max(scored, key=lambda item: (item[metric], item["threshold"]))


def validate_submission(submission: pd.DataFrame, test: pd.DataFrame) -> None:
    if list(submission.columns) != SUBMISSION_COLUMNS:
        raise ValueError(f"Submission columns must be {SUBMISSION_COLUMNS}.")
    if len(submission) != len(test):
        raise ValueError("Submission row count must match test row count.")
    if submission[ID_COLUMN].tolist() != test[ID_COLUMN].tolist():
        raise ValueError("Submission IDs must match test IDs exactly and in order.")
    labels = submission[TARGET_COLUMN]
    if labels.isna().any() or labels.astype(str).str.strip().eq("").any():
        raise ValueError("Submission labels must not be blank.")
    invalid = set(labels.astype(str).str.strip().unique()) - VALID_LABELS
    if invalid:
        raise ValueError(f"Submission labels must be Yes or No, found {sorted(invalid)}.")


def make_submission(test: pd.DataFrame, probabilities: Sequence[float], threshold: float) -> pd.DataFrame:
    p = np.asarray(probabilities, dtype=float)
    if len(p) != len(test):
        raise ValueError("Probability row count must match test row count.")
    labels = np.where(p >= threshold, POSITIVE_LABEL, NEGATIVE_LABEL)
    submission = pd.DataFrame({ID_COLUMN: test[ID_COLUMN].tolist(), TARGET_COLUMN: labels})
    validate_submission(submission, test)
    return submission


def write_submission(submission: pd.DataFrame, test: pd.DataFrame, path: Path, mirror_path: Path | None = None) -> Path:
    validate_submission(submission, test)
    path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(path, index=False, encoding="utf-8-sig")
    if mirror_path is not None:
        mirror_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, mirror_path)
    return path


def _make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    numeric = features.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [column for column in features.columns if column not in numeric and column != ID_COLUMN]
    numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical_pipeline = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", _make_one_hot_encoder())])
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric),
            ("categorical", categorical_pipeline, categorical),
        ],
        remainder="drop",
    )


def create_model(model_name: str, random_state: int = 42) -> Any:
    if model_name == "logistic":
        return LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear", random_state=random_state)
    if model_name == "extratrees":
        return ExtraTreesClassifier(n_estimators=300, random_state=random_state, class_weight="balanced", n_jobs=-1)
    if model_name == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise RuntimeError("lightgbm is not installed. Install it with `pip install lightgbm`.") from exc
        return LGBMClassifier(n_estimators=600, learning_rate=0.03, num_leaves=31, subsample=0.9, colsample_bytree=0.9, random_state=random_state)
    if model_name == "catboost":
        try:
            from catboost import CatBoostClassifier
        except ImportError as exc:
            raise RuntimeError("catboost is not installed. Install it with `pip install catboost`.") from exc
        return CatBoostClassifier(iterations=600, learning_rate=0.03, depth=6, loss_function="Logloss", verbose=False, random_seed=random_state)
    if model_name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise RuntimeError("xgboost is not installed. Install it with `pip install xgboost`.") from exc
        return XGBClassifier(n_estimators=600, learning_rate=0.03, max_depth=4, subsample=0.9, colsample_bytree=0.9, eval_metric="logloss", random_state=random_state)
    raise ValueError(f"Unsupported model_name: {model_name}")


def _classification_metrics(y_true: Sequence[int], probabilities: Sequence[float]) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    metrics: dict[str, float] = {}
    metrics["roc_auc"] = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
    metrics["pr_auc"] = float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
    return metrics


def train_single_model_cv(
    train: pd.DataFrame,
    test: pd.DataFrame,
    model_name: str = "logistic",
    n_splits: int = 5,
    random_state: int = 42,
    threshold_metric: str = "f1",
) -> ModelResult:
    supervised = prepare_supervised_training_frame(train)
    y = supervised[TARGET_COLUMN].map({NEGATIVE_LABEL: 0, POSITIVE_LABEL: 1}).to_numpy(dtype=int)
    x = build_features(supervised.drop(columns=[TARGET_COLUMN]))
    x_test = build_features(test)
    drop_cols = [ID_COLUMN]
    x = x.drop(columns=[column for column in drop_cols if column in x], errors="ignore")
    x_test = x_test.drop(columns=[column for column in drop_cols if column in x_test], errors="ignore")

    splits = min(n_splits, int(np.bincount(y).min())) if len(np.unique(y)) > 1 else 2
    if splits < 2:
        raise ValueError("At least two examples per class are required for cross-validation.")
    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=random_state)
    oof = np.zeros(len(x), dtype=float)
    test_fold_predictions = []

    for fold, (train_idx, valid_idx) in enumerate(cv.split(x, y)):
        preprocessor = _build_preprocessor(x)
        model = clone(create_model(model_name, random_state=random_state + fold))
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
        pipeline.fit(x.iloc[train_idx], y[train_idx])
        oof[valid_idx] = pipeline.predict_proba(x.iloc[valid_idx])[:, 1]
        test_fold_predictions.append(pipeline.predict_proba(x_test)[:, 1])

    test_probabilities = np.mean(test_fold_predictions, axis=0)
    metrics = _classification_metrics(y, oof)
    threshold_metrics = tune_threshold(y, oof, metric=threshold_metric)
    return ModelResult(
        model_name=model_name,
        oof_probabilities=oof,
        test_probabilities=np.asarray(test_probabilities, dtype=float),
        metrics=metrics,
        threshold_metrics=threshold_metrics,
    )


def rank_average_predictions(predictions: Sequence[Sequence[float]], weights: Sequence[float] | None = None) -> np.ndarray:
    arrays = [np.asarray(prediction, dtype=float) for prediction in predictions]
    if not arrays:
        raise ValueError("At least one prediction array is required.")
    lengths = {len(array) for array in arrays}
    if len(lengths) != 1:
        raise ValueError("All prediction arrays must have the same length.")
    if weights is None:
        weight_array = np.ones(len(arrays), dtype=float)
    else:
        weight_array = np.asarray(weights, dtype=float)
        if len(weight_array) != len(arrays):
            raise ValueError("weights length must match predictions length.")
    weight_array = weight_array / weight_array.sum()
    ranks = []
    for array in arrays:
        order = pd.Series(array).rank(method="average").to_numpy(dtype=float)
        denom = max(len(array) - 1, 1)
        ranks.append((order - 1) / denom)
    return np.average(np.vstack(ranks), axis=0, weights=weight_array)


def find_best_ensemble_weights(
    y_true: Sequence[int],
    oof_predictions: Sequence[Sequence[float]],
    metric: str = "f1",
) -> tuple[list[float], dict[str, float]]:
    n = len(oof_predictions)
    if n == 1:
        probabilities = rank_average_predictions(oof_predictions)
        return [1.0], tune_threshold(y_true, probabilities, metric=metric)
    candidates: list[list[float]] = []
    candidates.append([1 / n] * n)
    for idx in range(n):
        weights = [0.0] * n
        weights[idx] = 1.0
        candidates.append(weights)
    best_weights = candidates[0]
    best_metrics = tune_threshold(y_true, rank_average_predictions(oof_predictions, best_weights), metric=metric)
    for weights in candidates[1:]:
        metrics = tune_threshold(y_true, rank_average_predictions(oof_predictions, weights), metric=metric)
        if (metrics[metric], metrics["threshold"]) > (best_metrics[metric], best_metrics["threshold"]):
            best_weights, best_metrics = weights, metrics
    return best_weights, best_metrics


def save_oof_predictions(ids: Sequence[Any], y_true: Sequence[int], probabilities: Sequence[float], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({ID_COLUMN: list(ids), "target": list(y_true), "oof_probability": list(probabilities)})
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def run_experiment(
    model_names: list[str] | None = None,
    threshold_metric: str = "f1",
    force_download: bool = False,
) -> dict[str, Any]:
    if is_colab_runtime():
        ensure_competition_data_available(force_download=force_download)
    paths = resolve_data_paths()
    data = load_competition_data(paths)
    supervised = prepare_supervised_training_frame(data.train)
    y = supervised[TARGET_COLUMN].map({NEGATIVE_LABEL: 0, POSITIVE_LABEL: 1}).to_numpy(dtype=int)
    selected_models = model_names or ["lightgbm", "catboost", "xgboost", "extratrees", "logistic"]

    results: list[ModelResult] = []
    skipped: dict[str, str] = {}
    for model_name in selected_models:
        try:
            results.append(train_single_model_cv(data.train, data.test, model_name=model_name, threshold_metric=threshold_metric))
        except RuntimeError as exc:
            skipped[model_name] = str(exc)
    if not results:
        raise RuntimeError(f"No models completed successfully. Skipped models: {skipped}")

    weights, threshold_metrics = find_best_ensemble_weights(
        y,
        [result.oof_probabilities for result in results],
        metric=threshold_metric,
    )
    ensemble_oof = rank_average_predictions([result.oof_probabilities for result in results], weights)
    ensemble_test = rank_average_predictions([result.test_probabilities for result in results], weights)
    ensemble_metrics = _classification_metrics(y, ensemble_oof)
    submission = make_submission(data.test, ensemble_test, threshold_metrics["threshold"])
    submission_path = write_submission(submission, data.test, paths.submission_path, paths.working_submission_path)
    oof_path = save_oof_predictions(supervised[ID_COLUMN].tolist(), y, ensemble_oof, Path(LOCAL_OUTPUT_DIR) / "oof" / "ensemble_oof.csv")
    return {
        "submission_rows": int(len(submission)),
        "submission_path": str(submission_path),
        "working_submission_path": str(paths.working_submission_path) if paths.working_submission_path else None,
        "positive_predictions": int((submission[TARGET_COLUMN] == POSITIVE_LABEL).sum()),
        "threshold_metrics": threshold_metrics,
        "ensemble_metrics": ensemble_metrics,
        "model_metrics": {result.model_name: result.metrics for result in results},
        "model_weights": dict(zip([result.model_name for result in results], weights)),
        "skipped_models": skipped,
        "oof_path": str(oof_path),
    }
