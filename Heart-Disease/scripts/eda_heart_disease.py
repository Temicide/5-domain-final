from __future__ import annotations

import math
import sqlite3
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "super-ai-engineer-ss-6-individual-heart-disease-prediction"
ANALYSIS_DIR = PROJECT_ROOT / "analysis" / "exploratory-heart-disease"
DB_PATH = ANALYSIS_DIR / "heart_disease.sqlite"

ID_COLUMN = "ID"
TARGET_COLUMN = "History of HeartDisease or Attack"
POSITIVE_LABEL = "Yes"


def read_csv(name: str) -> pd.DataFrame:
    frame = pd.read_csv(DATA_DIR / f"{name}.csv", encoding="utf-8-sig")
    frame.columns = [str(column).lstrip("\ufeff") for column in frame.columns]
    return frame


def md_table(frame: pd.DataFrame, max_rows: int | None = None, float_digits: int = 3) -> str:
    if max_rows is not None:
        frame = frame.head(max_rows)
    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: "" if pd.isna(value) else f"{value:.{float_digits}f}")
    display = display.fillna("").astype(str)
    headers = [str(column) for column in display.columns]
    rows = display.values.tolist()
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)
    return "\n".join(lines)


def pct(value: float) -> str:
    if pd.isna(value):
        return ""
    return f"{100 * value:.2f}%"


def bar(value: float, max_value: float, width: int = 28) -> str:
    if max_value <= 0 or pd.isna(value):
        return ""
    filled = max(0, int(round(width * value / max_value)))
    return "#" * filled


def target_mask(train: pd.DataFrame) -> pd.Series:
    target = train[TARGET_COLUMN]
    return target.notna() & target.astype(str).str.strip().ne("")


def positive_rate(frame: pd.DataFrame) -> float:
    if len(frame) == 0:
        return float("nan")
    return (frame[TARGET_COLUMN].astype(str).str.strip() == POSITIVE_LABEL).mean()


def segment_rates(frame: pd.DataFrame, column: str, min_count: int = 50) -> pd.DataFrame:
    rows = []
    values = frame[column].fillna("Missing").astype(str).replace({"": "Missing"})
    grouped = frame.assign(_segment=values).groupby("_segment", dropna=False)
    for segment, group in grouped:
        count = len(group)
        if count < min_count:
            continue
        rate = positive_rate(group)
        rows.append(
            {
                "feature": column,
                "segment": segment,
                "n": count,
                "positive_rate": rate,
                "lift_vs_base": rate / BASE_RATE if BASE_RATE else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["positive_rate", "n"], ascending=[False, False])


def train_test_distribution(train: pd.DataFrame, test: pd.DataFrame, column: str) -> pd.DataFrame:
    train_values = train[column].fillna("Missing").astype(str).replace({"": "Missing"})
    test_values = test[column].fillna("Missing").astype(str).replace({"": "Missing"})
    train_dist = train_values.value_counts(normalize=True)
    test_dist = test_values.value_counts(normalize=True)
    levels = sorted(set(train_dist.index) | set(test_dist.index))
    rows = []
    for level in levels:
        rows.append(
            {
                "feature": column,
                "level": level,
                "train_pct": train_dist.get(level, 0.0),
                "test_pct": test_dist.get(level, 0.0),
                "abs_diff_pct": abs(train_dist.get(level, 0.0) - test_dist.get(level, 0.0)),
            }
        )
    return pd.DataFrame(rows).sort_values("abs_diff_pct", ascending=False)


def binary_indicator(frame: pd.DataFrame, column: str) -> pd.Series:
    values = frame[column].fillna("Missing").astype(str).str.strip().str.lower()
    return values.isin({"yes", "1", "true"}).astype(int)


def rank_map(frame: pd.DataFrame, column: str, mapping: dict[str, int]) -> pd.Series:
    values = frame[column].fillna("Missing").astype(str).str.strip().str.lower()
    return values.map(mapping).fillna(0).astype(int)


def write(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def rebuild_sqlite(train: pd.DataFrame, test: pd.DataFrame, sample: pd.DataFrame) -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        train.to_sql("train_raw", conn, if_exists="replace", index=False)
        test.to_sql("test_raw", conn, if_exists="replace", index=False)
        sample.to_sql("sample_submission_raw", conn, if_exists="replace", index=False)
        conn.execute("DROP TABLE IF EXISTS train_labeled")
        conn.execute(
            f"""
            CREATE TABLE train_labeled AS
            SELECT *
            FROM train_raw
            WHERE "{TARGET_COLUMN}" IS NOT NULL
              AND trim("{TARGET_COLUMN}") <> ''
            """
        )


def schema_frame(frame: pd.DataFrame, table_name: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "table": table_name,
            "column": frame.columns,
            "dtype": [str(dtype) for dtype in frame.dtypes],
            "non_null": [int(frame[column].notna().sum()) for column in frame.columns],
            "missing": [int(frame[column].isna().sum()) for column in frame.columns],
            "unique_values": [int(frame[column].nunique(dropna=True)) for column in frame.columns],
        }
    )


def numeric_summary(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    rows = []
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        rows.append(
            {
                "column": column,
                "count": int(values.notna().sum()),
                "missing": int(values.isna().sum()),
                "min": values.min(),
                "p01": values.quantile(0.01),
                "p25": values.quantile(0.25),
                "median": values.median(),
                "p75": values.quantile(0.75),
                "p99": values.quantile(0.99),
                "max": values.max(),
                "mean": values.mean(),
            }
        )
    return pd.DataFrame(rows)


def id_decile_rates(frame: pd.DataFrame) -> pd.DataFrame:
    values = frame[[ID_COLUMN, TARGET_COLUMN]].copy()
    values["_id_number"] = pd.to_numeric(values[ID_COLUMN].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    values = values.dropna(subset=["_id_number"])
    values["_id_decile"] = pd.qcut(values["_id_number"], q=10, labels=False, duplicates="drop") + 1
    grouped = values.groupby("_id_decile")
    rows = []
    for decile, group in grouped:
        rate = positive_rate(group)
        rows.append(
            {
                "id_decile": int(decile),
                "id_min": int(group["_id_number"].min()),
                "id_max": int(group["_id_number"].max()),
                "n": len(group),
                "positive_rate": rate,
                "bar": bar(rate, 0.13),
            }
        )
    return pd.DataFrame(rows)


def age_patterns(frame: pd.DataFrame) -> pd.DataFrame:
    values = frame.copy()
    values["Age"] = pd.to_numeric(values["Age"], errors="coerce")
    values["_age_bin"] = pd.cut(
        values["Age"],
        bins=[-np.inf, 34, 44, 54, 64, np.inf],
        labels=["<=34", "35-44", "45-54", "55-64", "65+"],
    )
    rows = []
    for segment, group in values.groupby("_age_bin", observed=False):
        rate = positive_rate(group)
        rows.append(
            {
                "age_bin": str(segment),
                "n": len(group),
                "positive_rate": rate,
                "lift_vs_base": rate / BASE_RATE,
                "bar": bar(rate, 0.16),
            }
        )
    return pd.DataFrame(rows)


def bmi_patterns(frame: pd.DataFrame) -> pd.DataFrame:
    values = frame.copy()
    values["Body Mass Index"] = pd.to_numeric(values["Body Mass Index"], errors="coerce")
    values["_bmi_class"] = pd.cut(
        values["Body Mass Index"],
        bins=[-np.inf, 18.5, 25, 30, 35, 40, np.inf],
        labels=["underweight", "normal", "overweight", "obese_i", "obese_ii", "obese_iii"],
    ).astype("object").fillna("Missing")
    rows = []
    for segment, group in values.groupby("_bmi_class"):
        rate = positive_rate(group)
        rows.append(
            {
                "bmi_class": str(segment),
                "n": len(group),
                "positive_rate": rate,
                "lift_vs_base": rate / BASE_RATE,
                "bar": bar(rate, 0.12),
            }
        )
    order = ["underweight", "normal", "overweight", "obese_i", "obese_ii", "obese_iii", "Missing"]
    result = pd.DataFrame(rows)
    result["_order"] = result["bmi_class"].map({name: i for i, name in enumerate(order)})
    return result.sort_values("_order").drop(columns="_order")


def risk_counts(frame: pd.DataFrame) -> pd.DataFrame:
    values = frame.copy()
    clinical_cols = [
        "High Blood Pressure",
        "Told High Cholesterol",
        "Diagnosed Stroke",
        "Diagnosed Diabetes",
        "Difficulty Walking",
    ]
    lifestyle_protective = (
        binary_indicator(values, "Leisure Physical Activity")
        + (1 - binary_indicator(values, "Heavy Alcohol Consumption"))
        + binary_indicator(values, "Vegetable or Fruit Intake (1+ per Day)")
    )
    values["_clinical_risk_count"] = sum(binary_indicator(values, column) for column in clinical_cols)
    values["_lifestyle_protective_count"] = lifestyle_protective
    rows = []
    for count, group in values.groupby("_clinical_risk_count"):
        rate = positive_rate(group)
        rows.append(
            {
                "clinical_risk_count": int(count),
                "n": len(group),
                "positive_rate": rate,
                "lift_vs_base": rate / BASE_RATE,
                "bar": bar(rate, 0.35),
            }
        )
    clinical = pd.DataFrame(rows).sort_values("clinical_risk_count")
    rows = []
    for count, group in values.groupby("_lifestyle_protective_count"):
        rate = positive_rate(group)
        rows.append(
            {
                "lifestyle_protective_count": int(count),
                "n": len(group),
                "positive_rate": rate,
                "lift_vs_base": rate / BASE_RATE,
                "bar": bar(rate, 0.14),
            }
        )
    lifestyle = pd.DataFrame(rows).sort_values("lifestyle_protective_count")
    return clinical, lifestyle


def top_interactions(frame: pd.DataFrame) -> pd.DataFrame:
    candidates = [
        ("Diagnosed Stroke", "General Health"),
        ("Diagnosed Stroke", "Difficulty Walking"),
        ("Diagnosed Stroke", "Diagnosed Diabetes"),
        ("High Blood Pressure", "Told High Cholesterol"),
        ("High Blood Pressure", "Diagnosed Diabetes"),
        ("Difficulty Walking", "General Health"),
        ("Sex", "Age"),
        ("Income Level", "General Health"),
        ("Education Level", "Income Level"),
    ]
    rows = []
    for left, right in candidates:
        left_values = frame[left].fillna("Missing").astype(str).replace({"": "Missing"})
        right_values = frame[right].fillna("Missing").astype(str).replace({"": "Missing"})
        grouped = frame.assign(_left=left_values, _right=right_values).groupby(["_left", "_right"])
        for (left_value, right_value), group in grouped:
            if len(group) < 300:
                continue
            rate = positive_rate(group)
            rows.append(
                {
                    "interaction": f"{left}={left_value} | {right}={right_value}",
                    "n": len(group),
                    "positive_rate": rate,
                    "lift_vs_base": rate / BASE_RATE,
                }
            )
    return pd.DataFrame(rows).sort_values(["positive_rate", "n"], ascending=[False, False]).head(20)


def correlation_table(frame: pd.DataFrame) -> pd.DataFrame:
    encoded = pd.DataFrame(index=frame.index)
    encoded["target_yes"] = (frame[TARGET_COLUMN].astype(str).str.strip() == POSITIVE_LABEL).astype(int)
    encoded["age"] = pd.to_numeric(frame["Age"], errors="coerce")
    encoded["bmi"] = pd.to_numeric(frame["Body Mass Index"], errors="coerce")
    for column in [
        "High Blood Pressure",
        "Told High Cholesterol",
        "Cholesterol Checked",
        "Smoked 100+ Cigarettes",
        "Diagnosed Stroke",
        "Diagnosed Diabetes",
        "Leisure Physical Activity",
        "Heavy Alcohol Consumption",
        "Health Care Coverage",
        "Doctor Visit Cost Barrier",
        "Difficulty Walking",
        "Vegetable or Fruit Intake (1+ per Day)",
    ]:
        encoded[f"{column}=Yes"] = binary_indicator(frame, column)
    encoded["male"] = (frame["Sex"].fillna("").astype(str).str.lower() == "male").astype(int)
    encoded["general_health_rank"] = rank_map(
        frame,
        "General Health",
        {"excellent": 1, "very good": 2, "good": 3, "fair": 4, "poor": 5, "very poor": 6},
    )
    encoded["income_rank"] = rank_map(
        frame,
        "Income Level",
        {
            "less than $10,000": 1,
            "($10,000 to less than $15,000": 2,
            "$10,000 to $15,000": 2,
            "$15,000 to less than $20,000": 3,
            "$15,000 to $20,000": 3,
            "$20,000 to less than $25,000": 4,
            "$20,000 to $25,000": 4,
            "$25,000 to less than $35,000": 5,
            "$25,000 to $35,000": 5,
            "$35,000 to less than $50,000": 6,
            "$35,000 to $50,000": 6,
            "$50,000 to less than $75,000": 7,
            "$50,000 to $75,000": 7,
            "$75,000 or more": 8,
        },
    )
    encoded["education_rank"] = rank_map(
        frame,
        "Education Level",
        {
            "never attended": 1,
            "never attended school": 1,
            "elementary": 2,
            "some high school": 3,
            "high school": 4,
            "high school graduate": 4,
            "some college or technical school": 5,
            "college": 6,
            "college graduate": 6,
            "graduate": 6,
        },
    )
    encoded = encoded.drop(columns=["target_yes"]).corrwith(encoded["target_yes"]).rename("corr_with_target").reset_index()
    encoded = encoded.rename(columns={"index": "feature"})
    encoded["abs_corr"] = encoded["corr_with_target"].abs()
    return encoded.sort_values("abs_corr", ascending=False).drop(columns="abs_corr")


def categorical_summary(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [column for column in frame.columns if column not in {ID_COLUMN, TARGET_COLUMN, "Age", "Body Mass Index"}]
    pieces = [segment_rates(frame, column, min_count=50) for column in columns]
    return pd.concat(pieces, ignore_index=True).sort_values(["positive_rate", "n"], ascending=[False, False])


def distribution_drift(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in test.columns:
        if column == ID_COLUMN:
            continue
        if column in {"Age", "Body Mass Index"}:
            train_values = pd.to_numeric(train[column], errors="coerce")
            test_values = pd.to_numeric(test[column], errors="coerce")
            rows.append(
                {
                    "feature": column,
                    "largest_level_or_metric": "missing_rate",
                    "train_pct": train_values.isna().mean(),
                    "test_pct": test_values.isna().mean(),
                    "abs_diff_pct": abs(train_values.isna().mean() - test_values.isna().mean()),
                }
            )
        else:
            top = train_test_distribution(train, test, column).head(1).iloc[0]
            rows.append(
                {
                    "feature": column,
                    "largest_level_or_metric": top["level"],
                    "train_pct": top["train_pct"],
                    "test_pct": top["test_pct"],
                    "abs_diff_pct": top["abs_diff_pct"],
                }
            )
    return pd.DataFrame(rows).sort_values("abs_diff_pct", ascending=False)


def numeric_drift(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in ["Age", "Body Mass Index"]:
        train_values = pd.to_numeric(train[column], errors="coerce")
        test_values = pd.to_numeric(test[column], errors="coerce")
        rows.append(
            {
                "feature": column,
                "train_mean": train_values.mean(),
                "test_mean": test_values.mean(),
                "mean_diff": test_values.mean() - train_values.mean(),
                "standardized_diff": (test_values.mean() - train_values.mean()) / train_values.std(),
                "train_median": train_values.median(),
                "test_median": test_values.median(),
            }
        )
    return pd.DataFrame(rows)


def sample_submission_blanks(sample: pd.DataFrame) -> int:
    values = sample[TARGET_COLUMN]
    return int(values.fillna("").astype(str).str.strip().eq("").sum())


def format_rate_columns(frame: pd.DataFrame) -> pd.DataFrame:
    display = frame.copy()
    for column in [
        "positive_rate",
        "lift_vs_base",
        "train_pct",
        "test_pct",
        "abs_diff_pct",
        "train_missing_pct",
        "test_missing_pct",
        "pct_train",
        "pct_labeled",
        "corr_with_target",
    ]:
        if column in display:
            if column == "lift_vs_base":
                display[column] = display[column].map(lambda value: "" if pd.isna(value) else f"{value:.2f}x")
            elif column == "corr_with_target":
                display[column] = display[column].map(lambda value: "" if pd.isna(value) else f"{value:.3f}")
            else:
                display[column] = display[column].map(lambda value: pct(value))
    return display


def write_reports(train: pd.DataFrame, test: pd.DataFrame, sample: pd.DataFrame) -> None:
    labeled = train.loc[target_mask(train)].copy()
    positive_count = int((labeled[TARGET_COLUMN].astype(str).str.strip() == POSITIVE_LABEL).sum())
    negative_count = len(labeled) - positive_count
    missing_summary = (
        pd.DataFrame(
            {
                "column": train.columns,
                "train_missing": [int(train[column].isna().sum()) for column in train.columns],
                "train_missing_pct": [train[column].isna().mean() for column in train.columns],
                "test_missing": [int(test[column].isna().sum()) if column in test.columns else math.nan for column in train.columns],
                "test_missing_pct": [test[column].isna().mean() if column in test.columns else math.nan for column in train.columns],
            }
        )
        .sort_values("train_missing", ascending=False)
        .query("train_missing > 0 or test_missing > 0")
    )
    numeric_train = numeric_summary(train, ["Age", "Body Mass Index"])
    schema = pd.concat(
        [
            schema_frame(train, "train_raw"),
            schema_frame(test, "test_raw"),
            schema_frame(sample, "sample_submission_raw"),
        ],
        ignore_index=True,
    )
    target_counts = pd.DataFrame(
        {
            "target_value": ["No", "Yes", "Missing"],
            "rows": [negative_count, positive_count, int((~target_mask(train)).sum())],
            "pct_train": [negative_count / len(train), positive_count / len(train), (~target_mask(train)).mean()],
            "pct_labeled": [negative_count / len(labeled), positive_count / len(labeled), np.nan],
        }
    )
    age_table = age_patterns(labeled)
    id_table = id_decile_rates(labeled)
    bmi_table = bmi_patterns(labeled)
    categorical = categorical_summary(labeled)
    drift = distribution_drift(train, test)
    numeric_shift = numeric_drift(train, test)
    clinical, lifestyle = risk_counts(labeled)
    interactions = top_interactions(labeled)
    correlations = correlation_table(labeled)

    write(
        ANALYSIS_DIR / "00 - overview.md",
        f"""
# Heart Disease Kaggle EDA Overview

## Context

Dataset: `super-ai-engineer-ss-6-individual-heart-disease-prediction`

Source: local files in `{DATA_DIR}`.

Goal: find target patterns and data quirks that can improve a Kaggle submission for `{TARGET_COLUMN}`.

SQLite workspace: `{DB_PATH}` with `train_raw`, `test_raw`, `sample_submission_raw`, and `train_labeled`.

## Exploration Summary

- Train has {len(train):,} rows, test has {len(test):,} rows, and sample submission has {len(sample):,} rows.
- Usable supervised train rows: {len(labeled):,}. Missing/blank target rows: {(~target_mask(train)).sum():,}.
- Labeled positive rate: {pct(BASE_RATE)} ({positive_count:,} `Yes`, {negative_count:,} `No`).
- No true date/time column exists. The temporal phase therefore checks the absence of temporal coverage, age-ordered risk, and ID-order stability.
- Strongest univariate signals are stroke history, poor general health, difficulty walking, diabetes, blood pressure, cholesterol, age, sex, income, and education.
- Most important data quality issue for scoring: train-only missingness in `Told High Cholesterol`, `Body Mass Index`, and target; test has no missing feature values.

## Highest-Value Submission Ideas

1. Use GBDT models with native/robust categorical handling; the data is mixed tabular, imbalanced, and nonlinear.
2. Add interaction/risk-count features: clinical risk count, cardiometabolic cluster, general-health x walking difficulty, age x high blood pressure, BMI class, ordered income/education.
3. Preserve missingness in training but avoid over-trusting train-only missing flags because test has no missing feature values.
4. Tune the final label threshold from out-of-fold predictions. The base positive rate is only {pct(BASE_RATE)}, so default `0.5` probability threshold is unlikely to be optimal for F1-like metrics.
5. Validate improvements with stratified OOF metrics before using public leaderboard feedback.
""",
    )

    write(
        ANALYSIS_DIR / "01 - data-familiarization.md",
        f"""
# Data Familiarization

## Exploration Context

**Dataset:** Heart Disease Prediction Kaggle competition data
**Source:** `{DATA_DIR}`
**Exploration Goal:** discover patterns and quirks that can help score the Kaggle submission.

## Tables Overview

| table | rows | columns | apparent purpose | grain |
|---|---:|---:|---|---|
| `train_raw` | {len(train):,} | {train.shape[1]} | training survey records with target | one respondent |
| `train_labeled` | {len(labeled):,} | {train.shape[1]} | supervised subset after removing missing/blank target | one labeled respondent |
| `test_raw` | {len(test):,} | {test.shape[1]} | competition test records without target | one respondent |
| `sample_submission_raw` | {len(sample):,} | {sample.shape[1]} | required submission shape | one test ID |

## Schema Details

```sql
PRAGMA table_info(train_raw);
PRAGMA table_info(test_raw);
PRAGMA table_info(sample_submission_raw);
```

{md_table(schema)}

**Potential relationships:** `sample_submission_raw.ID` should align one-to-one and in order with `test_raw.ID`; no other join tables are present.

**Temporal coverage:** no date/time column exists. `Age` is an ordered respondent attribute, not a collection timestamp.

## Data Quality Assessment

### Target Balance

```sql
SELECT
  "History of HeartDisease or Attack" AS target_value,
  COUNT(*) AS rows
FROM train_raw
GROUP BY target_value;
```

{md_table(format_rate_columns(target_counts))}

### Completeness

```sql
SELECT COUNT(*) AS rows FROM train_raw;
-- Per-column missingness computed across train_raw and test_raw.
```

{md_table(format_rate_columns(missing_summary))}

### Numeric Ranges

```sql
SELECT
  MIN(Age) AS min_age,
  MAX(Age) AS max_age,
  MIN("Body Mass Index") AS min_bmi,
  MAX("Body Mass Index") AS max_bmi
FROM train_raw;
```

{md_table(numeric_train, float_digits=2)}

### ID and Submission Integrity

| check | result |
|---|---:|
| duplicate train IDs | {int(train[ID_COLUMN].duplicated().sum()):,} |
| duplicate test IDs | {int(test[ID_COLUMN].duplicated().sum()):,} |
| train/test ID overlap | {len(set(train[ID_COLUMN]) & set(test[ID_COLUMN])):,} |
| sample rows matching test rows | {sample[ID_COLUMN].tolist() == test[ID_COLUMN].tolist()} |
| blank sample target values | {sample_submission_blanks(sample):,} |

## Initial Impressions

- Data quality is good enough for modeling after dropping only missing target rows for supervised training.
- Predictor missingness is asymmetric: train has missing features, while test has no missing features.
- The base positive class is rare ({pct(BASE_RATE)}), so accuracy alone is misleading.
- Many features are categorical strings; treating ordinal categorical values as unordered may leave signal on the table.

## Exploration Strategy

1. **Time-based patterns:** no true temporal features; inspect age-ordered risk and ID-order stability as the closest available ordered dimensions.
2. **Segmentation patterns:** compare target rates across clinical, demographic, socioeconomic, access, and lifestyle segments.
3. **Relationship patterns:** inspect correlations, risk-count features, interactions, and train/test distribution drift.
""",
    )

    write(
        ANALYSIS_DIR / "02-temporal-patterns.md",
        f"""
# Temporal Pattern Exploration

## Objective

Discover trends, seasonality, cycles, or irregular ordered patterns. This dataset has no collection date, timestamp, year, month, or visit time column, so true temporal analysis is not possible from the provided files.

## Exploration Approach

1. Confirm no true temporal fields exist.
2. Use `Age` as an ordered life-stage proxy, not a timestamp.
3. Check `ID` deciles for hidden row/order effects that could indicate split artifacts.

## Analysis 1: Temporal Field Availability

### Rationale

Before looking for time trends, verify whether a temporal axis exists.

### Query

```sql
PRAGMA table_info(train_raw);
```

### Results

No column represents date, time, survey year, month, day, or event timestamp. `Age` is an attribute of the respondent; `ID` is an identifier.

### Observations

- No seasonality or period-over-period analysis can be supported by the available columns.
- Avoid introducing pseudo-temporal assumptions from row order unless they validate against target stability.

## Analysis 2: Age-Ordered Risk

### Rationale

Age is not temporal coverage, but it is an ordered predictor and often captures cumulative cardiovascular risk.

### Query

```sql
SELECT
  CASE
    WHEN Age <= 34 THEN '<=34'
    WHEN Age <= 44 THEN '35-44'
    WHEN Age <= 54 THEN '45-54'
    WHEN Age <= 64 THEN '55-64'
    ELSE '65+'
  END AS age_bin,
  COUNT(*) AS n,
  AVG("History of HeartDisease or Attack" = 'Yes') AS positive_rate
FROM train_labeled
GROUP BY age_bin;
```

### Results

{md_table(format_rate_columns(age_table))}

### Observations

- Positive rate rises monotonically with age bins.
- The `65+` bin has a materially higher rate than the youngest group, supporting nonlinear age features.

## Analysis 3: ID-Order Stability

### Rationale

If IDs are ordered by acquisition or split process, decile-level target shifts can reveal leakage risk or validation drift.

### Query

```sql
-- ID deciles computed with NTILE-style quantile bins in pandas.
SELECT id_decile, id_min, id_max, n, positive_rate FROM derived_id_decile_rates;
```

### Results

{md_table(format_rate_columns(id_table))}

### Observations

- ID deciles are broadly stable relative to the {pct(BASE_RATE)} base rate.
- There is no obvious ID-order target leak strong enough to justify using raw `ID` as a model feature.

## Temporal Patterns Summary

### Key Findings

1. **No true temporal coverage:** no time-series trend/seasonality can be estimated.
2. **Age risk is ordered and nonlinear:** age bins should be modeled nonlinearly.
3. **ID order appears stable:** raw ID should remain an identifier, not a predictor.

### Interesting Anomalies

- The absence of temporal metadata is a modeling constraint rather than a data error.

### Questions Raised

- Would adding survey year/geography from source BRFSS metadata improve generalization? That cannot be answered from these local competition files.
""",
    )

    write(
        ANALYSIS_DIR / "03-segmentation-patterns.md",
        f"""
# Segmentation Pattern Exploration

## Objective

Discover how heart-disease target rate varies across clinical, demographic, socioeconomic, access, and lifestyle groups.

## Exploration Approach

1. Compute univariate target rates for every categorical feature.
2. Inspect ordered BMI and age segments.
3. Check high-rate cross-segments that may become interaction features.

## Analysis 1: Strongest Categorical Segments

### Rationale

Target-rate lift by category identifies features that should receive careful encoding and interaction treatment.

### Query

```sql
SELECT
  '<feature>' AS feature,
  '<level>' AS segment,
  COUNT(*) AS n,
  AVG("History of HeartDisease or Attack" = 'Yes') AS positive_rate
FROM train_labeled
GROUP BY feature, segment
HAVING COUNT(*) >= 50
ORDER BY positive_rate DESC;
```

### Results

{md_table(format_rate_columns(categorical.head(25)))}

### Observations

- Prior stroke, very poor/poor general health, difficulty walking, diabetes, high blood pressure, high cholesterol, male sex, low income, and low education all show elevated target rates.
- The strongest single segment is stroke history, with lift far above the base rate.

## Analysis 2: BMI Segments

### Rationale

BMI is continuous but clinically nonlinear; bins help check whether a tree model should receive explicit BMI class features.

### Query

```sql
SELECT bmi_class, COUNT(*) AS n, AVG(target_yes) AS positive_rate
FROM derived_bmi_classes
GROUP BY bmi_class;
```

### Results

{md_table(format_rate_columns(bmi_table))}

### Observations

- BMI is not monotonic in the same clean way as age.
- Missing BMI has a lower target rate in train, but this missingness does not appear in test.

## Analysis 3: High-Rate Cross-Segments

### Rationale

Kaggle tabular scores often improve when important univariate signals are combined into compact interaction/risk-count features.

### Query

```sql
SELECT
  '<feature_a>=<level_a> | <feature_b>=<level_b>' AS interaction,
  COUNT(*) AS n,
  AVG("History of HeartDisease or Attack" = 'Yes') AS positive_rate
FROM train_labeled
GROUP BY interaction
HAVING COUNT(*) >= 300
ORDER BY positive_rate DESC
LIMIT 20;
```

### Results

{md_table(format_rate_columns(interactions))}

### Observations

- Stroke plus poor health/walking difficulty creates very high-risk regions.
- General health and walking difficulty interact strongly enough to justify explicit interaction features.
- Diabetes, blood pressure, cholesterol, and BMI should likely be grouped into a cardiometabolic feature family.

## Segmentation Patterns Summary

### Key Findings

1. Clinical history features dominate target-rate lift.
2. General health and mobility variables are among the strongest non-diagnosis survey signals.
3. Socioeconomic variables encode meaningful risk gradients and should be treated as ordered, not just nominal.

### Notable Differences

- Positive rates for high-risk clinical segments are several times the base rate.
- Male positive rate is higher than female positive rate, but sex is weaker than stroke/general health/walking difficulty.

### Questions Raised

- Which categorical encodings best preserve ordinal structure without leaking target statistics?
- Do explicit clinical-risk interactions improve OOF PR-AUC/F1 beyond native tree splits?
""",
    )

    write(
        ANALYSIS_DIR / "04-relationship-patterns.md",
        f"""
# Relationship Pattern Exploration

## Objective

Discover correlations, associations, dependencies, and drift patterns that affect feature engineering and validation.

## Exploration Approach

1. Encode simple clinical/demographic indicators and compute correlation with the target.
2. Derive risk-count features and inspect target rates.
3. Compare train/test distributions for drift.

## Analysis 1: Correlation With Target

### Rationale

Simple correlations are not a final model, but they reveal signal direction and sanity-check feature importance.

### Query

```sql
-- Binary/ordinal encodings were derived from train_labeled, then correlated with target_yes.
SELECT feature, corr_with_target
FROM derived_correlations
ORDER BY ABS(corr_with_target) DESC;
```

### Results

{md_table(format_rate_columns(correlations.head(20)))}

### Observations

- General health rank, difficulty walking, age, stroke, diabetes, high blood pressure, and cholesterol are directionally aligned with higher risk.
- Income and education have negative correlations with target after ordinal encoding: higher socioeconomic rank aligns with lower observed positive rate.

## Analysis 2: Risk Counts

### Rationale

Risk counts compress several sparse categorical predictors into dense features that can help both linear and tree models.

### Query

```sql
SELECT clinical_risk_count, COUNT(*) AS n, AVG(target_yes) AS positive_rate
FROM derived_clinical_risk_counts
GROUP BY clinical_risk_count;
```

### Results: Clinical Risk Count

{md_table(format_rate_columns(clinical))}

### Results: Lifestyle Protective Count

{md_table(format_rate_columns(lifestyle))}

### Observations

- Clinical risk count is strongly monotonic and should be a high-priority engineered feature.
- Lifestyle protective count has a weaker but useful gradient.

## Analysis 3: Train/Test Distribution Drift

### Rationale

Submission performance depends on generalizing from train to test. Distribution shifts flag features that may be risky or need careful imputation.

### Query

```sql
-- For categorical features, compare the largest train/test level-frequency difference.
-- For numeric features, compare missingness and mean shift.
SELECT feature, largest_level_or_metric, train_pct, test_pct, abs_diff_pct
FROM derived_train_test_drift
ORDER BY abs_diff_pct DESC;
```

### Results

{md_table(format_rate_columns(drift.head(20)))}

### Numeric Mean Shifts

{md_table(numeric_shift, float_digits=2)}

### Observations

- The largest train/test differences are train-only missingness in high cholesterol and BMI.
- Test respondents are older on average than train respondents, so age-related validation checks are worth watching.
- Most non-missing categorical distributions are close enough for standard stratified CV, but missingness artifacts should be handled carefully.

## Relationship Patterns Summary

### Key Findings

1. **Clinical risk count is a compact, monotonic relationship feature.**
2. **General health, mobility, and stroke create high-signal interactions.**
3. **Train-only missingness is the largest drift source.**

### Correlations Identified

- Strong positive: general health rank, difficulty walking, age, diagnosed stroke, diabetes.
- Moderate positive: high blood pressure, high cholesterol, smoking, male sex.
- Negative: income rank, education rank, physical activity.

### Questions Raised

- Do risk-count and interaction features improve OOF PR-AUC/F1 in LightGBM/CatBoost?
- Should missingness flags be included, excluded, or regularized because test has no missing feature values?
""",
    )

    write(
        ANALYSIS_DIR / "05 - anomaly-investigation.md",
        f"""
# Anomaly Investigation

## Anomalies Identified

1. **Train-only feature missingness** - found in data familiarization and train/test drift.
2. **Missing/blank target rows** - found in target balance profiling.
3. **Blank local sample submission targets** - found in submission integrity checks.
4. **No temporal fields** - found in temporal exploration.
5. **Very high-risk but small clinical intersections** - found in segmentation cross-segments.

## Anomaly 1: Train-Only Feature Missingness

### Where Found

`01 - data-familiarization.md` and `04-relationship-patterns.md`.

### Why It's Anomalous

Test has no missing feature values, while train has missing values in important predictors.

### Investigation Query

```sql
SELECT
  COUNT(*) AS train_rows,
  SUM("Told High Cholesterol" IS NULL) AS missing_cholesterol,
  SUM("Body Mass Index" IS NULL) AS missing_bmi
FROM train_raw;
```

### Results

{md_table(format_rate_columns(missing_summary.head(8)))}

### Explanation

**Determination:** data quality/split artifact.

**Reasoning:** The missingness is concentrated in train and absent from test, so a model that leans heavily on missingness flags may overfit train-only data collection artifacts.

**Action:** Preserve missingness during CV, but test model variants with and without missingness flags. For BMI use numeric imputation or tree-native missing support; for categorical missingness use a `Missing` level but monitor feature importance.

## Anomaly 2: Missing/Blank Target Rows

### Where Found

Target balance profiling.

### Why It's Anomalous

{(~target_mask(train)).sum():,} training rows lack supervised labels.

### Investigation Query

```sql
SELECT
  COUNT(*) AS rows,
  SUM("{TARGET_COLUMN}" IS NULL OR trim("{TARGET_COLUMN}") = '') AS missing_target
FROM train_raw;
```

### Results

| metric | value |
|---|---:|
| train rows | {len(train):,} |
| missing/blank target rows | {(~target_mask(train)).sum():,} |
| labeled rows | {len(labeled):,} |

### Explanation

**Determination:** data quality issue for supervised training.

**Action:** Exclude these rows from supervised model fitting and OOF metrics. They could be used only in a deliberate semi-supervised experiment.

## Anomaly 3: Blank Sample Submission Targets

### Where Found

Submission integrity checks.

### Why It's Anomalous

The local `sample_submission.csv` has the required shape but {sample_submission_blanks(sample):,} blank target values.

### Investigation Query

```sql
SELECT COUNT(*) AS rows
FROM sample_submission_raw
WHERE "{TARGET_COLUMN}" IS NULL OR trim("{TARGET_COLUMN}") = '';
```

### Results

| metric | value |
|---|---:|
| sample rows | {len(sample):,} |
| blank target values | {sample_submission_blanks(sample):,} |

### Explanation

**Determination:** expected sample-submission placeholder.

**Action:** A valid generated submission must fill every row with `Yes` or `No` and keep test ID order unchanged.

## Anomaly 4: No Temporal Fields

### Where Found

Temporal exploration.

### Why It's Anomalous

The exploratory-analysis process expects temporal exploration, but the dataset has no time axis.

### Explanation

**Determination:** real limitation of competition files, not an error.

**Action:** Do not invent seasonality. Use age as an ordered health feature, and keep validation stratified rather than time-based.

## Anomaly 5: High-Risk Small Clinical Intersections

### Where Found

Cross-segmentation analysis.

### Why It's Anomalous

Some intersections have very high positive rates but limited sample sizes, increasing overfit risk for target encoding or hand-built rules.

### Results

{md_table(format_rate_columns(interactions.head(10)))}

### Explanation

**Determination:** real phenomenon with modeling caveat.

**Action:** Use interactions as model features, not hard rules. Validate with out-of-fold metrics and avoid target encoding without leakage-safe folds.

## Anomalies Summary

### Real Phenomena

1. No temporal metadata in the competition files.
2. High-risk clinical intersections among stroke, general health, walking difficulty, diabetes, blood pressure, and cholesterol.

### Data Quality Issues

1. Train-only predictor missingness.
2. Missing target rows in train.
3. Blank target values in sample submission.

### Implications for Pattern Discovery

- Missing target rows must be excluded from supervised EDA and modeling.
- Missingness itself may be signal in train, but it is also a drift source because test has no missing predictors.
- Strong clinical interactions are useful candidates for feature engineering but need OOF validation.
""",
    )

    write(
        ANALYSIS_DIR / "06 - insights.md",
        f"""
# Insights

## Insight Criteria

For this exploration, an insight must be:
- **Actionable:** informs modeling, validation, feature engineering, or submission generation.
- **Surprising:** non-obvious or easy to mishandle.
- **Meaningful:** large enough to affect Kaggle score.

## Insight 1: Clinical Risk Count Is A High-Value Engineered Feature

### The Finding

Combining high blood pressure, high cholesterol, stroke, diabetes, and walking difficulty into a clinical risk count creates a strong monotonic gradient.

### Why It's Significant

This compact feature gives linear/diversity models a signal that trees can learn but may still benefit from explicitly. It also creates a stable feature family for ensembling.

### Supporting Evidence

Found in: `04-relationship-patterns.md`.

{md_table(format_rate_columns(clinical))}

### Caveats and Limitations

Clinical risk count is not causal proof and may duplicate information already learned by tree splits.

### Confidence Level

**Confidence:** High.

**Reasoning:** The pattern is monotonic, uses large groups, and aligns with multiple univariate signals.

### Recommended Action

- Add `clinical_risk_count`, `cardiometabolic_cluster`, `age_x_high_bp`, and `general_health_x_walking` features.
- Validate feature impact against OOF PR-AUC/F1, not public leaderboard only.

## Insight 2: Train-Only Missingness Can Help CV And Hurt Test Generalization

### The Finding

Train has missing values in important predictors, especially `Told High Cholesterol` and `Body Mass Index`; test has no missing predictors.

### Why It's Significant

Missingness may correlate with target in train, but the competition test distribution cannot use that same missingness signal. This can inflate CV if feature handling is too aggressive.

### Supporting Evidence

Found in: `01 - data-familiarization.md`, `04-relationship-patterns.md`, and `05 - anomaly-investigation.md`.

{md_table(format_rate_columns(drift.head(8)))}

### Caveats and Limitations

Because labels are unavailable for test, we cannot know whether missingness reflects true source-population differences or only data-entry artifacts.

### Confidence Level

**Confidence:** High.

**Reasoning:** The distribution mismatch is directly observed and affects high-signal predictors.

### Recommended Action

- Compare model variants with categorical `Missing` levels, numeric imputation, and missingness flags.
- If missingness flags increase OOF but shift feature importance strongly toward missing indicators, treat leaderboard gains cautiously.

## Insight 3: Age Needs Nonlinear Treatment

### The Finding

Age-bin target rates rise with age rather than behaving as a simple flat demographic variable.

### Why It's Significant

Linear models need bins/splines or interactions; tree models should still receive age bins because they are stable and interpretable.

### Supporting Evidence

Found in: `02-temporal-patterns.md`.

{md_table(format_rate_columns(age_table))}

### Confidence Level

**Confidence:** High.

**Reasoning:** The pattern is monotonic across large bins and domain-plausible.

### Recommended Action

- Add age bins and age interactions with high blood pressure, cholesterol, diabetes, and sex.
- Avoid using raw ID/order as a temporal proxy.

## Insight 4: Socioeconomic Variables Are Ordered Risk Signals

### The Finding

Income and education levels show target-rate gradients and negative correlation with heart-disease history when encoded as higher-is-better ranks.

### Why It's Significant

Treating income and education as purely nominal categories discards ordinal structure that may improve generalization.

### Supporting Evidence

Found in: `03-segmentation-patterns.md` and `04-relationship-patterns.md`.

{md_table(format_rate_columns(correlations[correlations["feature"].isin(["income_rank", "education_rank"])].head(5)))}

### Confidence Level

**Confidence:** Medium.

**Reasoning:** Direction is consistent, but survey socioeconomic categories can interact with age, health care access, and missingness.

### Recommended Action

- Add ordered encodings for income and education alongside raw categorical values.
- Test income/education interactions with general health and health-care access.

## Insight 5: Threshold Tuning Is Central To Submission Quality

### The Finding

The supervised positive rate is only {pct(BASE_RATE)}. Many high-risk segments are still minority-positive, so default `0.5` thresholds can under-predict positives for F1-like metrics.

### Why It's Significant

If the official metric rewards F1/balanced accuracy, the label threshold matters as much as small model AUC gains. If the metric is accuracy, the threshold should be different again.

### Supporting Evidence

Found in: target balance and segmentation rates.

| metric | value |
|---|---:|
| labeled rows | {len(labeled):,} |
| positive rows | {positive_count:,} |
| base positive rate | {pct(BASE_RATE)} |

### Confidence Level

**Confidence:** Medium.

**Reasoning:** The need for threshold tuning is clear, but the official competition metric must be verified in the Kaggle UI.

### Recommended Action

- Generate OOF probabilities, tune threshold on OOF only, and write labels using the selected threshold.
- Keep separate candidate thresholds for F1, balanced accuracy, and accuracy until the official metric is confirmed.

## Non-Insights

### Pattern: ID Deciles Are Broadly Stable

**Why not an insight:** useful sanity check, but it does not suggest using ID as a feature.

### Pattern: Lifestyle Features Have Signal

**Why not an insight:** actionability is weaker than clinical and socioeconomic features; include them, but they are not primary improvement levers.

## Insights Summary

### Highest Priority Insights

1. **Clinical risk count and interactions:** likely useful feature engineering for Kaggle score.
2. **Train-only missingness:** biggest data-quality risk for generalization.
3. **Threshold tuning:** required because target is imbalanced and submission labels are `Yes`/`No`.

### Cross-Cutting Themes

The dataset is an imbalanced, mixed categorical tabular problem where clinical history, self-reported health status, and ordered socioeconomic/demographic structure carry the strongest signal.

### Strategic Implications

Prioritize feature-engineered GBDTs and OOF thresholding over deep models or raw one-hot baselines. Treat missingness and public leaderboard feedback carefully.
""",
    )

    write(
        ANALYSIS_DIR / "07 - next-questions.md",
        f"""
# Questions for Further Investigation

## Purpose

Exploratory analysis identified patterns that should be converted into concrete validation experiments before trusting Kaggle submission changes.

## Question 1: Do Clinical Interaction Features Improve OOF Score?

### What We Learned in Exploration

Clinical risk count is monotonic and high-signal.

### The Question

Do `clinical_risk_count`, `cardiometabolic_cluster`, `general_health_x_walking`, `age_x_high_bp`, and BMI/age bins improve 5-fold OOF ROC-AUC, PR-AUC, and F1-threshold proxy versus the current feature set?

### Why It Matters

These features are the most actionable route from EDA to leaderboard gain.

### Recommended Process

**Process Skill:** `hypothesis-testing` or a controlled modeling experiment.

### Data Required

- Current train/test CSVs.
- OOF predictions from baseline and feature-engineered models.

### Investigation Approach

1. Train baseline model with existing features.
2. Train same model/seed/folds with EDA features added.
3. Compare OOF ROC-AUC, PR-AUC, best F1, threshold, and test prediction distribution.

### Priority

**Priority:** High.

## Question 2: Should Missingness Flags Be Used?

### What We Learned in Exploration

Train has missing high cholesterol/BMI values; test does not.

### The Question

Do missingness flags improve real generalization, or do they overfit train-only artifacts?

### Why It Matters

This is the largest train/test drift source and can create misleading OOF gains.

### Recommended Process

**Process Skill:** `comparative-analysis`.

### Data Required

- Current train/test CSVs.
- Same fold split across variants.

### Investigation Approach

Compare three variants: no missing flags, missing category/imputation only, and explicit missing flags. Inspect OOF metrics and feature importance.

### Priority

**Priority:** High.

## Question 3: What Threshold Matches The Official Metric?

### What We Learned in Exploration

The target is imbalanced at {pct(BASE_RATE)}, and the submission requires labels.

### The Question

What OOF-tuned threshold should be used after verifying whether Kaggle scores F1, accuracy, balanced accuracy, or another label-based metric?

### Why It Matters

The same probability model can produce very different leaderboard scores depending on threshold.

### Recommended Process

**Process Skill:** `guided-investigation`.

### Data Required

- Official metric from Kaggle UI.
- OOF probabilities for top models.

### Investigation Approach

Tune threshold only on OOF predictions for the official metric proxy; do not tune threshold from public leaderboard feedback.

### Priority

**Priority:** High.

## Question 4: Which Categorical Strategy Wins?

### What We Learned in Exploration

Many important predictors are categorical strings, with some ordinal structure.

### The Question

Does native categorical handling, one-hot encoding, ordered encodings, or leakage-safe target encoding perform best across LightGBM/CatBoost/XGBoost?

### Why It Matters

Encoding choices can dominate tabular model performance.

### Recommended Process

**Process Skill:** `comparative-analysis`.

### Data Required

- Current train/test CSVs.
- Shared fold splits.

### Investigation Approach

Run model-family comparisons with identical folds and feature groups.

### Priority

**Priority:** Medium.

## Question 5: Can A Diverse Ensemble Improve PR-AUC/F1?

### What We Learned in Exploration

Signals include nonlinear clinical interactions and ordered socioeconomic gradients.

### The Question

Can LightGBM + CatBoost + logistic/diversity model OOF blending improve the official-metric proxy?

### Why It Matters

Ensembling often improves Kaggle tabular stability if base models make different errors.

### Recommended Process

**Process Skill:** `hypothesis-testing`.

### Data Required

- OOF/test predictions from candidate models.

### Investigation Approach

Blend OOF probabilities with constrained weights, tune threshold from blended OOF, and then apply the same blend/threshold to test.

### Priority

**Priority:** Medium.

## Questions Summary

### High Priority Questions

1. Do clinical interaction features improve OOF score?
2. Should missingness flags be used?
3. What threshold matches the official metric?

### Medium Priority Questions

1. Which categorical strategy wins?
2. Can a diverse ensemble improve PR-AUC/F1?

### Investigation Roadmap

1. Lock a stratified 5-fold split and baseline OOF metric table.
2. Run feature-engineering ablations from this EDA.
3. Run missingness handling comparison.
4. Verify official metric in Kaggle UI and tune threshold from OOF.
5. Blend top candidates only after individual gains are OOF-supported.

## Data Collection Priorities

Official metric verification is the highest-value missing external information. Survey year/geography would be useful but is not available in the competition files.
""",
    )


if __name__ == "__main__":
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    train_df = read_csv("train")
    test_df = read_csv("test")
    sample_df = read_csv("sample_submission")
    labeled_df = train_df.loc[target_mask(train_df)].copy()
    BASE_RATE = positive_rate(labeled_df)
    rebuild_sqlite(train_df, test_df, sample_df)
    write_reports(train_df, test_df, sample_df)
    print(f"Wrote EDA reports to {ANALYSIS_DIR}")
    print(f"Wrote SQLite database to {DB_PATH}")
