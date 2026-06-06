
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

| age_bin | n | positive_rate | lift_vs_base | bar |
| --- | --- | --- | --- | --- |
| <=34 | 35394 | 0.71% | 0.09x | # |
| 35-44 | 28492 | 1.79% | 0.22x | ### |
| 45-54 | 38617 | 4.43% | 0.54x | ######## |
| 55-64 | 50997 | 8.26% | 1.01x | ############## |
| 65+ | 67890 | 16.78% | 2.06x | ############################# |

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

| id_decile | id_min | id_max | n | positive_rate | bar |
| --- | --- | --- | --- | --- | --- |
| 1 | 1 | 22317 | 22139 | 7.82% | ################# |
| 2 | 22318 | 44621 | 22139 | 7.67% | ################# |
| 3 | 44622 | 66903 | 22139 | 8.14% | ################## |
| 4 | 66904 | 89209 | 22139 | 8.76% | ################### |
| 5 | 89210 | 111502 | 22139 | 8.04% | ################# |
| 6 | 111503 | 133845 | 22139 | 7.96% | ################# |
| 7 | 133846 | 156145 | 22139 | 8.22% | ################## |
| 8 | 156146 | 178480 | 22139 | 9.21% | #################### |
| 9 | 178481 | 200781 | 22139 | 7.33% | ################ |
| 10 | 200782 | 223084 | 22139 | 8.46% | ################## |

### Observations

- ID deciles are broadly stable relative to the 8.16% base rate.
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
