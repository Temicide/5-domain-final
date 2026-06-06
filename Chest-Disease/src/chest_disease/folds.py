from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from .config import LABEL_COLUMNS


def make_multilabel_folds(frame: pd.DataFrame, num_folds: int, seed: int) -> pd.DataFrame:
    result = frame.copy()
    y = result[LABEL_COLUMNS].astype(int).to_numpy()
    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

        splitter = MultilabelStratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
        splits = splitter.split(np.zeros(len(result)), y)
    except Exception:
        splitter = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
        splits = splitter.split(result)
    result["fold"] = -1
    for fold, (_, valid_idx) in enumerate(splits):
        result.loc[result.index[valid_idx], "fold"] = fold
    if (result["fold"] < 0).any():
        raise RuntimeError("fold assignment failed")
    return result
