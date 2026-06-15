from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

try:  # pragma: no cover - fallback is for older sklearn only
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:  # pragma: no cover
    StratifiedGroupKFold = None  # type: ignore[assignment]


def make_grouped_splits(
    X: Any,
    y: Any,
    groups: Any,
    *,
    n_splits: int,
    random_state: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], str, int]:
    y_arr = np.asarray(y)
    groups_arr = np.asarray(groups)
    requested = min(int(n_splits), len(np.unique(groups_arr)))
    if requested < 2:
        raise ValueError("grouped split needs at least two unique groups")

    label_counts = Counter(y_arr.tolist())
    min_label_count = min(label_counts.values()) if label_counts else 0
    stratified_splits = min(requested, min_label_count)
    if StratifiedGroupKFold is not None and stratified_splits >= 2:
        splitter = StratifiedGroupKFold(
            n_splits=stratified_splits,
            shuffle=True,
            random_state=random_state,
        )
        return list(splitter.split(X, y_arr, groups_arr)), "stratified_group_kfold", stratified_splits

    splitter = GroupKFold(n_splits=requested)
    return list(splitter.split(X, y_arr, groups_arr)), "group_kfold", requested


def fold_label_counts(
    *,
    record_ids: Any,
    labels: Any,
    groups: Any,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    record_arr = np.asarray(record_ids)
    label_arr = np.asarray(labels)
    group_arr = np.asarray(groups)
    rows: list[dict[str, Any]] = []
    for fold, (_train_idx, test_idx) in enumerate(splits, start=1):
        fold_labels = Counter(label_arr[test_idx].tolist())
        rows.append(
            {
                "fold": fold,
                "n_records": int(len(test_idx)),
                "n_groups": int(len(set(group_arr[test_idx].tolist()))),
                "record_ids": ";".join(str(value) for value in sorted(record_arr[test_idx].tolist())),
                **{f"label_count_{label}": int(count) for label, count in sorted(fold_labels.items())},
            }
        )
    return pd.DataFrame(rows)
