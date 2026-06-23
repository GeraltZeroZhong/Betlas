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
    required_labels = set(label_counts)

    def missing_coverage(
        splits: list[tuple[np.ndarray, np.ndarray]],
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for fold, (train_idx, test_idx) in enumerate(splits, start=1):
            for split_name, idx in (("train", train_idx), ("test", test_idx)):
                present = set(y_arr[idx].tolist())
                missing = sorted(required_labels - present)
                if missing:
                    rows.append({"fold": fold, "split": split_name, "missing_classes": missing})
        return rows

    def complete_or_none(candidate: int) -> tuple[list[tuple[np.ndarray, np.ndarray]], str] | None:
        if StratifiedGroupKFold is not None and candidate <= min_label_count:
            splitter = StratifiedGroupKFold(
                n_splits=candidate,
                shuffle=True,
                random_state=random_state,
            )
            strategy = "stratified_group_kfold"
        else:
            splitter = GroupKFold(n_splits=candidate)
            strategy = "group_kfold"
        splits = list(splitter.split(X, y_arr, groups_arr))
        if not missing_coverage(splits):
            return splits, strategy
        return None

    first_missing: list[dict[str, Any]] = []
    for candidate in range(requested, 1, -1):
        result = complete_or_none(candidate)
        if result is not None:
            splits, strategy = result
            if candidate < requested:
                strategy = f"{strategy}_class_complete_downshifted_from_{requested}"
            return splits, strategy, candidate
        if not first_missing:
            if StratifiedGroupKFold is not None and candidate <= min_label_count:
                probe = list(
                    StratifiedGroupKFold(
                        n_splits=candidate,
                        shuffle=True,
                        random_state=random_state,
                    ).split(X, y_arr, groups_arr)
                )
            else:
                probe = list(GroupKFold(n_splits=candidate).split(X, y_arr, groups_arr))
            first_missing = missing_coverage(probe)

    examples = "; ".join(
        f"fold {row['fold']} {row['split']} missing {row['missing_classes']}"
        for row in first_missing[:5]
    )
    more = f"; plus {len(first_missing) - 5} more" if len(first_missing) > 5 else ""
    raise ValueError(
        "grouped cross-validation cannot produce train/test folds with complete class coverage "
        f"for observed labels {sorted(required_labels)} at any split count >=2: {examples}{more}"
    )


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
