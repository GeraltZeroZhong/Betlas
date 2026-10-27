"""Frozen, coverage-aware classification of caller-supplied CATH labels.

Splitting never reads features. Fit preprocessing and weights on training rows
only, and reuse the saved split table for every representation. These commands
run CPU baselines with fixed hyperparameters; they do not tune on held-out data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from betlas.ml.benchmark import ID_COLUMNS
from betlas.features.current import FEATURE_COLUMNS

A = "cath_architecture_code"
T = "cath_topology_code"
H = "cath_homology_code"
S = "cath_s35_cluster_id"
MANIFEST_COLUMNS = ("record_id", "pdb_id", A, T, H, S)
WEIGHTINGS = ("none", "inverse", "sqrt", "effective", "group-balanced")
ROLES = {"train", "test", "unavailable", "purged"}


def _require(table: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    missing = set(columns) - set(table.columns)
    if missing:
        raise ValueError(f"{name} lacks columns: {sorted(missing)}")
    for column in columns:
        values = table[column]
        if values.isna().any() or values.astype(str).str.strip().eq("").any():
            raise ValueError(f"{name}.{column} contains missing values")


def _manifest(table: pd.DataFrame) -> pd.DataFrame:
    _require(table, MANIFEST_COLUMNS, "manifest")
    result = table.copy().reset_index(drop=True)
    for column in MANIFEST_COLUMNS:
        result[column] = result[column].astype(str)
    result["pdb_id"] = result["pdb_id"].str.lower()
    if result.record_id.duplicated().any():
        raise ValueError("manifest record_id must be unique")
    if result.empty:
        raise ValueError("manifest must not be empty")
    for child, parent in ((T, A), (H, T), (S, H)):
        if result.groupby(child)[parent].nunique().gt(1).any():
            raise ValueError(f"{child} must be a complete key with one {parent} parent")
    return result


def _source_available(table: pd.DataFrame) -> pd.Series:
    """Use only declared input QC, never extractor/model success or scores."""
    available = pd.Series(True, index=table.index)
    for column in ("representative_status", "coordinate_status"):
        if column in table:
            available &= ~table[column].astype(str).str.strip().str.lower().eq("unavailable")
    return available


def connected_groups(table: pd.DataFrame, group_column: str) -> pd.Series:
    """Union all records sharing an explicit hierarchy key OR a PDB entry.

Call on the full manifest before mapping to representatives: an unselected
domain can bridge two otherwise separate representatives. No hashes or fallback
group types are used. IDs are the smallest record_id in each component.
"""
    frame = _manifest(table)
    if group_column not in (S, H, T):
        raise ValueError(f"group_column must be one of {(S, H, T)}")
    parent = list(range(len(frame)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for column in (group_column, "pdb_id"):
        first: dict[str, int] = {}
        for index, value in enumerate(frame[column]):
            if value in first:
                left, right = find(index), find(first[value])
                parent[left] = right
            else:
                first[value] = index
    roots = [find(index) for index in range(len(frame))]
    names: dict[int, str] = {}
    for root, identifier in zip(roots, frame.record_id, strict=True):
        names[root] = min(names.get(root, identifier), identifier)
    return pd.Series([names[root] for root in roots], index=table.index, name="group_id")


def _mapped_groups(
    manifest: pd.DataFrame, group_column: str, grouping_manifest: pd.DataFrame | None,
    pdb_policy: str = "connected",
) -> pd.Series:
    universe = _manifest(grouping_manifest if grouping_manifest is not None else manifest)
    source = universe.set_index("record_id")
    chosen = manifest.set_index("record_id")
    if not chosen.index.isin(source.index).all():
        raise ValueError("grouping manifest must contain every selected record")
    if not chosen[list(MANIFEST_COLUMNS[1:])].equals(
        source.loc[chosen.index, list(MANIFEST_COLUMNS[1:])]
    ):
        raise ValueError("grouping manifest metadata differs from selected manifest")
    if pdb_policy not in ("connected", "purge"):
        raise ValueError("pdb_policy must be connected or purge")
    if group_column not in (S, H, T):
        raise ValueError(f"group_column must be one of {(S, H, T)}")
    values = connected_groups(universe, group_column).to_numpy() if pdb_policy == "connected" else universe[group_column].to_numpy()
    groups = pd.Series(values, index=universe.record_id)
    return manifest.record_id.map(groups)


def make_grouped_folds(
    manifest: pd.DataFrame,
    *,
    label_column: str = A,
    group_column: str = H,
    n_splits: int = 5,
    seed: int = 13,
    grouping_manifest: pd.DataFrame | None = None,
    pdb_policy: str = "connected",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return frozen long-form roles and per-class support, including singletons.

    Greedy allocation balances hierarchy/group counts, not duplicated domains.
    Connected mode unions hierarchy and PDB edges before assignment. Purge mode
    assigns hierarchy keys, freezes test members, then removes training PDBs
    shared with ANY full-catalog member of the held-out keys. Classes losing all
    training support remain unavailable, without reallocating the test set.
"""
    frame = _manifest(manifest)
    if label_column not in (A, T):
        raise ValueError("label_column must be a native architecture or topology column")
    if n_splits < 2:
        raise ValueError("n_splits must be at least two")
    frame["group_id"] = _mapped_groups(frame, group_column, grouping_manifest, pdb_policy)
    universe = _manifest(grouping_manifest if grouping_manifest is not None else frame)
    frame["input_source_available"] = _source_available(frame)
    ready = frame.loc[frame.input_source_available]
    incidence = pd.crosstab(ready.group_id, ready[label_column]).clip(upper=1)
    counts = incidence.sum(axis=0)
    # An all-unavailable input still produces a complete audit ledger.
    folds = max(1, min(n_splits, len(incidence)))
    rng = np.random.default_rng(seed)
    ties = dict(zip(incidence.index, rng.random(len(incidence)), strict=True))
    order = sorted(incidence.index, key=lambda g: (-float((incidence.loc[g] / counts).sum()), ties[g], g))
    loads = np.zeros((folds, len(counts)))
    sizes = np.zeros(folds)
    assignment: dict[str, int] = {}
    for group in order:
        contribution = incidence.loc[group].to_numpy(dtype=float) / counts.to_numpy()
        costs = ((loads + contribution) ** 2 - loads ** 2).sum(axis=1)
        costs += 1e-6 * sizes
        selected = int(np.argmin(costs))
        assignment[group] = selected
        loads[selected] += contribution
        sizes[selected] += 1
    frame["test_fold"] = frame.group_id.map(assignment)
    support = frame.groupby(label_column, sort=True).agg(
        n_records=("record_id", "size"), n_groups_before_input_qc=("group_id", "nunique")
    )
    available_support = frame.loc[frame.input_source_available].groupby(label_column).agg(
        n_input_available=("record_id", "size"), n_groups=("group_id", "nunique"), n_test_folds=("test_fold", "nunique"),
    )
    support = support.join(available_support).fillna(0).astype(int).reset_index().rename(columns={label_column: "label"})
    support["n_input_unavailable"] = support.n_records - support.n_input_available
    support["status"] = np.select(
        [support.n_input_available.eq(0), support.n_groups.lt(2) & support.n_groups_before_input_qc.ge(2),
         support.n_groups.lt(2), support.n_test_folds.lt(2)],
        ["input_source_unavailable", "insufficient_groups_after_input_qc", "single_independent_group", "allocation_not_separable"], default="eligible",
    )
    support["label_column"] = label_column
    support["group_column"] = group_column
    support["pdb_policy"] = pdb_policy
    reasons = support.set_index("label").status
    frame["eligibility_reason"] = frame[label_column].map(reasons)
    frame.loc[~frame.input_source_available, "eligibility_reason"] = "input_source_unavailable"
    rows = []
    for fold in range(folds):
        part = frame[list(MANIFEST_COLUMNS) + ["group_id", "eligibility_reason", "input_source_available"]].copy()
        part["split_id"] = f"fold-{fold}"
        part["role"] = np.where(frame.test_fold.eq(fold), "test", "train")
        part.loc[frame.eligibility_reason.ne("eligible"), "role"] = "unavailable"
        part["held_out"] = part.role.eq("test")
        if pdb_policy == "purge":
            held_out_keys = set(part.loc[part.role.eq("test"), "group_id"])
            held_out_pdbs = set(universe.loc[universe[group_column].isin(held_out_keys), "pdb_id"])
            purged = part.role.eq("train") & part.pdb_id.isin(held_out_pdbs)
            part.loc[purged, "role"] = "purged"
            part.loc[purged, "eligibility_reason"] = "shared_full_catalog_test_pdb"
            training_labels = set(part.loc[part.role.eq("train"), label_column])
            unsupported = part.role.eq("test") & ~part[label_column].isin(training_labels)
            part.loc[unsupported, "role"] = "unavailable"
            part.loc[unsupported, "eligibility_reason"] = "class_absent_after_pdb_purge"
        part["label_column"] = label_column
        part["group_column"] = group_column
        part["pdb_policy"] = pdb_policy
        part["target_label"] = frame[label_column]
        rows.append(part)
    result = pd.concat(rows, ignore_index=True)
    tested = result[result.role.eq("test")]
    support["n_test_folds"] = support.label.map(tested.groupby("target_label").split_id.nunique()).fillna(0).astype(int)
    support["n_evaluable_test_records"] = support.label.map(tested.groupby("target_label").record_id.nunique()).fillna(0).astype(int)
    originally_eligible = support.status.eq("eligible")
    support.loc[originally_eligible & support.n_evaluable_test_records.eq(0), "status"] = "not_evaluable_after_pdb_purge"
    support.loc[originally_eligible & support.n_evaluable_test_records.gt(0) & support.n_evaluable_test_records.lt(support.n_input_available), "status"] = "partially_evaluable_after_pdb_purge"
    return result, support


def make_rare_topology_challenge(
    manifest: pd.DataFrame,
    *,
    max_homologies: int = 4,
    seed: int = 13,
    grouping_manifest: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Freeze an unseen-topology challenge using labels only, then purge PDBs.

Sample approximately half the low-H topologies within architecture and H=1
versus H=2..limit strata. Keep two development topologies where feasible and
at least one otherwise. A missing architecture after purge is unavailable,
never silently returned to training. Challenge roles are for A prediction.
"""
    frame = _manifest(manifest)
    universe = _manifest(grouping_manifest if grouping_manifest is not None else manifest)
    _mapped_groups(frame, T, universe)  # Also checks universe membership and metadata.
    if max_homologies < 1:
        raise ValueError("max_homologies must be positive")
    catalog = universe.groupby([A, T], sort=True).agg(n_h=(H, "nunique"), n_s=(S, "nunique")).reset_index()
    rng = np.random.default_rng(seed)
    selected: set[str] = set()
    for _, subset in catalog.groupby(A, sort=True):
        candidates = subset[subset.n_h.le(max_homologies)].copy()
        candidates["stratum"] = np.where(candidates.n_h.eq(1), "one", "multiple")
        capacity = max(0, len(subset) - (2 if len(subset) >= 3 else 1))
        proposed: list[str] = []
        for _, stratum in candidates.groupby("stratum", sort=True):
            values = rng.permutation(sorted(stratum[T])).tolist()
            proposed.extend(values[:max(1, int(np.ceil(len(values) / 2)))])
        if capacity:
            selected.update(rng.permutation(proposed).tolist()[:capacity])
    test_universe = universe[T].isin(selected)
    test_pdbs = set(universe.loc[test_universe, "pdb_id"])
    frame["role"] = np.select(
        [frame[T].isin(selected), frame.pdb_id.isin(test_pdbs)], ["test", "purged"], default="train",
    )
    frame["held_out"] = frame.role.eq("test")
    frame["pdb_policy"] = "challenge_purge"
    frame["input_source_available"] = _source_available(frame)
    frame["eligibility_reason"] = "eligible"
    frame.loc[frame.role.eq("purged"), "eligibility_reason"] = "shared_test_pdb"
    frame.loc[~frame.input_source_available, "role"] = "unavailable"
    frame.loc[~frame.input_source_available, "eligibility_reason"] = "input_source_unavailable"
    train_architectures = set(frame.loc[frame.role.eq("train"), A])
    unsupported = frame.role.eq("test") & ~frame[A].isin(train_architectures)
    frame.loc[unsupported, "eligibility_reason"] = "architecture_absent_after_purge"
    frame.loc[unsupported, "role"] = "unavailable"
    # PDB-purged domains are removed before constructing this challenge's
    # components. Unlike cross-validation, this is an explicit purge protocol;
    # discarded bridge domains cannot remain training observations.
    active_universe = universe[universe[T].isin(selected) | ~universe.pdb_id.isin(test_pdbs)].copy()
    active_groups = pd.Series(
        connected_groups(active_universe, T).to_numpy(), index=active_universe.record_id
    )
    frame["group_id"] = frame.record_id.map(active_groups).fillna("purged:" + frame.record_id)
    frame["split_id"] = "rare-topology-challenge"
    frame["label_column"], frame["group_column"] = A, T
    frame["target_label"] = frame[A]
    catalog["selected_for_challenge"] = catalog[T].isin(selected)
    catalog["low_sequence_support"] = catalog.n_s.le(max_homologies)
    catalog["training_architecture_records_after_purge"] = catalog[A].map(
        frame.loc[frame.role.eq("train")].groupby(A).size()
    ).fillna(0).astype(int)
    catalog["training_architecture_topologies_after_purge"] = catalog[A].map(
        frame.loc[frame.role.eq("train")].groupby(A)[T].nunique()
    ).fillna(0).astype(int)
    catalog["selected_input_available_records"] = catalog[T].map(
        frame.loc[frame.input_source_available].groupby(T).size()
    ).fillna(0).astype(int)
    catalog["label_column"], catalog["group_column"] = A, T
    return frame, catalog


def training_weights(
    labels: Sequence[str], *, method: str = "none", groups: Sequence[str] | None = None, beta: float = 0.999
) -> np.ndarray:
    """Training-only weights, normalized to mean one; no double rebalancing."""
    if method not in WEIGHTINGS:
        raise ValueError(f"unknown weighting: {method}")
    y = pd.Series(list(labels), dtype=str)
    if y.empty or y.eq("").any():
        raise ValueError("training labels must be nonempty")
    count = y.map(y.value_counts()).to_numpy(dtype=float)
    if method == "none":
        weight = np.ones(len(y))
    elif method == "inverse":
        weight = 1 / count
    elif method == "sqrt":
        weight = 1 / np.sqrt(count)
    elif method == "effective":
        if not 0 <= beta < 1:
            raise ValueError("effective-number beta must be in [0, 1)")
        weight = np.ones(len(y)) if beta == 0 else (1 - beta) / -np.expm1(count * np.log(beta))
    else:
        if groups is None or len(groups) != len(y):
            raise ValueError("group-balanced weighting requires one training group per row")
        table = pd.DataFrame({"label": y, "group": list(groups)})
        _require(table, ["group"], "training")
        per_group = table.groupby(["label", "group"])["label"].transform("size")
        per_class = table["label"].map(table.groupby("label")["group"].nunique())
        weight = (1 / (per_group * per_class)).to_numpy()
    return weight / weight.mean()


def _features(
    manifest: pd.DataFrame, features: pd.DataFrame, columns: Sequence[str] | None
) -> tuple[pd.DataFrame, list[str]]:
    _require(features, ["record_id"], "features")
    features = features.copy()
    features["record_id"] = features.record_id.astype(str)
    if features.record_id.duplicated().any():
        raise ValueError("feature record_id must be unique")
    selected = list(columns) if columns is not None else list(FEATURE_COLUMNS)
    forbidden = set(ID_COLUMNS) | set(MANIFEST_COLUMNS) | {"target_label", "label", "group_id", "split_id", "role"}
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("feature allowlist must be nonempty and unique")
    if any(col in forbidden or col.startswith("cath_") for col in selected):
        raise ValueError("feature allowlist contains label or identifier metadata")
    if set(selected) - set(features):
        raise ValueError("feature allowlist contains absent columns")
    payload = features[["record_id"] + selected].copy()
    for column in selected:
        payload[column] = pd.to_numeric(payload[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    valid = payload[selected].notna().any(axis=1)
    if "betlas_parse_ok" in features:
        valid &= features.betlas_parse_ok.astype(str).str.lower().isin(["true", "1", "1.0"])
    if "feature_status" in features:
        valid &= features.feature_status.eq("ok")
    payload["feature_available"] = valid
    joined = manifest.merge(payload, on="record_id", how="left", validate="one_to_one")
    joined["feature_available"] = joined.feature_available.eq(True)
    return joined, selected


def _pipeline(model: str, seed: int, n_jobs: int, n_estimators: int) -> Pipeline:
    steps = [("imputer", SimpleImputer(strategy="median", keep_empty_features=True))]
    if model == "logistic":
        steps += [("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=2000, random_state=seed))]
    elif model == "random_forest":
        steps += [("model", RandomForestClassifier(
            n_estimators=n_estimators, min_samples_leaf=2, class_weight=None, n_jobs=n_jobs, random_state=seed
        ))]
    else:
        raise ValueError("model must be logistic or random_forest")
    return Pipeline(steps)


def _validate_splits(manifest: pd.DataFrame, splits: pd.DataFrame, label_column: str) -> None:
    _require(splits, ["record_id", "split_id", "role", "group_id", "target_label", "label_column", "group_column"], "splits")
    if not set(splits.role).issubset(ROLES) or not splits.label_column.eq(label_column).all():
        raise ValueError("split role or label-column contract mismatch")
    if splits.duplicated(["split_id", "record_id"]).any():
        raise ValueError("duplicate record within a split")
    expected = manifest.set_index("record_id")
    for _, part in splits.groupby("split_id", sort=True):
        if set(part.record_id) != set(expected.index):
            raise ValueError("every split must retain every manifest record")
        indexed = part.set_index("record_id").loc[expected.index]
        if not indexed.target_label.eq(expected[label_column]).all():
            raise ValueError("frozen split labels differ from manifest")
        if indexed.group_column.nunique() != 1 or indexed.group_column.iloc[0] not in (S, H, T):
            raise ValueError("split has no single explicit grouping contract")
        train = indexed.role.eq("train")
        test = indexed.role.eq("test")
        if ((train | test) & ~_source_available(expected)).any():
            raise ValueError("frozen split includes known unavailable input sources in train/test")
        if set(indexed.loc[train, "group_id"]) & set(indexed.loc[test, "group_id"]):
            raise ValueError("connected group leakage in frozen split")
        for column in ("pdb_id", indexed.group_column.iloc[0]):
            if set(expected.loc[train, column]) & set(expected.loc[test, column]):
                raise ValueError(f"{column} leakage in frozen split")
        if not set(expected.loc[test, label_column]).issubset(set(expected.loc[train, label_column])):
            raise ValueError("test class absent from frozen training side")


def make_development_folds(
    manifest: pd.DataFrame, challenge: pd.DataFrame, *, label_column: str = A,
    group_column: str = H, n_splits: int = 5, seed: int = 13,
    grouping_manifest: pd.DataFrame | None = None,
    pdb_policy: str = "connected",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cross-validate only a frozen challenge's development side.

    Reserved challenge and purged rows remain in every fold's ledger. The full
    catalog supplies hidden bridges for connected mode and held-out PDBs for
    purge mode. Challenge labels never enter model selection.
"""
    frame = _manifest(manifest)
    if challenge.split_id.nunique() != 1:
        raise ValueError("development reservation requires one frozen challenge")
    _validate_splits(frame, challenge, A)
    roles = challenge.set_index("record_id").role
    development_mask = frame.record_id.map(roles).eq("train")
    if not development_mask.any():
        if not _source_available(frame).any():
            return make_grouped_folds(
                frame, label_column=label_column, group_column=group_column,
                n_splits=n_splits, seed=seed, grouping_manifest=grouping_manifest,
                pdb_policy=pdb_policy,
            )
        raise ValueError("frozen challenge leaves no development records")
    universe = grouping_manifest if grouping_manifest is not None else frame
    inner, support = make_grouped_folds(
        frame.loc[development_mask], label_column=label_column, group_column=group_column,
        n_splits=n_splits, seed=seed, grouping_manifest=universe,
        pdb_policy=pdb_policy,
    )
    reserved = frame.loc[~development_mask, list(MANIFEST_COLUMNS)].copy()
    full_groups = _mapped_groups(frame, group_column, universe, pdb_policy)
    reserved["group_id"] = full_groups.loc[reserved.index]
    reserved["role"] = np.where(reserved.record_id.map(roles).eq("purged"), "purged", "unavailable")
    reserved["eligibility_reason"] = "reserved_frozen_challenge"
    reserved.loc[reserved.role.eq("purged"), "eligibility_reason"] = "shared_frozen_challenge_pdb"
    reserved["input_source_available"] = _source_available(frame).loc[reserved.index]
    reserved.loc[~reserved.input_source_available, "role"] = "unavailable"
    reserved.loc[~reserved.input_source_available, "eligibility_reason"] = "input_source_unavailable"
    reserved["label_column"], reserved["group_column"] = label_column, group_column
    reserved["pdb_policy"] = pdb_policy
    reserved["held_out"] = False
    reserved["target_label"] = reserved[label_column]
    output = [inner]
    for split_id in sorted(inner.split_id.unique()):
        part = reserved.copy()
        part["split_id"] = split_id
        output.append(part)
    absent = sorted(set(frame[label_column]) - set(support.label))
    if absent:
        extra = pd.DataFrame({
            "label": absent, "n_records": 0, "n_groups": 0, "n_test_folds": 0,
            "status": "reserved_frozen_challenge", "label_column": label_column, "group_column": group_column,
            "pdb_policy": pdb_policy, "n_evaluable_test_records": 0,
            "n_groups_before_input_qc": 0, "n_input_available": 0, "n_input_unavailable": 0,
        })
        support = pd.concat([support, extra], ignore_index=True)
    support["n_reserved_records"] = support.label.map(reserved.groupby(label_column).size()).fillna(0).astype(int)
    return pd.concat(output, ignore_index=True), support


def run_classification(
    manifest: pd.DataFrame, features: pd.DataFrame, splits: pd.DataFrame,
    *, label_column: str = A, feature_columns: Sequence[str] | None = None,
    model: str = "logistic", weighting: str = "none", beta: float = 0.999,
    weight_group_column: str = H,
    seed: int = 13, n_jobs: int = 1, n_estimators: int = 200,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Fit each frozen training side and return test/failure/unavailable records.

    Feature failure never changes the saved split or removes a test denominator.
    Loss of a training class after extraction produces an evaluation flag; the
    predictor itself never uses held-out labels. No tuning is performed.
    Group-balanced weights equalize H groups within each training class by
    default; S35 groups are an explicit alternative for sequence redundancy.
"""
    frame = _manifest(manifest)
    if label_column not in (A, T):
        raise ValueError("label_column must be architecture or topology")
    if weight_group_column not in (H, S):
        raise ValueError("weight_group_column must be homology or S35 cluster")
    frozen = splits.copy()
    frozen["record_id"] = frozen.record_id.astype(str)
    frozen["target_label"] = frozen.target_label.astype(str)
    _validate_splits(frame, frozen, label_column)
    joined, columns = _features(frame, features, feature_columns)
    joined = joined.set_index("record_id")
    rows: list[pd.DataFrame] = []
    fit_records: list[dict] = []
    for split_id, part in frozen.groupby("split_id", sort=True):
        data = joined.loc[part.record_id].reset_index()
        data["role"] = part.role.to_numpy()
        data["input_source_available"] = _source_available(data)
        data["group_id"] = part.group_id.to_numpy()
        data["held_out"] = (
            part.held_out.astype(str).str.lower().isin(["true", "1"]).to_numpy()
            if "held_out" in part else data.role.eq("test")
        )
        data["pdb_policy"] = part.pdb_policy.to_numpy() if "pdb_policy" in part else "unspecified"
        data["split_id"] = str(split_id)
        data["target_label"] = data[label_column]
        data["eligibility_reason"] = part.get("eligibility_reason", pd.Series("unspecified", index=part.index)).to_numpy()
        train = data.role.eq("train") & data.feature_available
        test = data.role.eq("test")
        train_labels = set(data.loc[train, label_column])
        data["prediction"] = ""
        data["prediction_status"] = data.role
        data.loc[test, "prediction_status"] = "feature_failure"
        can_predict = test & data.feature_available
        data.loc[can_predict, "prediction_status"] = "insufficient_training_classes"
        pipeline = None
        if len(train_labels) >= 2:
            pipeline = _pipeline(model, seed, n_jobs, n_estimators)
            weights = training_weights(
                data.loc[train, label_column], method=weighting,
                groups=data.loc[train, weight_group_column], beta=beta,
            )
            pipeline.fit(data.loc[train, columns], data.loc[train, label_column], model__sample_weight=weights)
            if can_predict.any():
                data.loc[can_predict, "prediction"] = pipeline.predict(data.loc[can_predict, columns])
                data.loc[can_predict, "prediction_status"] = "ok"
        # This is an evaluation flag, never an input to the fitted predictor.
        data["class_evaluation_status"] = np.where(
            data[label_column].isin(train_labels), "training_class_available", "training_class_unavailable"
        )
        fit_record = {
            "split_id": str(split_id), "n_train_planned": int(data.role.eq("train").sum()),
            "n_train_available": int(train.sum()), "training_classes": sorted(train_labels),
            "weighting": weighting, "weight_group_column": weight_group_column, "model": model,
        }
        if pipeline is not None:
            fit_record["imputer_statistics"] = pipeline.named_steps["imputer"].statistics_.tolist()
        fit_records.append(fit_record)
        rows.append(data.loc[~data.role.eq("train"), list(MANIFEST_COLUMNS) + [
            "group_id", "split_id", "role", "held_out", "pdb_policy", "target_label", "eligibility_reason", "input_source_available", "feature_available", "prediction", "prediction_status", "class_evaluation_status"
        ]])
    predictions = pd.concat(rows, ignore_index=True)
    summary, per_class = summarize_predictions(predictions, labels=sorted(frame[label_column].unique()))
    summary.update({"model": model, "weighting": weighting, "weight_group_column": weight_group_column,
                    "beta": beta, "seed": seed, "feature_columns": columns, "fits": fit_records})
    return predictions, summary, per_class


def summarize_predictions(
    predictions: pd.DataFrame, *, labels: Sequence[str] | None = None
) -> tuple[dict, pd.DataFrame]:
    """Fixed-denominator effectiveness plus successful-prediction accuracy."""
    _require(predictions, ["record_id", "role", "target_label", "prediction_status", "group_id"], "predictions")
    if "prediction" not in predictions:
        raise ValueError("predictions lack the prediction column")
    test = predictions[predictions.role.eq("test")].copy()
    test["correct"] = test.prediction_status.eq("ok") & test.prediction.eq(test.target_label)
    labels = sorted(set(labels) if labels is not None else predictions.target_label.unique())
    rows = []
    for label in labels:
        true = test.target_label.eq(label)
        guessed = test.prediction_status.eq("ok") & test.prediction.eq(label)
        tp, support, predicted = int((true & guessed).sum()), int(true.sum()), int(guessed.sum())
        rows.append({
            "label": label, "test_support": support, "test_groups": int(test.loc[true, "group_id"].nunique()),
            "prediction_coverage": float(test.loc[true, "prediction_status"].eq("ok").mean()) if support else None,
            "precision": tp / predicted if predicted else (0.0 if support else None),
            "recall": tp / support if support else None,
            "f1": 2 * tp / (support + predicted) if support else None,
            "status": "estimable" if support else "no_held_out_support",
        })
    per_class = pd.DataFrame(rows, columns=["label", "test_support", "test_groups", "prediction_coverage", "precision", "recall", "f1", "status"])
    available = test.prediction_status.eq("ok")
    predicted_labels = labels + sorted(set(test.loc[available, "prediction"]) - set(labels)) + ["<no_prediction>"]
    confusion = pd.crosstab(test.target_label, test.prediction.where(available, "<no_prediction>"))
    confusion = confusion.reindex(index=labels, columns=predicted_labels, fill_value=0)
    summary = {
        "n_test": len(test), "n_predictions": int(available.sum()),
        "n_planned_held_out": int(predictions.held_out.astype(str).str.lower().isin(["true", "1"]).sum()) if "held_out" in predictions else len(test),
        "n_estimable_classes": int(per_class.test_support.gt(0).sum()),
        "macro_recall": float(per_class.recall.mean()) if len(test) else None,
        "macro_f1": float(per_class.f1.mean()) if len(test) else None,
        "effectiveness_accuracy": float(test.correct.mean()) if len(test) else None,
        "prediction_coverage": float(available.mean()) if len(test) else None,
        "accuracy_on_predictions": float(test.loc[available, "correct"].mean()) if available.any() else None,
        "excluded_records": int(predictions.loc[predictions.role.eq("unavailable"), "record_id"].nunique()),
        "input_source_unavailable_records": int(predictions.loc[predictions.eligibility_reason.eq("input_source_unavailable"), "record_id"].nunique()) if "eligibility_reason" in predictions else None,
        "denominator": "all frozen test rows; missing/failed predictions count as incorrect",
        "confusion_matrix": {"true_labels": labels, "predicted_labels": predicted_labels, "counts": confusion.to_numpy(dtype=int).tolist()},
    }
    if len(test) and set((A, T, H, S)).issubset(test.columns):
        summary.update(hierarchical_accuracy(test))
        summary["hierarchical_metric_scope"] = "all frozen test topologies; R_rare is a rare-topology estimand only for a rare-topology challenge"
    return summary, per_class


def hierarchical_accuracy(table: pd.DataFrame, value_column: str = "correct") -> dict:
    """Equalize S35 within H, H within T, then optionally T within A."""
    _require(table, [A, T, H, S, value_column], "hierarchical metric")
    if table.empty:
        return {"r_rare": None, "r_a_t_h": None, "n_topologies": 0}
    per_s = table.groupby([A, T, H, S], sort=True)[value_column].mean()
    per_h = per_s.groupby(level=[0, 1, 2]).mean()
    per_t = per_h.groupby(level=[0, 1]).mean()
    per_a = per_t.groupby(level=0).mean()
    return {"r_rare": float(per_t.mean()), "r_a_t_h": float(per_a.mean()), "n_topologies": len(per_t)}


def paired_topology_bootstrap(
    left: pd.DataFrame, right: pd.DataFrame, *, n_resamples: int = 10000, seed: int = 13,
    grouping_manifest: pd.DataFrame | None = None,
) -> dict:
    """Paired R_rare difference for fixed predictions, resampling T-PDB blocks.

All predictions must describe the same single frozen test challenge. Connected
T-PDB components protect dependence across topologies. One component cannot
identify sampling uncertainty. Intervals remain descriptive for few blocks and
do not include model refitting or claim architecture-stratified precision.
"""
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")
    parts = []
    for table in (left, right):
        part = table[table.role.eq("test")].copy()
        if part.split_id.nunique() != 1 or part.record_id.duplicated().any():
            raise ValueError("bootstrap requires one fixed challenge with unique test records")
        part["correct"] = (part.prediction_status.eq("ok") & part.prediction.eq(part.target_label)).astype(float)
        parts.append(part.set_index("record_id").sort_index())
    lft, rgt = parts
    if not lft.index.equals(rgt.index) or not lft[list(MANIFEST_COLUMNS[1:]) + ["target_label", "split_id"]].equals(rgt[list(MANIFEST_COLUMNS[1:]) + ["target_label", "split_id"]]):
        raise ValueError("paired methods must share identical test records, labels and split")
    data = lft.reset_index()
    data["difference"] = lft.correct.to_numpy() - rgt.correct.to_numpy()
    data["block"] = _mapped_groups(_manifest(data), T, grouping_manifest)
    per_t = data.groupby([A, T, H, S]).difference.mean().groupby(level=[0, 1, 2]).mean().groupby(level=[0, 1]).mean()
    t_blocks = data.groupby(T).block.first()
    blocks = sorted(data.block.unique())
    estimate = float(per_t.mean()) if len(per_t) else None
    result = {"difference": estimate, "n_blocks": len(blocks), "n_topologies": len(per_t),
              "estimand": "paired R_rare, left minus right, fixed predictors", "n_resamples": n_resamples}
    if len(blocks) < 2:
        return {**result, "status": "insufficient_independent_blocks", "ci95": None}
    block_sums = np.array([per_t.loc[per_t.index.get_level_values(1).map(t_blocks) == block].sum() for block in blocks])
    block_counts = np.array([int((per_t.index.get_level_values(1).map(t_blocks) == block).sum()) for block in blocks])
    rng = np.random.default_rng(seed)
    draws = rng.integers(len(blocks), size=(n_resamples, len(blocks)))
    values = block_sums[draws].sum(axis=1) / block_counts[draws].sum(axis=1)
    return {**result, "status": "descriptive_fixed_predictor_interval", "ci95": np.quantile(values, [0.025, 0.975]).tolist()}


def _read(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def load_feature_matrix(path: str | Path) -> tuple[pd.DataFrame, list[str] | None]:
    """Read a CSV or an aligned ESM NPZ without unpickling stored objects.

    The NPZ availability mask is authoritative: aligned placeholder vectors
    remain failed features even when their numeric values are finite zeros.
    CSVs retain the ordinary raw-geometry default feature selection.
    """
    if Path(path).suffix.lower() != ".npz":
        return _read(path), None
    with np.load(path, allow_pickle=False) as archive:
        required = {"record_id", "embeddings", "esmc_available"}
        if required - set(archive.files):
            raise ValueError(f"Embedding NPZ lacks keys: {sorted(required - set(archive.files))}")
        identifiers = archive["record_id"]
        vectors = archive["embeddings"]
        mask = archive["esmc_available"]
    if identifiers.ndim != 1 or identifiers.dtype.kind not in "US":
        raise ValueError("Embedding record_id must be a one-dimensional string array")
    identifiers = identifiers.astype(str)
    if any(not value.strip() for value in identifiers) or len(set(identifiers)) != len(identifiers):
        raise ValueError("Embedding record_id must be nonempty and unique")
    if vectors.ndim != 2 or vectors.shape[0] != len(identifiers) or vectors.shape[1] == 0:
        raise ValueError("Embedding shape must be (number of record IDs, positive feature dimension)")
    if vectors.dtype.kind not in "fiu":
        raise ValueError("Embeddings must be a real numeric matrix")
    if mask.ndim != 1 or len(mask) != len(identifiers) or mask.dtype.kind not in "bfiu" or not np.isin(mask, [0, 1]).all():
        raise ValueError("esmc_available must be an aligned one-dimensional binary mask")
    available = mask.astype(bool)
    if not np.isfinite(vectors[available]).all():
        raise ValueError("Available embedding rows must contain only finite values")
    columns = [f"embedding_{index:04d}" for index in range(vectors.shape[1])]
    table = pd.DataFrame(vectors, columns=columns)
    table.insert(0, "record_id", identifiers)
    table["feature_status"] = np.where(available, "ok", "unavailable")
    return table, columns


def _input_identity(path: str | Path) -> dict:
    source = Path(path).resolve()
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(source), "bytes": source.stat().st_size, "sha256": digest.hexdigest()}


def assert_comparable_protocols(left: dict, right: dict, *, same_features: bool = False) -> None:
    """Verify shared frozen inputs before pairing results across methods.

Feature matrices may differ when comparing representations; set same_features
for classifier/weighting ablations on the same representation. Paths may differ
across machines, so content identities, not absolute paths, are compared.
"""
    if left.get("command") != "run" or right.get("command") != "run":
        raise ValueError("comparability requires two completed run protocols")
    required = ["manifest", "splits"] + (["features"] if same_features else [])
    for name in required:
        identity_left = left.get("inputs", {}).get(name, {}).get("sha256")
        identity_right = right.get("inputs", {}).get(name, {}).get("sha256")
        if not identity_left or identity_left != identity_right:
            raise ValueError(f"runs use different or unrecorded {name} contents")
    if left.get("settings", {}).get("label_column") != right.get("settings", {}).get("label_column"):
        raise ValueError("runs use different target label columns")
    if same_features and left.get("feature_columns") != right.get("feature_columns"):
        raise ValueError("runs use different feature allowlists")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("split", "challenge", "run"):
        command = commands.add_parser(name)
        command.add_argument("--manifest", required=True)
        command.add_argument("--out-dir", required=True)
        command.add_argument("--seed", type=int, default=13)
        if name != "run":
            command.add_argument("--grouping-manifest", help="Full catalog used to retain PDB bridges before representative selection")
        if name != "challenge":
            command.add_argument("--label-column", choices=(A, T), default=A)
    split = commands.choices["split"]
    split.add_argument("--group-column", choices=(S, H, T), default=H)
    split.add_argument("--n-splits", type=int, default=5)
    split.add_argument("--pdb-policy", choices=("connected", "purge"), default="connected")
    split.add_argument("--development-from", help="Frozen challenge splits.csv; only its training side enters development folds")
    commands.choices["challenge"].add_argument("--max-homologies", type=int, default=4)
    run = commands.choices["run"]
    run.add_argument("--features", required=True, help="Feature CSV or aligned ESM NPZ with availability mask")
    run.add_argument("--splits", required=True)
    run.add_argument("--feature-columns", help="Comma-separated numeric allowlist; default: raw Betlas geometry for CSV, all embedding dimensions for NPZ")
    run.add_argument("--model", choices=("logistic", "random_forest"), default="logistic")
    run.add_argument("--weighting", choices=WEIGHTINGS, default="none")
    run.add_argument("--weight-group-column", choices=(H, S), default=H,
                     help="Training-only groups for group-balanced weighting; default: homology (H)")
    run.add_argument("--beta", type=float, default=0.999)
    run.add_argument("--n-jobs", type=int, default=1)
    run.add_argument("--n-estimators", type=int, default=200)
    args = parser.parse_args(argv)
    input_keys = ["manifest"] + (["features", "splits"] if args.command == "run" else [])
    if getattr(args, "grouping_manifest", None):
        input_keys.append("grouping_manifest")
    if getattr(args, "development_from", None):
        input_keys.append("development_from")
    protocol = {
        "schema": "betlas.cath-classification-protocol.v1", "command": args.command,
        "inputs": {key: _input_identity(getattr(args, key)) for key in input_keys},
        "settings": {key: value for key, value in vars(args).items() if key not in input_keys},
        "software": {"python": platform.python_version(), "scikit_learn": sklearn.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__},
        "identity_consumer": "assert_comparable_protocols validates paired model/weighting comparisons",
    }
    manifest = _read(args.manifest)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.command in ("split", "challenge"):
        universe = _read(args.grouping_manifest) if args.grouping_manifest else None
        if args.command == "split":
            options = dict(label_column=args.label_column, group_column=args.group_column,
                           n_splits=args.n_splits, seed=args.seed, grouping_manifest=universe,
                           pdb_policy=args.pdb_policy)
            if args.development_from:
                roles, support = make_development_folds(manifest, _read(args.development_from), **options)
            else:
                roles, support = make_grouped_folds(manifest, **options)
        else:
            roles, support = make_rare_topology_challenge(manifest, max_homologies=args.max_homologies,
                                                        seed=args.seed, grouping_manifest=universe)
        roles.to_csv(out / "splits.csv", index=False)
        support.to_csv(out / "support.csv", index=False)
        protocol["settings"]["group_column"] = roles.group_column.iloc[0]
        protocol["settings"]["label_column"] = roles.label_column.iloc[0]
        protocol["grouping_scope"] = "full grouping manifest" if universe is not None else "input manifest only"
        protocol["selection_scope"] = "frozen challenge training side" if getattr(args, "development_from", None) else "whole input cohort; do not use for tuning a separately frozen challenge"
        grouped_records = roles.drop_duplicates("record_id")
        protocol["grouping_summary"] = {
            "n_selected_records": len(grouped_records),
            "n_selected_hierarchy_keys": int(grouped_records[roles.group_column.iloc[0]].nunique()),
            "n_groups": int(grouped_records.group_id.nunique()),
            "largest_selected_component_fraction": float(grouped_records.groupby("group_id").size().max() / len(grouped_records)),
        }
        protocol["split_identity"] = _input_identity(out / "splits.csv")
    else:
        features, default_columns = load_feature_matrix(args.features)
        predictions, summary, per_class = run_classification(
            manifest, features, _read(args.splits), label_column=args.label_column,
            feature_columns=args.feature_columns.split(",") if args.feature_columns else default_columns,
            model=args.model, weighting=args.weighting, beta=args.beta, seed=args.seed,
            weight_group_column=args.weight_group_column,
            n_jobs=args.n_jobs, n_estimators=args.n_estimators,
        )
        predictions.to_csv(out / "predictions.csv", index=False)
        per_class.to_csv(out / "per_class.csv", index=False)
        (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        protocol["feature_columns"] = summary["feature_columns"]
        protocol["group_columns"] = sorted(_read(args.splits).group_column.unique())
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
