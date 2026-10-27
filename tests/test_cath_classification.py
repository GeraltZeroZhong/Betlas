"""Synthetic hierarchy and failure cases for frozen CATH evaluation."""

import json

import numpy as np
import pandas as pd
import pytest

from betlas.benchmarks import classification
from betlas.benchmarks.classification import (
    A,
    H,
    S,
    T,
    assert_comparable_protocols,
    connected_groups,
    hierarchical_accuracy,
    load_feature_matrix,
    main,
    make_development_folds,
    make_grouped_folds,
    make_rare_topology_challenge,
    paired_topology_bootstrap,
    run_classification,
    training_weights,
)
from betlas.ml.benchmark import _make_models, load_benchmark_config


def manifest(groups_per_class=(4, 4), domains=2):
    rows = []
    for architecture, n_groups in enumerate(groups_per_class, 1):
        for homology in range(n_groups):
            for copy in range(domains):
                record = f"a{architecture}h{homology}d{copy}"
                rows.append({
                    "record_id": record, "pdb_id": record,
                    A: f"2.{architecture}", T: f"2.{architecture}.{homology}",
                    H: f"2.{architecture}.{homology}.1", S: f"2.{architecture}.{homology}.1.{copy}",
                })
    return pd.DataFrame(rows)


def features(frame):
    return pd.DataFrame({
        "record_id": frame.record_id,
        "betlas_size": [float(value.split(".")[1]) for value in frame[A]],
        "betlas_shape": np.arange(len(frame), dtype=float),
        "betlas_parse_ok": True,
    })


def test_connected_components_keep_unselected_bridges():
    full = manifest((2,), domains=1)
    full.loc[1, "pdb_id"] = "other"
    bridge = full.iloc[1].copy()
    bridge["record_id"] = "bridge"
    bridge["pdb_id"] = full.iloc[0].pdb_id
    full = pd.concat([full, bridge.to_frame().T], ignore_index=True)
    assert connected_groups(full, H).nunique() == 1
    selected = full.iloc[:2]
    _, support = make_grouped_folds(selected, grouping_manifest=full)
    assert support.iloc[0].status == "single_independent_group"
    assert connected_groups(selected, H).nunique() == 2


def test_partial_fold_rare_class_and_singleton_remain_visible():
    frame = manifest((2, 6, 1), domains=1)
    splits, support = make_grouped_folds(frame, n_splits=5)
    rare = splits[(splits.target_label == "2.1") & (splits.role == "test")]
    assert len(rare) == 2
    assert rare.split_id.nunique() == 2
    single = splits[splits.target_label == "2.3"]
    assert set(single.role) == {"unavailable"}
    assert support.set_index("label").loc["2.3", "status"] == "single_independent_group"
    for _, fold in splits.groupby("split_id"):
        train, test = fold[fold.role == "train"], fold[fold.role == "test"]
        assert set(train.group_id).isdisjoint(test.group_id)
        assert set(test.target_label).issubset(train.target_label)


def test_purge_uses_hidden_full_catalog_pdb_members_of_heldout_h():
    selected = manifest((3, 3), domains=1)
    initial, _ = make_grouped_folds(selected, n_splits=3, pdb_policy="purge")
    held_out = initial[(initial.split_id == "fold-0") & (initial.role == "test") & (initial[A] == "2.1")].iloc[0]
    training = initial[(initial.split_id == "fold-0") & (initial.role == "train") & (initial[A] == "2.1")].iloc[0]
    bridge = selected[selected.record_id.eq(held_out.record_id)].iloc[0].copy()
    bridge["record_id"] = "hidden-full-catalog-domain"
    bridge["pdb_id"] = training.pdb_id
    full = pd.concat([selected, bridge.to_frame().T], ignore_index=True)
    roles, support = make_grouped_folds(selected, n_splits=3, grouping_manifest=full, pdb_policy="purge")
    fold = roles[roles.split_id.eq("fold-0")].set_index("record_id")
    assert fold.loc[training.record_id, "role"] == "purged"
    assert fold.loc[training.record_id, "eligibility_reason"] == "shared_full_catalog_test_pdb"
    assert fold.loc[held_out.record_id, "role"] == "test"
    assert roles.group_id.eq(roles[H]).all()
    assert set(support.pdb_policy) == {"purge"}
    # The unselected bridge must never become a training/test observation.
    assert "hidden-full-catalog-domain" not in set(roles.record_id)
    run_classification(selected, features(selected), roles, feature_columns=["betlas_size", "betlas_shape"])


def test_purge_class_loss_is_unavailable_without_reassigning_test_members():
    frame = manifest((2, 3), domains=1)
    frame.loc[frame[A].eq("2.1"), "pdb_id"] = "shared"
    roles, support = make_grouped_folds(frame, n_splits=3, pdb_policy="purge")
    held_out = roles[roles.held_out & roles[A].eq("2.1")]
    assert len(held_out) == 2
    assert set(held_out.role) == {"unavailable"}
    assert set(held_out.eligibility_reason) == {"class_absent_after_pdb_purge"}
    row = support.set_index("label").loc["2.1"]
    assert row.status == "not_evaluable_after_pdb_purge"
    assert row.n_evaluable_test_records == 0
    assert row.n_groups == 2
    prediction, summary, _ = run_classification(frame, features(frame), roles, feature_columns=["betlas_size", "betlas_shape"])
    assert summary["n_planned_held_out"] == len(frame)
    assert summary["n_test"] == 3
    assert set(prediction.pdb_policy) == {"purge"}


def test_splits_are_order_invariant_and_freeze_native_topology_labels():
    frame = manifest((3, 3), domains=2)
    # Two distinct H keys per native T permit native T prediction.
    frame[H] = frame[H] + "." + frame.record_id.str[-1]
    original, _ = make_grouped_folds(frame, label_column=T, n_splits=3)
    shuffled, _ = make_grouped_folds(frame.sample(frac=1, random_state=8), label_column=T, n_splits=3)
    keys = ["split_id", "record_id"]
    pd.testing.assert_frame_equal(original.sort_values(keys).reset_index(drop=True), shuffled.sort_values(keys).reset_index(drop=True))
    assert original.target_label.nunique() == 6
    assert (original.role == "test").sum() == len(frame)


@pytest.mark.parametrize("column", ["pdb_id", H, S])
def test_missing_groups_do_not_fall_back(column):
    frame = manifest()
    frame.loc[0, column] = ""
    with pytest.raises(ValueError, match="missing values"):
        make_grouped_folds(frame)


def test_unscoped_cluster_keys_are_rejected():
    frame = manifest()
    frame[S] = "1"
    with pytest.raises(ValueError, match="complete key"):
        make_grouped_folds(frame)


@pytest.mark.parametrize("status_column", ["representative_status", "coordinate_status"])
@pytest.mark.parametrize("pdb_policy", ["connected", "purge"])
def test_declared_input_absence_is_not_an_algorithm_failure(status_column, pdb_policy):
    frame = manifest((4, 4), domains=1)
    frame[status_column] = "available"
    frame.loc[0, status_column] = "unavailable"
    roles, support = make_grouped_folds(frame, pdb_policy=pdb_policy, n_splits=3)
    unavailable = roles[roles.record_id.eq(frame.record_id.iloc[0])]
    assert set(unavailable.role) == {"unavailable"}
    assert set(unavailable.eligibility_reason) == {"input_source_unavailable"}
    row = support.set_index("label").loc["2.1"]
    assert row.n_records == 4
    assert row.n_groups_before_input_qc == 4
    assert row.n_groups == 3
    assert row.n_input_unavailable == 1
    prediction, summary, _ = run_classification(frame, features(frame), roles, feature_columns=["betlas_size", "betlas_shape"])
    assert summary["n_test"] == len(frame) - 1
    assert summary["input_source_unavailable_records"] == 1
    assert not prediction.prediction_status.eq("feature_failure").any()
    assert prediction.loc[prediction.record_id.eq(frame.record_id.iloc[0]), "prediction"].eq("").all()


def test_input_qc_lost_class_support_is_declared_before_partition():
    frame = manifest((2, 3), domains=1)
    frame["representative_status"] = "available"
    frame.loc[0, "representative_status"] = "unavailable"
    roles, support = make_grouped_folds(frame, pdb_policy="purge")
    assert support.set_index("label").loc["2.1", "status"] == "insufficient_groups_after_input_qc"
    assert set(roles.loc[roles[A].eq("2.1"), "role"]) == {"unavailable"}
    assert "input_source_unavailable" in set(roles.eligibility_reason)


def test_all_source_unavailable_preserves_audit_ledgers():
    frame = manifest((3, 3), domains=1)
    frame["coordinate_status"] = "unavailable"
    roles, support = make_grouped_folds(frame, pdb_policy="purge")
    assert len(roles) == len(frame)
    assert set(roles.role) == {"unavailable"}
    assert set(support.status) == {"input_source_unavailable"}
    challenge, _ = make_rare_topology_challenge(frame)
    assert set(challenge.eligibility_reason) == {"input_source_unavailable"}
    development, _ = make_development_folds(frame, challenge, pdb_policy="purge")
    assert len(development) == len(frame)
    prediction, summary, _ = run_classification(frame, features(frame), roles, feature_columns=["betlas_size", "betlas_shape"])
    assert summary["n_test"] == 0
    assert summary["n_predictions"] == 0
    assert summary["input_source_unavailable_records"] == len(frame)
    assert set(prediction.prediction_status) == {"unavailable"}


def test_challenge_never_trains_on_declared_unavailable_sources():
    frame = manifest((4, 4), domains=1)
    initial, _ = make_rare_topology_challenge(frame)
    training_id = initial.loc[initial.role.eq("train"), "record_id"].iloc[0]
    frame["representative_status"] = "available"
    frame.loc[frame.record_id.eq(training_id), "representative_status"] = "unavailable"
    challenge, _ = make_rare_topology_challenge(frame)
    source = challenge.set_index("record_id").loc[training_id]
    assert source.role == "unavailable"
    assert source.eligibility_reason == "input_source_unavailable"
    development, _ = make_development_folds(frame, challenge, pdb_policy="purge")
    assert set(development.loc[development.record_id.eq(training_id), "eligibility_reason"]) == {"input_source_unavailable"}


def test_training_weights_preserve_mean_and_equalize_requested_units():
    labels = ["a", "a", "a", "b"]
    for method in ("none", "inverse", "sqrt", "effective", "group-balanced"):
        weights = training_weights(labels, method=method, groups=["x", "x", "y", "z"])
        assert np.mean(weights) == pytest.approx(1)
        assert np.isfinite(weights).all()
    weights = training_weights(labels, method="inverse")
    assert weights[:3].sum() == pytest.approx(weights[3])
    weights = training_weights(labels, method="group-balanced", groups=["x", "x", "y", "z"])
    assert weights[:2].sum() == pytest.approx(weights[2])
    assert weights[:3].sum() == pytest.approx(weights[3])
    np.testing.assert_allclose(training_weights(labels, method="effective", beta=0), 1)
    with pytest.raises(ValueError, match="beta"):
        training_weights(labels, method="effective", beta=1)


@pytest.mark.parametrize("weight_group_column", [None, S])
def test_group_balancing_uses_requested_training_groups_only(monkeypatch, weight_group_column):
    frame = manifest((4, 4), domains=3)
    # Each S35 has one representative, but H groups have unequal numbers of S35s.
    frame = frame[[int(row.record_id[-1]) <= int(row.record_id.split("h")[1][0]) % 3
                   for row in frame.itertuples()]].reset_index(drop=True)
    splits, _ = make_grouped_folds(frame, n_splits=2)
    captured = []

    def capture_weights(labels, **kwargs):
        result = training_weights(labels, **kwargs)
        captured.append((list(labels), list(kwargs["groups"]), result))
        return result

    monkeypatch.setattr(classification, "training_weights", capture_weights)
    options = {} if weight_group_column is None else {"weight_group_column": weight_group_column}
    _, summary, _ = run_classification(frame, features(frame), splits, weighting="group-balanced", **options, feature_columns=["betlas_size", "betlas_shape"])
    chosen_group = H if weight_group_column is None else weight_group_column
    assert summary["weight_group_column"] == chosen_group
    differs_from_inverse = []
    for ((_, fold), (labels, groups, weights), fit) in zip(
        splits.groupby("split_id", sort=True), captured, summary["fits"], strict=True
    ):
        training = fold[fold.role.eq("train")]
        held_out = fold[fold.role.eq("test")]
        assert labels == training[A].tolist()
        assert groups == training[chosen_group].tolist()
        assert set(groups).isdisjoint(held_out[chosen_group])
        assert fit["weight_group_column"] == chosen_group
        inverse = training_weights(labels, method="inverse")
        differs_from_inverse.append(not np.allclose(weights, inverse))
        mass = pd.DataFrame({"label": labels, "group": groups, "weight": weights})
        for _, within_class in mass.groupby("label"):
            group_mass = within_class.groupby("group").weight.sum()
            np.testing.assert_allclose(group_mass, group_mass.iloc[0])
    # S35 balancing degenerates to inverse frequency on this cohort; H does not.
    assert any(differs_from_inverse) if chosen_group == H else not any(differs_from_inverse)


@pytest.mark.parametrize("model", ["logistic", "random_forest"])
def test_fixed_denominator_and_unavailable_class_with_missing_feature_rows(model):
    frame = manifest((4, 4, 1))
    splits, _ = make_grouped_folds(frame, n_splits=3)
    values = features(frame).iloc[1:].copy()
    values.loc[values.index[0], "betlas_parse_ok"] = False
    prediction, summary, classes = run_classification(frame, values, splits, model=model, n_estimators=5, feature_columns=["betlas_size", "betlas_shape"])
    assert summary["n_test"] == len(frame) - 2
    assert summary["n_predictions"] == len(frame) - 4
    assert summary["excluded_records"] == 2
    assert len(prediction[prediction.prediction_status == "feature_failure"]) == 2
    assert classes.set_index("label").loc["2.3", "status"] == "no_held_out_support"
    assert "betlas_parse_ok" not in summary["feature_columns"]
    assert summary["macro_recall"] <= 1
    confusion = np.asarray(summary["confusion_matrix"]["counts"])
    assert confusion.sum() == summary["n_test"]
    assert confusion[:, -1].sum() == 2


def test_root_extraction_statuses_remain_in_frozen_denominator():
    frame = manifest()
    splits, _ = make_grouped_folds(frame, n_splits=3)
    values = features(frame)
    values["feature_status"] = "ok"
    values.loc[0, "feature_status"] = "no_beta_segments"
    values.loc[1, "feature_status"] = "mapping_failed"
    prediction, summary, _ = run_classification(frame, values, splits, feature_columns=["betlas_size", "betlas_shape"])
    assert summary["n_test"] == len(frame)
    assert summary["n_predictions"] == len(frame) - 2
    assert set(prediction.loc[prediction.record_id.isin(frame.record_id.iloc[:2]), "prediction_status"]) == {"feature_failure"}


def test_preprocessing_statistics_only_use_training_rows():
    frame = manifest(domains=1)
    splits, _ = make_grouped_folds(frame, n_splits=2)
    values = features(frame)
    first_test = splits[(splits.split_id == "fold-0") & (splits.role == "test")].record_id
    values.loc[values.record_id.isin(first_test), "betlas_shape"] = 1e12
    _, summary, _ = run_classification(frame, values, splits, feature_columns=["betlas_size", "betlas_shape"])
    fit = summary["fits"][0]
    expected = values.loc[~values.record_id.isin(first_test), "betlas_shape"].median()
    shape_index = summary["feature_columns"].index("betlas_shape")
    assert fit["imputer_statistics"][shape_index] == expected


def test_explicit_metadata_feature_and_tampered_split_are_rejected():
    frame = manifest()
    splits, _ = make_grouped_folds(frame)
    values = features(frame)
    values[A] = frame[A]
    with pytest.raises(ValueError, match="metadata"):
        run_classification(frame, values, splits, feature_columns=[A])
    broken = splits.copy()
    broken.loc[0, "target_label"] = "wrong"
    with pytest.raises(ValueError, match="labels differ"):
        run_classification(frame, values, broken, feature_columns=["betlas_size", "betlas_shape"])
    duplicate = pd.concat([values, values.iloc[:1]])
    with pytest.raises(ValueError, match="unique"):
        run_classification(frame, duplicate, splits, feature_columns=["betlas_size", "betlas_shape"])


def test_feature_loss_of_a_training_class_does_not_use_test_labels_for_predictions():
    frame = manifest((2, 2, 2), domains=1)
    splits, _ = make_grouped_folds(frame, n_splits=2)
    one = splits[splits.split_id == "fold-0"].copy()
    values = features(frame)
    removed = set(one[(one.role == "train") & (one.target_label == "2.3")].record_id)
    values = values[~values.record_id.isin(removed)]
    prediction, _, _ = run_classification(frame, values, one, feature_columns=["betlas_size", "betlas_shape"])
    absent = prediction[(prediction.role == "test") & (prediction.target_label == "2.3")]
    assert len(absent) > 0
    assert set(absent.class_evaluation_status) == {"training_class_unavailable"}
    assert set(absent.prediction_status) == {"ok"}
    assert absent.prediction.ne("").all()


def test_rare_topology_challenge_is_frozen_and_pdb_purge_keeps_a_support():
    frame = manifest((6, 6), domains=1)
    first, catalog = make_rare_topology_challenge(frame)
    second, _ = make_rare_topology_challenge(frame.sample(frac=1, random_state=3))
    assert set(first[first.role == "test"][T]) == set(second[second.role == "test"][T])
    assert catalog.selected_for_challenge.sum() > 0
    train, test = first[first.role == "train"], first[first.role == "test"]
    assert set(train[T]).isdisjoint(test[T])
    assert set(test[A]).issubset(train[A])
    assert set(train.pdb_id).isdisjoint(test.pdb_id)
    predictions, summary, _ = run_classification(frame, features(frame), first, feature_columns=["betlas_size", "betlas_shape"])
    assert summary["n_test"] == len(test)
    assert predictions.split_id.nunique() == 1


def test_challenge_purge_cannot_silently_reintroduce_unsupported_architecture():
    frame = manifest((4, 4), domains=1)
    frame["pdb_id"] = frame[A]
    roles, _ = make_rare_topology_challenge(frame)
    assert not roles.role.eq("test").any()
    assert "architecture_absent_after_purge" in set(roles.eligibility_reason)
    assert roles.role.eq("purged").any()


@pytest.mark.parametrize("pdb_policy", ["connected", "purge"])
def test_frozen_challenge_cannot_enter_development_cross_validation(pdb_policy):
    frame = manifest((6, 6), domains=2)
    challenge, _ = make_rare_topology_challenge(frame)
    development, support = make_development_folds(frame, challenge, n_splits=3, pdb_policy=pdb_policy)
    reserved = set(challenge.loc[challenge.role.ne("train"), "record_id"])
    assert reserved
    assert set(development.loc[development.role.isin(["train", "test"]), "record_id"]).isdisjoint(reserved)
    assert set(development.loc[development.record_id.isin(reserved), "eligibility_reason"]) == {"reserved_frozen_challenge"}
    for _, fold in development.groupby("split_id"):
        assert set(fold.record_id) == set(frame.record_id)
    assert support.n_reserved_records.sum() == len(reserved)
    _, summary, _ = run_classification(frame, features(frame), development, feature_columns=["betlas_size", "betlas_shape"])
    assert summary["n_test"] == len(frame) - len(reserved)


def test_hierarchical_metric_equalizes_s_h_t_instead_of_domains():
    rows = []
    # T1: H1 has S means 1 and 0; H2 has mean 1, giving T1=0.75.
    for topology, homology, cluster, score, copies in [(1, 1, 1, 1, 20), (1, 1, 2, 0, 1), (1, 2, 1, 1, 1), (2, 1, 1, 0, 1)]:
        for _ in range(copies):
            rows.append({A: "a", T: str(topology), H: str(homology), S: str(cluster), "correct": score})
    result = hierarchical_accuracy(pd.DataFrame(rows))
    assert result["r_rare"] == pytest.approx(0.375)
    assert result["r_a_t_h"] == pytest.approx(0.375)


def test_paired_bootstrap_resamples_connected_topologies_and_checks_pairing():
    frame = manifest((4, 4), domains=1)
    roles, _ = make_rare_topology_challenge(frame)
    prediction, _, _ = run_classification(frame, features(frame), roles, feature_columns=["betlas_size", "betlas_shape"])
    left = prediction.copy()
    left["prediction"] = left.target_label
    left["prediction_status"] = "ok"
    right = left.copy()
    right["prediction"] = "wrong"
    result = paired_topology_bootstrap(left, right, n_resamples=20)
    assert result["difference"] == pytest.approx(1)
    assert result["ci95"] == pytest.approx([1, 1])
    left["pdb_id"] = "same"
    right["pdb_id"] = "same"
    assert paired_topology_bootstrap(left, right, n_resamples=20)["ci95"] is None
    with pytest.raises(ValueError, match="identical"):
        paired_topology_bootstrap(left, right.iloc[1:], n_resamples=20)


@pytest.mark.parametrize("weight_group_column", [None, S])
def test_cli_split_then_run_uses_caller_features(tmp_path, weight_group_column):
    frame = manifest((3, 3), domains=1)
    frame.to_csv(tmp_path / "manifest.csv", index=False)
    values = features(frame).rename(columns={"betlas_size": "embedding_0"})
    values.to_csv(tmp_path / "features.csv", index=False)
    assert main(["split", "--manifest", str(tmp_path / "manifest.csv"), "--out-dir", str(tmp_path / "split")]) == 0
    weight_options = [] if weight_group_column is None else ["--weight-group-column", weight_group_column]
    assert main(["run", "--manifest", str(tmp_path / "manifest.csv"), "--features", str(tmp_path / "features.csv"),
                 "--splits", str(tmp_path / "split/splits.csv"), "--feature-columns", "embedding_0",
                 "--out-dir", str(tmp_path / "run"), "--weighting", "group-balanced", *weight_options]) == 0
    assert (tmp_path / "run/summary.json").is_file()
    protocol = json.loads((tmp_path / "run/protocol.json").read_text())
    assert protocol["feature_columns"] == ["embedding_0"]
    assert protocol["software"]["scikit_learn"]
    assert protocol["settings"]["weight_group_column"] == (H if weight_group_column is None else weight_group_column)
    assert_comparable_protocols(protocol, protocol, same_features=True)
    changed = json.loads(json.dumps(protocol))
    changed["inputs"]["splits"]["sha256"] = "different"
    with pytest.raises(ValueError, match="splits"):
        assert_comparable_protocols(protocol, changed)


@pytest.mark.parametrize("feature_override", [None, "embedding_0001"])
def test_cli_aligned_npz_preserves_failed_zero_rows(tmp_path, feature_override):
    frame = manifest((4, 4), domains=1)
    frame.to_csv(tmp_path / "manifest.csv", index=False)
    splits, _ = make_grouped_folds(frame, n_splits=2)
    splits.to_csv(tmp_path / "splits.csv", index=False)
    vectors = np.arange(len(frame) * 3, dtype=float).reshape(len(frame), 3)
    vectors[0] = 0
    mask = np.ones(len(frame), dtype=np.float32)
    mask[0] = 0
    feature_path = tmp_path / "aligned.npz"
    np.savez_compressed(feature_path, record_id=frame.record_id.to_numpy(dtype=str),
                        embeddings=vectors, esmc_available=mask)
    loaded, columns = load_feature_matrix(feature_path)
    assert columns == ["embedding_0000", "embedding_0001", "embedding_0002"]
    assert loaded.feature_status.tolist() == ["unavailable"] + ["ok"] * (len(frame) - 1)
    options = [] if feature_override is None else ["--feature-columns", feature_override]
    assert main(["run", "--manifest", str(tmp_path / "manifest.csv"), "--features", str(feature_path),
                 "--splits", str(tmp_path / "splits.csv"), "--out-dir", str(tmp_path / "run"), *options]) == 0
    predictions = pd.read_csv(tmp_path / "run/predictions.csv", keep_default_na=False)
    missing = predictions[predictions.record_id.eq(frame.record_id.iloc[0])]
    assert len(missing) == 1
    assert missing.prediction_status.tolist() == ["feature_failure"]
    assert missing.prediction.eq("").all()
    summary = json.loads((tmp_path / "run/summary.json").read_text())
    assert summary["n_test"] == len(frame)
    assert summary["n_predictions"] == len(frame) - 1
    protocol = json.loads((tmp_path / "run/protocol.json").read_text())
    assert protocol["feature_columns"] == (columns if feature_override is None else [feature_override])
    assert protocol["inputs"]["features"]["sha256"]


@pytest.mark.parametrize("change,reason", [
    ("missing_key", "lacks keys"), ("duplicate_ids", "unique"), ("object_ids", "Object arrays"),
    ("one_dimensional", "shape"), ("wrong_rows", "shape"), ("text_vectors", "numeric"),
    ("wrong_mask_shape", "binary mask"), ("nonbinary_mask", "binary mask"),
    ("nonfinite_available", "finite"),
])
def test_npz_loader_rejects_malformed_alignment(tmp_path, change, reason):
    payload = {"record_id": np.asarray(["first", "second"]), "embeddings": np.ones((2, 3)),
               "esmc_available": np.asarray([1, 0])}
    if change == "missing_key":
        del payload["esmc_available"]
    elif change == "duplicate_ids":
        payload["record_id"] = np.asarray(["same", "same"])
    elif change == "object_ids":
        payload["record_id"] = np.asarray(["first", "second"], dtype=object)
    elif change == "one_dimensional":
        payload["embeddings"] = np.ones(2)
    elif change == "wrong_rows":
        payload["embeddings"] = np.ones((3, 3))
    elif change == "text_vectors":
        payload["embeddings"] = np.full((2, 3), "text")
    elif change == "wrong_mask_shape":
        payload["esmc_available"] = np.ones((2, 1))
    elif change == "nonbinary_mask":
        payload["esmc_available"] = np.asarray([1, 0.5])
    else:
        payload["embeddings"][0, 0] = np.nan
    path = tmp_path / "invalid.npz"
    np.savez_compressed(path, **payload)
    with pytest.raises(ValueError, match=reason):
        load_feature_matrix(path)


@pytest.mark.parametrize("model,default", [("logistic_l2", "balanced"), ("random_forest", "balanced_subsample"), ("extra_trees", "balanced")])
def test_legacy_explicit_null_class_weight_is_respected(model, default):
    base = load_benchmark_config(overrides={"models": {"include": [model]}})
    assert _make_models(13, ("x", "y"), base)[model].named_steps["model"].class_weight == default
    override = load_benchmark_config(overrides={"models": {"include": [model]}, model: {"class_weight": None}})
    assert _make_models(13, ("x", "y"), override)[model].named_steps["model"].class_weight is None


def test_default_classifier_requires_complete_current_descriptor_contract():
    frame = manifest((3, 3), domains=1)
    roles, _ = make_grouped_folds(frame, n_splits=2)
    with pytest.raises(ValueError, match="absent columns"):
        run_classification(frame, features(frame), roles)
