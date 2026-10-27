"""Analytical endpoint and leakage checks; no research datasets or model weights."""

from __future__ import annotations

import json

import pytest

from betlas.benchmarks.specialists import (
    BinaryCase,
    Score,
    binary_metrics,
    calibrate_threshold,
    evaluate_frozen_threshold,
    import_domain_scores,
    import_pair_calls,
    main,
    pair_metrics,
)


def cases(split="development"):
    return [BinaryCase("positive", 1, split, "H1", "pdb1"),
            BinaryCase("negative", 0, split, "H2", "pdb2")]


def test_unavailable_negative_is_not_correct_rejection():
    result = binary_metrics(cases(), [Score("positive", .8, "ok")], threshold=.5)
    assert result["confusion_by_true_class"] == [[0, 0, 1], [0, 1, 0]]
    assert result["accuracy"] == .5
    assert result["balanced_accuracy"] == .5
    assert result["recall"] == [0, 1]
    assert result["ranking"]["average_precision"] is None
    assert result["coverage"] == .5


def test_empty_valid_predictions_and_all_unavailable_are_not_conflated():
    valid = binary_metrics(cases(), [Score("positive", -3, "ok"), Score("negative", -2, "ok")], threshold=0)
    unavailable = binary_metrics(cases(), [], threshold=0)
    assert valid["coverage"] == 1
    assert valid["balanced_accuracy"] == .5
    assert unavailable["coverage"] == 0
    assert unavailable["balanced_accuracy"] == 0


def test_threshold_operator_preserves_native_raphael_sign_boundary():
    rows = [Score("positive", 0, "ok"), Score("negative", -1, "ok")]
    assert binary_metrics(cases(), rows, threshold=0, operator="gt")["accuracy"] == .5
    assert binary_metrics(cases(), rows, threshold=0, operator="ge")["accuracy"] == 1


def test_calibration_is_development_only_and_checks_all_overlap_axes():
    rows = [Score("positive", .8, "ok"), Score("negative", .2, "ok")]
    calibration = calibrate_threshold(cases(), rows, method="synthetic")
    assert calibration["threshold"] == .8
    with pytest.raises(ValueError, match="development-only"):
        calibrate_threshold(cases("test"), rows, method="synthetic")
    for case in [BinaryCase("positive", 1, "test", "newH", "newP"),
                 BinaryCase("new", 1, "test", "H1", "newP"),
                 BinaryCase("new", 1, "test", "newH", "PDB1")]:
        with pytest.raises(ValueError, match="overlap"):
            evaluate_frozen_threshold([case], [], calibration, method="synthetic")
    test_cases = [BinaryCase("newP", 1, "test", "H3", "pdb3"),
                  BinaryCase("newN", 0, "test", "H4", "pdb4")]
    result = evaluate_frozen_threshold(test_cases, [Score("newP", .9, "ok"), Score("newN", .1, "ok")],
                                       calibration, method="synthetic")
    assert result["accuracy"] == 1


@pytest.mark.parametrize("rows", [
    [Score("positive", float("nan"), "ok")],
    [Score("positive", .9, "failed")],
    [Score("unknown", .9, "ok")],
    [Score("positive", .9, "ok"), Score("positive", .1, "ok")],
])
def test_invalid_score_artifacts_do_not_change_denominators(rows):
    with pytest.raises(ValueError):
        binary_metrics(cases(), rows, threshold=.5)








def pair(sample, i, j, target=1, orientation="parallel"):
    return dict(sample_id=sample, residue_i=i, residue_j=j, target=target, orientation=orientation)


def test_contact_pair_symmetry_orientation_and_unavailable_macro_denominator():
    reference = [pair("a", 1, 4), pair("a", 2, 5, 0), pair("b", 1, 4)]
    calls = [pair("a", 4, 1, orientation="antiparallel"), pair("a", 2, 5)]
    result = pair_metrics(reference, calls, {"a": "ok", "b": "no_msa"},
                          unit="residue_contact", input_regime="sequence_predicted_ss")
    assert result["coverage"] == .5
    assert result["macro_recall"] == .5
    assert result["macro_f1"] == pytest.approx(1 / 3)
    assert result["macro_orientation_recall"] == 0
    assert result["per_sample"]["a"]["precision"] == .5
    assert result["per_sample"]["b"]["fn"] == 1


def test_pair_universe_is_not_silently_extended():
    with pytest.raises(ValueError, match="outside"):
        pair_metrics([pair("a", 1, 4)], [pair("a", 2, 5)], {"a": "ok"},
                     unit="residue_contact", input_regime="sequence_predicted_ss")
    with pytest.raises(ValueError, match="Duplicate"):
        pair_metrics([pair("a", 1, 4), pair("a", 4, 1)], [], {},
                     unit="strand_pair", input_regime="structure_reference")
    with pytest.raises(ValueError, match="successful"):
        pair_metrics([pair("a", 1, 4)], [pair("a", 1, 4)], {},
                     unit="residue_contact", input_regime="sequence_predicted_ss")


def test_native_empty_success_and_isolated_bridge_roundtrip():
    native = [dict(sample_id="a", tool="BCov", normalized_index_base=1, status="empty", contacts=[]),
              dict(sample_id="b", tool="BCov", normalized_index_base=1, status="ok",
                   contacts=[dict(residue_i=1, residue_j=4, orientation="bridge", score=None)])]
    calls, statuses = import_pair_calls(native, unit="residue_contact")
    result = pair_metrics([pair("a", 1, 4), pair("b", 1, 4, orientation="bridge")], calls,
                          {s["sample_id"]: s["status"] for s in statuses},
                          unit="residue_contact", input_regime="sequence_predicted_ss")
    assert result["coverage"] == 1
    assert result["macro_recall"] == .5
    assert calls[0]["orientation"] == "bridge"
    assert result["per_sample"]["b"]["orientation_recall"] is None


def test_dense_probability_map_cannot_be_imported_as_empty_hard_calls():
    with pytest.raises(ValueError, match="probability map"):
        import_pair_calls([dict(sample_id="a", tool="BetaPro", normalized_index_base=1,
                                status="ok", beta_residue_positions=[1, 4], contacts=[])],
                          unit="residue_contact")


def test_domain_adapter_bridge_keeps_score_meaning_and_unavailable_status():
    rows = import_domain_scores([dict(sample_id="a", method="SOLeNNoID", status="ok", beta_fraction=.5),
                                 dict(sample_id="b", method="SOLeNNoID", status="unavailable")],
                                method="SOLeNNoID")
    assert rows[0]["score"] == .5
    assert rows[1]["score"] == ""
    assert rows[1]["status"] == "unavailable"
    assert import_domain_scores([dict(sample_id="c", method="RAPHAEL", status="ok", svm_score=-4)],
                                 method="RAPHAEL")[0]["score"] == -4


def test_cli_protects_existing_results_and_keeps_missing_scores(tmp_path):
    labels, scores, out = [tmp_path / name for name in ["labels.csv", "scores.csv", "result.json"]]
    labels.write_text("sample_id,target,split,group_id,pdb_id\np,1,test,h1,pdb1\nn,0,test,h2,pdb2\n")
    scores.write_text("sample_id,score,status\np,0.8,ok\nn,,missing_model\n")
    args = ["evaluate", "--labels", str(labels), "--scores", str(scores), "--method", "toy",
            "--fixed-threshold", "0.5", "--out", str(out)]
    assert main(args) == 0
    assert json.loads(out.read_text())["coverage"] == .5
    with pytest.raises(SystemExit):
        main(args)


def test_native_hard_calls_have_coverage_without_fabricated_ranking():
    rows = [Score("positive", None, "ok", 1), Score("negative", None, "ok", 0)]
    result = binary_metrics(cases(), rows, threshold=None)
    assert result["coverage"] == 1
    assert result["balanced_accuracy"] == 1
    assert result["score_available_n"] == 0
    assert result["ranking"] == {"average_precision": None, "roc_auc": None}
    with pytest.raises(ValueError, match="Hard calls"):
        binary_metrics(cases(), rows, threshold=.5)
    with pytest.raises(ValueError, match="hard calls"):
        calibrate_threshold(cases("development"), rows, method="cooper_beta")


def test_import_native_barrel_calls_preserves_unavailable_and_missing_score():
    rows = import_domain_scores([
        dict(sample_id="positive", method="cooper_beta", status="ok", score=None, prediction=1),
        dict(sample_id="negative", method="cooper_beta", status="unavailable", reason="DSSP failure"),
    ], method="cooper_beta")
    assert rows[0]["prediction"] == 1 and rows[0]["score"] == ""
    assert rows[1]["prediction"] == rows[1]["score"] == ""
    result = binary_metrics(cases(), [Score("positive", None, "ok", 1)], threshold=None)
    assert result["confusion_by_true_class"] == [[0, 0, 1], [0, 1, 0]]


def test_native_decision_cli_and_score_ranking_are_separate(tmp_path):
    labels = tmp_path / "labels.csv"
    labels.write_text("sample_id,target,split,group_id,pdb_id\npositive,1,test,p,9p00\nnegative,0,test,n,9n00\n")
    scores = tmp_path / "scores.csv"
    scores.write_text("sample_id,score,status,prediction\npositive,,ok,1\nnegative,,ok,0\n")
    out = tmp_path / "native.json"
    assert main(["evaluate", "--labels", str(labels), "--scores", str(scores), "--method",
                 "cooper_beta", "--native-decisions", "--out", str(out)]) == 0
    result = json.loads(out.read_text())
    assert result["operator"] == "native" and result["threshold"] is None
    assert result["accuracy"] == 1 and result["score_available_n"] == 0


def test_task_mixing_rejected_before_import():
    with pytest.raises(ValueError, match="one task"):
        import_domain_scores([
            dict(sample_id="a", method="Foldseek", status="ok", score=.2, task="beta_solenoid"),
            dict(sample_id="b", method="Foldseek", status="ok", score=.1, task="barrel_geometry"),
        ], method="Foldseek")


def test_native_betlas_barrel_mapping_keeps_gate_rejection_and_missing_chain(tmp_path):
    from betlas.benchmarks.specialists import import_betlas_barrel_rows

    path = str(tmp_path / "input.cif")
    native = [dict(source_path=path, chain="A", result="NON_BARREL", result_stage="decision",
                   decision_score="0.0", reason="insufficient_scored_layers")]
    samples = [dict(sample_id="selected", source_path=path, chain_id="A"),
               dict(sample_id="absent", source_path=path, chain_id="B")]
    rows = import_betlas_barrel_rows(native, samples)
    assert rows[0]["status"] == "ok" and rows[0]["prediction"] == 0
    assert rows[0]["reason"] == "insufficient_scored_layers"
    assert rows[1]["status"] == "unavailable" and rows[1]["prediction"] == ""
    with pytest.raises(ValueError, match="Duplicate native"):
        import_betlas_barrel_rows(native * 2, samples)
