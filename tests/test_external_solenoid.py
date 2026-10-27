"""Solenoid adapters tested using invented structures and native-format records."""
import csv
import json
import sys

import pytest

from scripts.external_baselines.solenoid.raphael import parse_native_html
from scripts.external_baselines.solenoid.solennoid import (
    domain_summary,
    map_span_scores,
    parse_native_csv,
    run_baseline,
)
from scripts.external_baselines.solenoid.structures import inspect_structure, matrix_cost


def _structure(path, numbers=(10, 11, 12), *, insertion="", ca_only=False):
    lines = []
    serial = 1
    for position, number in enumerate(numbers):
        for atom, offset, element in [("N", 0.0, "N"), ("CA", 1.0, "C"), ("C", 2.0, "C"), ("O", 2.5, "O")]:
            if ca_only and atom != "CA":
                continue
            lines.append(
                f"ATOM  {serial:5d} {atom:^4s} ALA A{number:4d}{insertion or ' '}   "
                f"{position * 3.0 + offset:8.3f}{0:8.3f}{0:8.3f}{1:6.2f}{0:6.2f}          {element:>2s}\n"
            )
            serial += 1
    path.write_text("".join(lines) + "TER\nEND\n")
    return path


def _native_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["", "chain", "index", "residue", "one_hot", "class"])
        writer.writerows(rows)
    return path


def test_native_scores_preserve_omitted_channels_and_ties(tmp_path):
    inspected = inspect_structure(_structure(tmp_path / "input.pdb"))
    path = _native_csv(tmp_path / "native.csv", [
        [0, "A", 10, "ALA", "[.01 .02 .01 .01]", "beta"],
        [1, "A", 11, "ALA", "[.5 .5 0 0]", "non"],
        [2, "A", 12, "ALA", "[0 0 .4 .6]", "alpha"],
    ])
    predictions = parse_native_csv(path, sample_id="toy", mapping=inspected["residues"])
    assert predictions[0].native_score_beta == .02
    assert predictions[0].omitted_probability_mass == pytest.approx(.95)
    assert [x.predicted_class for x in predictions] == [1, 0, 3]
    summary = domain_summary(predictions, sample_id="toy")
    assert summary["beta_fraction"] == pytest.approx(1 / 3)
    assert summary["any_solenoid_fraction"] == pytest.approx(2 / 3)
    assert summary["score_is_probability"] is False
    assert domain_summary([], sample_id="empty")["beta_fraction"] is None


@pytest.mark.parametrize("row,match", [
    ([0, "A", 10, "GLY", "[1 0 0 0]", "non"], "name"),
    ([0, "B", 10, "ALA", "[1 0 0 0]", "non"], "identity"),
    ([0, "A", 10, "ALA", "[.1 .9 0 0]", "non"], "argmax"),
    ([0, "A", 10, "ALA", "[nan 0 0 0]", "non"], "finite"),
    ([0, "A", 10, "ALA", "[1 1 0 0]", "non"], "sum"),
])
def test_native_csv_rejects_identity_and_score_corruption(tmp_path, row, match):
    mapping = [{"chain_id": "A", "residue_number": 10, "residue_name": "ALA", "insertion_code": ""}]
    path = _native_csv(tmp_path / "native.csv", [row])
    with pytest.raises(ValueError, match=match):
        parse_native_csv(path, sample_id="toy", mapping=mapping)


def test_native_csv_does_not_turn_skipped_chains_into_negatives(tmp_path):
    mapping = inspect_structure(_structure(tmp_path / "input.pdb"))["residues"]
    path = _native_csv(tmp_path / "native.csv", [[0, "A", 10, "ALA", "[1 0 0 0]", "non"]])
    with pytest.raises(ValueError, match="Missing"):
        parse_native_csv(path, sample_id="toy", mapping=mapping)


def test_gaps_preserve_native_padding_and_map_observed_residues_only(tmp_path):
    inspection = inspect_structure(_structure(tmp_path / "gap.pdb", numbers=(10, 12)))
    assert inspection["status"] == "ready"
    assert inspection["matrix_span"] == 3
    assert inspection["n_missing_integer_positions"] == 1
    assert [x["native_span_index"] for x in inspection["residues"]] == [0, 2]
    scores = tmp_path / "span.csv"
    scores.write_text("span_index,score_0,score_1,score_2,score_3\n0,0,1,0,0\n1,1,0,0,0\n2,0,0,0,1\n")
    result = map_span_scores(scores, sample_id="toy", inspection=inspection)
    assert [x.residue_number for x in result] == [10, 12]
    assert [x.predicted_class for x in result] == [1, 3]
    with pytest.raises(ValueError, match="contiguous"):
        parse_native_csv(scores, sample_id="toy", mapping=inspection["residues"])
    scores.write_text("span_index,score_0,score_1,score_2,score_3\n0,0,1,0,0\n1,1,0,0,0\n")
    with pytest.raises(ValueError, match="span"):
        map_span_scores(scores, sample_id="toy", inspection=inspection)


@pytest.mark.parametrize("kwargs,reason", [
    ({"insertion": "A"}, "insertion_codes"),
    ({"ca_only": True}, "native_atom_filter"),
    ({"numbers": (10, 9)}, "nonmonotonic"),
    ({"numbers": (10,)}, "zero_distance"),
])
def test_unrepresentable_coordinate_inputs_are_explicit(tmp_path, kwargs, reason):
    inspection = inspect_structure(_structure(tmp_path / "input.pdb", **kwargs))
    assert inspection["status"] == "unavailable"
    assert reason in inspection["reason"]


def test_span_resource_limit_uses_numbering_not_observed_count(tmp_path):
    path = _structure(tmp_path / "gap.pdb", numbers=(1, 2000))
    inspection = inspect_structure(path, max_span=1000)
    assert inspection["n_observed_ca_residues"] == 2
    assert inspection["matrix_span"] == 2000
    assert inspection["reason"] == "matrix_span_exceeds_explicit_resource_limit"
    assert inspection["distance_matrix_bytes"] == 8 * 2000**2
    cost = matrix_cost(128)
    assert cost["diagonal_tiles"] == 2
    assert cost["native_prediction_matrix_bytes"] == 8 * 128**2 * 6


@pytest.mark.parametrize("residue_name", ["SEC", "PYL"])
def test_genetically_encoded_residues_remain_native_eligible(tmp_path, residue_name):
    path = _structure(tmp_path / "input.pdb")
    path.write_text(path.read_text().replace(" ALA ", f" {residue_name} "))
    inspected = inspect_structure(path)
    assert inspected["status"] == "ready"
    assert inspected["n_observed_ca_residues"] == 3
    assert {r["residue_name"] for r in inspected["residues"]} == {residue_name}


def test_runner_local_stub_and_preserved_inputs(tmp_path, monkeypatch):
    from scripts.external_baselines.solenoid import solennoid
    source = tmp_path / "upstream source"
    source.mkdir()
    (source / "solenoid_detector_onehot_v3.py").touch()
    model = tmp_path / "model.h5"
    model.touch()
    path = _structure(tmp_path / "input.pdb")
    before = path.read_bytes()
    output = tmp_path / "output"
    calls = []

    def stub(command, **kwargs):
        calls.append((command, kwargs))
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "-1"
        assert kwargs["timeout"] == 4
        (output / "native_span_scores.csv").write_text(
            "span_index,score_0,score_1,score_2,score_3\n0,1,0,0,0\n1,0,1,0,0\n2,0,1,0,0\n"
        )
        return type("Completed", (), {"returncode": 0})()

    monkeypatch.setattr(solennoid.subprocess, "run", stub)
    result = run_baseline(path, sample_id="toy", source_dir=source, model_path=model,
                          python_executable=sys.executable, output_dir=output, timeout_seconds=4)
    assert result["beta_fraction"] == pytest.approx(2 / 3)
    assert result["execution_mode"] == "local_cpu_api_worker"
    assert str(source) in calls[0][0]
    assert path.read_bytes() == before
    assert len(list(csv.DictReader((output / "residue_predictions.csv").open()))) == 3
    assert json.loads((output / "result.json").read_text())["status"] == "ok"
    with pytest.raises(FileExistsError):
        run_baseline(path, sample_id="toy", source_dir=source, model_path=model,
                     python_executable=sys.executable, output_dir=output)


def test_unsupported_runner_does_not_invoke_tool(tmp_path, monkeypatch):
    from scripts.external_baselines.solenoid import solennoid
    def unexpected(*args, **kwargs):
        raise AssertionError("Tool must not be invoked for unsupported input")
    monkeypatch.setattr(solennoid.subprocess, "run", unexpected)
    result = run_baseline(_structure(tmp_path / "input.pdb", insertion="A"), sample_id="toy",
                          source_dir=tmp_path, model_path=tmp_path / "missing.h5",
                          python_executable=sys.executable, output_dir=tmp_path / "output")
    assert result["status"] == "unavailable"
    assert "beta_fraction" not in result


def test_raphael_parses_decision_margin_not_total_score(tmp_path):
    html = tmp_path / "result.html"
    html.write_text('<p>Status: <span>finished</span></p><pre>PDB: toyA\n'
                    'Total score: 999\nSVM SCORE: -2.5...... (confidence text)\n'
                    'Predicted repeat length: 23.1</pre>')
    result = parse_native_html(html, sample_id="toy")
    assert result["svm_score"] == -2.5
    assert result["execution_mode"] == "saved_native_result_import"
    assert result["prediction_execution_verified"] is False
    assert result["native_decision"] == "globular"
    assert result["score_is_probability"] is False
    assert result["predicted_repeat_length"] == 23.1
    html.write_text('<p>Status: pending</p><pre>SVM SCORE: -2.5</pre>')
    with pytest.raises(ValueError, match="finished"):
        parse_native_html(html, sample_id="toy")
    html.write_text('<p>Status: finished</p><pre>SVM SCORE: 2\nSVM SCORE: 3</pre>')
    with pytest.raises(ValueError, match="exactly one"):
        parse_native_html(html, sample_id="toy")


def test_legacy_keras_compatibility_accepts_only_noop_group_setting():
    from scripts.external_baselines.solenoid.worker import legacy_groups_one_config
    original = {"groups": 1, "filters": 16, "kernel_size": [2, 2]}
    assert legacy_groups_one_config(original) == {"filters": 16, "kernel_size": [2, 2]}
    assert original["groups"] == 1
    with pytest.raises(ValueError, match="groups=1"):
        legacy_groups_one_config({"groups": 2})


def test_parse_failure_is_preserved_as_unavailable(tmp_path):
    result = run_baseline(tmp_path / "missing.pdb", sample_id="toy", source_dir=tmp_path,
                          model_path=tmp_path / "missing.h5", python_executable=sys.executable,
                          output_dir=tmp_path / "output")
    assert result["status"] == "unavailable"
    assert result["reason"] == "coordinate_parse_failed"
    assert "beta_fraction" not in result
