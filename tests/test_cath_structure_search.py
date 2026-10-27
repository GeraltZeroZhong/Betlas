"""Training-library and failure-denominator checks for local Foldseek baselines."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd
import pytest
from Bio.PDB import MMCIFIO

from betlas.benchmarks import classification as classification
from betlas.benchmarks import structure_search


def _structure(path, *, multichain=False):
    data = {"data_": "synthetic"}
    rows = []
    for chain in (["A", "B"] if multichain else ["A"]):
        for residue in range(1, 5):
            rows.append(["ATOM", str(len(rows) + 1), "CA", "ALA", chain, chain, str(residue),
                         str(residue), "?", ".", "1", str(residue), "0", "0", "1"])
    columns = ("group_PDB", "id", "label_atom_id", "label_comp_id", "auth_asym_id", "label_asym_id",
               "auth_seq_id", "label_seq_id", "pdbx_PDB_ins_code", "label_alt_id", "occupancy",
               "Cartn_x", "Cartn_y", "Cartn_z", "pdbx_PDB_model_num")
    for key, values in zip(columns, zip(*rows, strict=True), strict=True):
        data[f"_atom_site.{key}"] = list(values)
    writer = MMCIFIO()
    writer.set_dict(data)
    writer.save(str(path))


@pytest.fixture
def benchmark_inputs(tmp_path):
    rows, exports, splits = [], [], []
    specifications = [
        ("trainA", "2.10", "train", 1), ("trainB", "2.20", "train", 1),
        ("queryA", "2.10", "test", 2), ("queryB", "2.20", "test", 2),
        ("missing", "2.20", "test", 3), ("multichain", "2.10", "test", 3),
        ("rare", "2.30", "unavailable", 1), ("purged", "2.10", "purged", 4),
    ]
    for i, (identifier, label, role, family) in enumerate(specifications):
        topology, homology = label + ".1", label + f".1.{family}"
        row = dict(record_id=identifier, pdb_id=f"9a{i:02d}", cath_architecture_code=label,
                   cath_topology_code=topology, cath_homology_code=homology,
                   cath_s35_cluster_id=homology + ".1", structure_path="Original full structure is never searched")
        rows.append(row)
        splits.append({**row, "split_id": "fixed", "role": role, "group_id": homology,
                       "target_label": label, "label_column": classification.A,
                       "group_column": classification.H, "eligibility_reason": "synthetic"})
        path = tmp_path / (identifier + ".cif")
        _structure(path, multichain=identifier == "multichain")
        exports.append(dict(record_id=identifier, structure_path=str(path),
                            export_status="mapping_failed" if identifier == "missing" else "ok"))
    binary = tmp_path / "foldseek"
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)
    return pd.DataFrame(rows), pd.DataFrame(exports), pd.DataFrame(splits), binary


def _token(record_id):
    return record_id.encode().hex() + "_A"


@pytest.mark.parametrize("label_column", [classification.A, classification.T])
def test_foldseek_searches_only_training_domains_and_keeps_all_denominators(
    benchmark_inputs, tmp_path, monkeypatch, label_column,
):
    manifest, exports, splits, binary = benchmark_inputs
    splits["target_label"], splits["label_column"] = splits[label_column], label_column
    calls = []
    working_root = tmp_path / "linux-scratch"

    def backend(command, **kwargs):
        if command[1] == "version":
            return subprocess.CompletedProcess(command, 0, "10.synthetic\n", "")
        queries, training, hits = map(Path, command[2:5])
        working = Path(command[5])
        assert working.is_relative_to(working_root)
        assert not working.is_relative_to(tmp_path / "out")
        # The native backend creates symlinks in this working tree.
        probe = working.parent / "symlink-probe"
        probe.symlink_to(training)
        assert probe.is_dir()
        assert {bytes.fromhex(p.stem).decode() for p in training.iterdir()} == {"trainA", "trainB"}
        assert {bytes.fromhex(p.stem).decode() for p in queries.iterdir()} == {"queryA", "queryB"}
        assert command[command.index("--gpu") + 1] == "0"
        assert command[command.index("--threads") + 1] == "1"
        # Equal scores must choose by stable target ID, independent of line order.
        hits.write_text(f"{_token('queryA')}\t{_token('trainB')}\t50\t1e-8\n"
                        f"{_token('queryA')}\t{_token('trainA')}\t50\t1e-8\n")
        calls.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", backend)
    predictions, summary, per_class = structure_search.run_structure_search(
        manifest, exports, splits, tmp_path / "out", foldseek_binary=binary, threads=1,
        label_column=label_column, scratch_dir=working_root)
    indexed = predictions.set_index("record_id")
    assert len(calls) == 1
    assert summary["scratch_dir"] == str(working_root.resolve())
    assert list(working_root.iterdir()) == []
    assert indexed.loc["queryA", "prediction"] == manifest.set_index("record_id").loc["trainA", label_column]
    assert indexed.loc["queryB", "prediction_status"] == "no_hit"
    assert indexed.loc["missing", "prediction_status"] == "export_unavailable"
    assert indexed.loc["multichain", "prediction_status"] == "multichain_requires_explicit_pooling"
    assert indexed.loc["rare", "role"] == "unavailable"
    assert indexed.loc["purged", "role"] == "purged"
    assert summary["n_test"] == 4 and summary["n_predictions"] == 1
    assert summary["effectiveness_accuracy"] == 0.25
    assert per_class.test_support.sum() == 4
    training_manifest = pd.read_csv(tmp_path / "out" / "splits" / b"fixed".hex() / "training.csv")
    assert set(training_manifest.record_id) == {"trainA", "trainB"}


def test_outside_training_hit_is_protocol_error(benchmark_inputs, tmp_path, monkeypatch):
    manifest, exports, splits, binary = benchmark_inputs

    def backend(command, **kwargs):
        if command[1] == "version":
            return subprocess.CompletedProcess(command, 0, "10.synthetic", "")
        Path(command[4]).write_text(f"{_token('queryA')}\t{_token('queryB')}\t99\t1e-20\n")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", backend)
    with pytest.raises(ValueError, match="outside the frozen training"):
        structure_search.run_structure_search(manifest, exports, splits, tmp_path / "out", foldseek_binary=binary)
    assert not (tmp_path / "out" / "predictions.csv").exists()


def test_post_purge_unavailable_rows_preserve_planned_held_out_and_input_qc(
    benchmark_inputs, tmp_path, monkeypatch,
):
    manifest, exports, splits, binary = benchmark_inputs
    splits["held_out"] = splits.role.eq("test")
    splits["pdb_policy"] = "purge"
    splits.loc[splits.record_id.eq("rare"), ["held_out", "eligibility_reason"]] = [
        True, "architecture_absent_after_purge"]
    manifest["coordinate_status"] = "ok"
    manifest.loc[manifest.record_id.eq("missing"), "coordinate_status"] = "unavailable"
    splits.loc[splits.record_id.eq("missing"), ["role", "eligibility_reason"]] = [
        "unavailable", "input_source_unavailable"]
    # CSV-loaded booleans must have the same meaning as in-memory split flags.
    splits["held_out"] = splits.held_out.astype(str)

    def backend(command, **kwargs):
        if command[1] == "version":
            return subprocess.CompletedProcess(command, 0, "10.synthetic", "")
        Path(command[4]).write_text("")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", backend)
    predictions, summary, _ = structure_search.run_structure_search(
        manifest, exports, splits, tmp_path / "out", foldseek_binary=binary)
    indexed = predictions.set_index("record_id")
    assert summary["n_planned_held_out"] == 5
    assert summary["n_test"] == 3
    assert summary["excluded_records"] == 2
    assert summary["input_source_unavailable_records"] == 1
    assert indexed.loc["rare", "held_out"]
    assert indexed.loc["rare", "role"] == "unavailable"
    assert indexed.loc["rare", "input_source_available"]
    assert not indexed.loc["missing", "input_source_available"]
    assert indexed.loc["missing", "coordinate_status"] == "unavailable"
    assert predictions.pdb_policy.eq("purge").all()


def test_search_execution_failure_stays_in_test_denominator(benchmark_inputs, tmp_path, monkeypatch):
    manifest, exports, splits, binary = benchmark_inputs

    def backend(command, **kwargs):
        if command[1] == "version":
            return subprocess.CompletedProcess(command, 0, "10.synthetic", "")
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(subprocess, "run", backend)
    predictions, summary, _ = structure_search.run_structure_search(
        manifest, exports, splits, tmp_path / "out", foldseek_binary=binary)
    assert set(predictions[predictions.record_id.isin(["queryA", "queryB"])].prediction_status) == {"search_failed"}
    assert summary["n_test"] == 4 and summary["n_predictions"] == 0
    assert summary["effectiveness_accuracy"] == 0


def test_homology_or_pdb_leakage_rejected_before_execution(benchmark_inputs, tmp_path, monkeypatch):
    manifest, exports, splits, binary = benchmark_inputs
    manifest.loc[manifest.record_id.eq("queryA"), "pdb_id"] = manifest.loc[manifest.record_id.eq("trainA"), "pdb_id"].iloc[0]

    def forbidden(*args, **kwargs):
        pytest.fail("Leaking split must fail before running the backend")

    monkeypatch.setattr(subprocess, "run", forbidden)
    with pytest.raises(ValueError, match="pdb_id leakage"):
        structure_search.run_structure_search(manifest, exports, splits, tmp_path / "out", foldseek_binary=binary)


def test_missing_training_collection_does_not_execute_search(benchmark_inputs, tmp_path, monkeypatch):
    manifest, exports, splits, binary = benchmark_inputs
    exports.loc[exports.record_id.isin(["trainA", "trainB"]), "export_status"] = "mapping_failed"

    def version_only(command, **kwargs):
        assert command[1] == "version"
        return subprocess.CompletedProcess(command, 0, "10.synthetic", "")

    monkeypatch.setattr(subprocess, "run", version_only)
    predictions, summary, _ = structure_search.run_structure_search(
        manifest, exports, splits, tmp_path / "out", foldseek_binary=binary)
    assert set(predictions[predictions.record_id.isin(["queryA", "queryB"])].prediction_status) == {"no_training_structures"}
    assert summary["n_test"] == 4 and summary["n_predictions"] == 0
