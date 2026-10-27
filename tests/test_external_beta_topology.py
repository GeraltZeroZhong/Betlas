"""Synthetic native-format contracts; no research records or upstream fixtures."""

import json
import subprocess
import sys

import pytest

from scripts.external_baselines.beta_topology.formats import (
    FormatError,
    parse_bbcontacts,
    parse_bcov,
    parse_betapro_map,
    parse_betapro_str,
)
from scripts.external_baselines.beta_topology.runner import (
    _execute,
    run_bbcontacts,
    run_bcov,
    run_betapro,
    validate_psicov_msa,
)


def put(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return path


def test_bcov_native_indices_bridge_and_unscored_contacts(tmp_path):
    strands = put(tmp_path, "input.ss", "3\n0 0 1\n1 4 5\n2 8 8\n")
    topo = put(tmp_path, "out.topo", "3\n0 1 1\n1 2 2\n")
    contacts = put(tmp_path, "out.bmap", "9\n0 5\n1 4\n5 8\n")
    result = parse_bcov(topo, contacts, strands, sample_id="toy", length=9)
    assert result["contacts"][0] == dict(residue_i=1, residue_j=6,
        orientation="antiparallel", score=None, score_kind="hard_prediction")
    assert result["strand_pairs"][1]["orientation"] == "bridge"
    assert result["strands"][2] == dict(strand_id=3, start=9, end=9)
    contacts.write_text("9\n0 5 0.8\n")
    with pytest.raises(FormatError, match="two indices"):
        parse_bcov(topo, contacts, strands, sample_id="toy", length=9)


def test_bcov_rejects_unmapped_contacts_and_accepts_explicit_empty(tmp_path):
    strands = put(tmp_path, "input.ss", "2\n0 0 1\n1 4 5\n")
    topo = put(tmp_path, "out.topo", "2\n")
    contacts = put(tmp_path, "out.bmap", "6\n")
    assert parse_bcov(topo, contacts, strands, sample_id="toy", length=6)["status"] == "empty"
    contacts.write_text("6\n0 4\n")
    with pytest.raises(FormatError, match="no corresponding"):
        parse_bcov(topo, contacts, strands, sample_id="toy", length=6)


def test_bbcontacts_path_scores_and_canonical_pairs(tmp_path):
    path = put(tmp_path, "out.txt", "#identifier diversity direction viterbiscore indexpred state res1 res2\n"
               "toy 0.40 Antiparallel -2.1 1 first 8 2\n"
               "toy 0.40 Antiparallel -2.1 1 bulgei1 7 3\n")
    result = parse_bbcontacts(path, sample_id="toy", length=10, successful_exit=True)
    assert result["contacts"][0]["residue_i"] == 2
    assert result["contacts"][0]["score"] == -2.1
    assert result["contacts"][0]["score_kind"] == "path_viterbi_score"
    assert result["contacts"][1]["native_state"] == "bulgei1"


def test_bbcontacts_empty_requires_success_and_rejects_append(tmp_path):
    path = put(tmp_path, "out.txt", "toy 0.40 NA NA NA NA NA NA\n")
    with pytest.raises(FormatError, match="successful"):
        parse_bbcontacts(path, sample_id="toy", length=10, successful_exit=False)
    assert parse_bbcontacts(path, sample_id="toy", length=10, successful_exit=True)["status"] == "empty"
    path.write_text(path.read_text() * 2)
    with pytest.raises(FormatError, match="appended"):
        parse_bbcontacts(path, sample_id="toy", length=10, successful_exit=True)
    path.write_text("# header only\n")
    with pytest.raises(FormatError, match="incomplete"):
        parse_bbcontacts(path, sample_id="toy", length=10, successful_exit=True)


def test_betapro_map_uses_beta_positions_not_dense_indices(tmp_path):
    path = put(tmp_path, "out.map", "toy\nACDEFG\n.E..E.\n0 0 0 0 0 0\n0 0 0 0 0 0\n0 0.7\n0.7 0\n")
    result = parse_betapro_map(path, sample_id="toy", expected_sequence="ACDEFG")
    assert result["score_matrix"] == [dict(residue_i=2, residue_j=5,
        orientation="unspecified", score=0.7, score_kind="residue_pair_probability")]
    path.write_text(path.read_text().replace("0.7 0", "0.6 0"))
    with pytest.raises(FormatError, match="Asymmetric"):
        parse_betapro_map(path, sample_id="toy")


def strand_report(alignment="1--2:A:[2-3:8-7]:1.2"):
    return ("Name:\ntoy\n\nSequence:\nACDEFGHIK\n\n"
            "Secondary Structure(H:helix, E:Beta Strand/Bridge, C:Coil):\nCEECCCEEC\n\n"
            "The predicted beta-strands:\n1:[2-3]:[1-2]\n2:[7-8]:[3-4]\n\n"
            "The predicted strand pairs and strand alignments:\n" + alignment + "\n\n"
            "The predicted beta-sheets:\nThe number of beta-sheets: 1\n")


def test_betapro_alignment_expands_antiparallel_sequence_positions(tmp_path):
    path = put(tmp_path, "out.str", strand_report())
    result = parse_betapro_str(path, sample_id="toy")
    assert [(r["residue_i"], r["residue_j"]) for r in result["contacts"]] == [(2, 8), (3, 7)]
    assert result["strand_pairs"][0]["score_kind"] == "pseudo_binding_energy"
    path.write_text(strand_report("1--2:P:[2-3:8-7]:1.2"))
    with pytest.raises(FormatError, match="direction"):
        parse_betapro_str(path, sample_id="toy")


def test_betapro_detects_partial_wrapper_output(tmp_path):
    path = put(tmp_path, "out.str", strand_report().replace("2:[7-8]:[3-4]", ""))
    with pytest.raises(FormatError, match="cover exactly"):
        parse_betapro_str(path, sample_id="toy")


def test_msa_rejects_query_gaps_and_ragged_sequences(tmp_path):
    path = put(tmp_path, "msa.txt", "ACDE\nA-DE\n")
    assert validate_psicov_msa(path, "ACDE") == 2
    path.write_text("A-DE\nACDE\n")
    with pytest.raises(ValueError, match="first MSA"):
        validate_psicov_msa(path, "ACDE")
    path.write_text("ACDE\nADE\n")
    with pytest.raises(ValueError, match="equal-width"):
        validate_psicov_msa(path, "ACDE")


def test_bcov_runner_uses_native_arguments(tmp_path):
    script = put(tmp_path, "fake.py", "import sys,pathlib\na=sys.argv\n"
        "assert '-m' in a and '-a' not in a\n"
        "pathlib.Path(a[a.index('-o')+1]).write_text('2\\n0 1 0\\n')\n"
        "pathlib.Path(a[a.index('-c')+1]).write_text('6\\n0 4\\n')\n")
    strands = put(tmp_path, "input.ss", "2\n0 0 1\n1 4 5\n")
    matrix = put(tmp_path, "input.mat", "6\n0 4 0.7\n")
    result = run_bcov(command=[sys.executable, str(script)], sample_id="toy", sequence="ACDEFG",
        strands_file=strands, scores=matrix, ss_source="observed", output_dir=tmp_path / "run", timeout=3)
    assert result["contacts"][0]["orientation"] == "parallel"
    assert result["secondary_structure_source"] == "observed"


def test_bbcontacts_nonzero_with_na_is_failure(tmp_path):
    script = put(tmp_path, "fake.py", "import pathlib,sys\n"
        "pathlib.Path(sys.argv[3]+'.filteredoutput.txt').write_text('toy 0.5 NA NA NA NA NA NA\\n')\n"
        "raise SystemExit(7)\n")
    matrix = put(tmp_path, "input.mat", "0\t0\n0\t0\n")
    ss = put(tmp_path, "input.ss", ">toy\nEE\n")
    with pytest.raises(subprocess.CalledProcessError) as error:
        run_bbcontacts(command=[sys.executable, str(script)], sample_id="toy", sequence="AC",
            coupling_matrix=matrix, secondary_structure=ss, ss_source="predicted",
            alignment_sequence_count=1, output_dir=tmp_path / "run", timeout=3)
    assert error.value.returncode == 7
    assert not (tmp_path / "run" / "normalized.json").exists()


def test_timeout_records_failure_and_reaps_process(tmp_path):
    with pytest.raises(subprocess.TimeoutExpired):
        _execute([sys.executable, "-c", "import time;time.sleep(30)"], tmp_path, 0.1, "slow")
    receipt = json.loads((tmp_path / "slow.execution.json").read_text())
    assert receipt["status"] == "timeout"
    assert receipt["returncode"] < 0


def test_betapro_rejects_unconfigured_installation(tmp_path):
    root = tmp_path / "betapro"
    (root / "bin").mkdir(parents=True)
    (root / "bin" / "predict_strand.sh").write_text("/home/baldig/jianlin/old/predictor\n")
    source = put(tmp_path, "input.map", "placeholder\n")
    with pytest.raises(ValueError, match="historical"):
        run_betapro(installation=root, sample_id="toy", sequence="AC", input_file=source,
            input_kind="probability_map", ss_source="predicted", output_dir=tmp_path / "run", timeout=3)
    assert not (tmp_path / "run").exists()
