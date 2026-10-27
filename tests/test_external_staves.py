"""Synthetic output schemas and mocked local-tool contracts for stave adapters.

All sequences, labels, numeric summaries, and profile rows below are independently
constructed. External predictors and model weights are never loaded by these tests.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.external_baselines.beta_barrel_staves import parse_counts
from scripts.external_baselines.beta_barrel_staves.profile_from_pssm import (
    pssm_to_betaware_profile,
)

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts/external_baselines/beta_barrel_staves"


def test_segment_counts_and_boundaries():
    assert parse_counts.count_segments("", {"B", "b"}) == 0
    assert parse_counts.count_segments("BbiiBBb", {"B", "b"}) == 2
    assert parse_counts.count_segments("iii", {"B", "b"}) == 0


def test_tmbed_sequence_and_topology_records():
    result = parse_counts.parse_tmbed(
        ">toy description\nACGTACGT\niiBBooBb\n>zero\nACGT\niiii\n>truncated\nACGT\n"
    )
    assert result["toy"]["strand_count"] == 2
    assert result["zero"]["strand_count"] == 0
    assert result["truncated"]["strand_count"] is None
    assert result["truncated"]["status"] == "failed"
    assert parse_counts.parse_tmbed(">mismatch\nACGT\nBBB\n")["mismatch"]["strand_count"] is None


def test_juchmme_viterbi_counts_preserve_missing_path():
    result = parse_counts.parse_juchmme(
        "log line\nID: >toy description\nVP: IIMMMOOMM\nID: >missing\nID: >empty\nVP:\n"
    )
    assert result["toy"]["strand_count"] == 2
    assert result["missing"]["strand_count"] is None
    assert result["empty"]["strand_count"] is None


@pytest.mark.parametrize("heading", ["Topology", "TMB Strands"])
def test_betaware_wrapped_intervals(heading):
    text = f"{heading}   : 2-4, 7-9,\n                12-14\nOther field: 1\n"
    assert parse_counts.parse_betaware(text, "toy")["toy"]["strand_count"] == 3
    assert parse_counts.parse_betaware("Predicted TMBB : No 0.2", "toy")["toy"]["strand_count"] == 0
    assert parse_counts.parse_betaware("incomplete log", "toy")["toy"]["strand_count"] is None
    assert parse_counts.parse_betaware("Topology: invalid", "toy")["toy"]["status"] == "failed"


def test_proftmb_counts_keep_identifiers_and_unavailable_values():
    records = parse_counts.parse_proftmb(
        "SeqID\t#_Predicted_Strands\nquery.Q\t8\nzero.Q\t0\nmissing.Q\tNA\n"
    )
    assert records["query"]["strand_count"] == 8
    assert records["zero"]["strand_count"] == 0
    assert records["missing"]["strand_count"] is None
    with pytest.raises(ValueError, match="columns"):
        parse_counts.parse_proftmb("other\tcount\n")


def test_polarbearal_summary_ignores_diagnostics():
    records = parse_counts.parse_polarbearal(
        "diagnostic text\nTOY\t8\t1.2e1\t2.0\t-3.0\t10\nnoise\tbad\t0\t0\t0\t0\n"
    )
    assert set(records) == {"TOY"}
    assert records["TOY"]["strand_count"] == 8


def test_profile_conversion_uses_observed_percentages(tmp_path):
    source = tmp_path / "toy.pssm"
    source.write_text("header\n1 A " + " ".join(["-2"] * 20 + ["50", "25", "25"] + ["0"] * 17) + "\n")
    output = tmp_path / "profile"
    pssm_to_betaware_profile(source, output)
    assert output.read_text().split() == ["0.50", "0.25", "0.25"] + ["0.00"] * 17
    source.write_text("header only\n")
    with pytest.raises(ValueError, match="No ASCII PSSM rows"):
        pssm_to_betaware_profile(source, output)


@pytest.mark.parametrize("script", sorted(path.name for path in SCRIPTS.glob("*.sh")))
def test_shell_help_requires_no_external_installation(script, tmp_path):
    result = subprocess.run(
        ["bash", str(SCRIPTS / script), "--help"],
        env={"PATH": os.environ["PATH"]}, cwd=tmp_path, text=True, capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert "Usage:" in result.stdout


@pytest.fixture
def mocked_tool(tmp_path):
    executable = tmp_path / "fake tool"
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "Path(os.environ['TEST_CAPTURE']).write_text(json.dumps({'argv': sys.argv, 'cwd': os.getcwd(), "
        "'offline': os.environ.get('HF_HUB_OFFLINE'), 'betaware_root': os.environ.get('BETAWARE_ROOT')}))\n"
        "print('synthetic tool output')\n"
    )
    executable.chmod(0o755)
    capture = tmp_path / "capture.json"
    env = dict(os.environ, TEST_CAPTURE=str(capture))
    return executable, capture, env


def run_wrapper(name, args, env, cwd):
    result = subprocess.run(
        ["bash", str(SCRIPTS / name), *map(str, args)],
        env=env, cwd=cwd, text=True, capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    return result


def test_betaware_wrapper_preserves_profile_options(tmp_path, mocked_tool):
    executable, capture, env = mocked_tool
    tool = tmp_path / "betaware"
    tool.mkdir()
    (tool / "betaware.py").write_text("# stand-in source\n")
    fasta, profile, output = [tmp_path / name for name in ["toy.fasta", "toy.profile", "result"]]
    fasta.write_text(">toy\nAAAA\n")
    profile.write_text("synthetic profile\n")
    env.update(BETAWARE_ROOT=str(tool), PYTHON_BIN=str(executable), BETAWARE_THRESHOLD="0.4", BETAWARE_AA_ORDER="ARNDCQEGHILKMFPSTWYV")
    run_wrapper("run_betaware.sh", [fasta, profile, output], env, tmp_path)
    call = json.loads(capture.read_text())
    assert call["argv"][1:] == [str(tool / "betaware.py"), "-f", str(fasta), "-p", str(profile), "-s", "0.4", "-t", "-a", "ARNDCQEGHILKMFPSTWYV"]
    assert call["betaware_root"] == str(tool)
    assert output.read_text() == "synthetic tool output\n"


@pytest.mark.parametrize("mode,option,table,config", [
    ("hmm", "-e", "E_TMBB2_TRAINED", "conf.tmbb"),
    ("hnn", "-w", "W_PREDTMBB2_MYMODEL", "conf.tmbbHNN"),
])
def test_juchmme_wrapper_uses_mode_specific_tables(tmp_path, mocked_tool, mode, option, table, config):
    executable, capture, env = mocked_tool
    tool = tmp_path / "juchmme"
    (tool / "bin").mkdir(parents=True)
    fasta = tmp_path / "toy.fasta"
    fasta.write_text(">toy\nAAAA\n")
    output = tmp_path / "result"
    env.update(JUCHMME_ROOT=str(tool), JAVA_BIN=str(executable))
    run_wrapper("run_juchmme_pred_tmbb2.sh", [mode, fasta, output], env, tmp_path)
    argv = json.loads(capture.read_text())["argv"]
    assert argv[argv.index(option) + 1] == str(tool / "tables" / table)
    assert argv[argv.index("-c") + 1] == str(tool / "conf" / config)
    assert output.with_suffix(".err").is_file()


def test_tmbed_wrapper_resolves_relative_inputs_before_chdir(tmp_path, mocked_tool):
    executable, capture, env = mocked_tool
    tool = tmp_path / "tmbed"
    tool.mkdir()
    (tmp_path / "toy.fasta").write_text(">toy\nAAAA\n")
    (tmp_path / "toy.h5").write_text("mocked embedding input\n")
    env.update(TMBED_ROOT=str(tool), TMBED_PYTHON=str(executable), TMBED_THREADS="2")
    run_wrapper("run_tmbed.sh", ["toy.fasta", "toy.h5", "output/result.pred"], env, tmp_path)
    call = json.loads(capture.read_text())
    argv = call["argv"]
    assert call["cwd"] == str(tool)
    assert call["offline"] == "1"
    assert argv[argv.index("-f") + 1] == str(tmp_path / "toy.fasta")
    assert argv[argv.index("-e") + 1] == str(tmp_path / "toy.h5")
    assert argv[argv.index("-p") + 1] == str(tmp_path / "output/result.pred")
    assert "--no-use-gpu" in argv


def test_proftmb_wrapper_accepts_query_directory(tmp_path, mocked_tool):
    executable, capture, env = mocked_tool
    tool = tmp_path / "proftmb"
    (tool / "usr/share/proftmb").mkdir(parents=True)
    query = tmp_path / "queries"
    query.mkdir()
    env.update(PROFTMB_ROOT=str(tool), PROFTMB_BIN=str(executable))
    run_wrapper("run_proftmb.sh", [query, tmp_path / "output/result"], env, tmp_path)
    argv = json.loads(capture.read_text())["argv"]
    assert argv[argv.index("-q") + 1] == str(query)
    assert argv[1] == "@" + str(tool / "usr/share/proftmb/options")


def test_polarbearal_wrapper_uses_explicit_compiled_shim(tmp_path, mocked_tool):
    executable, capture, env = mocked_tool
    tool = tmp_path / "polar"
    tool.mkdir()
    (tool / "Program.cs").write_text("// stand-in source\n")
    dll = tmp_path / "shim.dll"
    dll.write_text("mocked assembly\n")
    pdb = tmp_path / "toy.pdb"
    pdb.write_text("END\n")
    env.update(POLARBEARAL_ROOT=str(tool), POLARBEARAL_DLL=str(dll), DOTNET_BIN=str(executable))
    run_wrapper("run_polarbearal.sh", [pdb, tmp_path / "output"], env, tmp_path)
    assert json.loads(capture.read_text())["argv"][1:] == [str(dll), str(tool), str(pdb), str(tmp_path / "output")]
    assert (tmp_path / "output/polarbearal.stdout").read_text() == "synthetic tool output\n"


def test_polarbearal_build_patches_only_copy_and_defaults_offline(tmp_path, mocked_tool):
    executable, capture, env = mocked_tool
    source = tmp_path / "source"
    source.mkdir()
    original = 'case "10":\n    change_to_custom_data();\n    default:\n'
    (source / "Program.cs").write_text(original)
    env.update(DOTNET_BIN=str(executable), PYTHON_BIN=sys.executable)
    build = tmp_path / "build"
    run_wrapper("build_polarbearal.sh", [source, build], env, tmp_path)
    assert (source / "Program.cs").read_text() == original
    assert "break;" in (build / "source/Program.cs").read_text()
    assert "--no-restore" in json.loads(capture.read_text())["argv"]


def test_parser_cli_writes_normalized_tsv_and_rejects_empty_input(tmp_path):
    source, output = tmp_path / "toy.pred", tmp_path / "counts.tsv"
    source.write_text(">toy\nACGTACGT\niiBBooBb\n")
    assert parse_counts.main(["--method", "tmbed", "--input", str(source), "--output", str(output)]) == 0
    lines = output.read_text().splitlines()
    assert lines[1].split("\t") == ["toy", "tmbed", "2", "ok", "iiBBooBb"]
    source.write_text("")
    with pytest.raises(SystemExit) as error:
        parse_counts.main(["--method", "tmbed", "--input", str(source), "--output", str(output)])
    assert error.value.code == 2
