"""Independent synthetic tests for local protein-search adapter contracts."""

from __future__ import annotations

import json
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.external_baselines.sequence_search import (
    blastp_command,
    mmseqs_command,
    read_hits,
    run_blastp,
    run_mmseqs,
)


def fake_tool(tmp_path: Path, body: str) -> Path:
    tool = tmp_path / "fake protein search"
    tool.write_text(f"#!{sys.executable}\n" + body, encoding="utf-8")
    tool.chmod(tool.stat().st_mode | stat.S_IXUSR)
    return tool


@pytest.mark.parametrize("method", ["blastp", "mmseqs"])
def test_empty_results_preserve_no_hit_search(tmp_path: Path, method: str) -> None:
    output = tmp_path / "empty.tsv"
    output.write_text("")
    assert read_hits(output, method=method) == []


def test_blast_reader_preserves_multiple_hits_and_taxonomy(tmp_path: Path) -> None:
    output = tmp_path / "hits.tsv"
    output.write_text(
        "toy_query\ttoy_target_a\t50\t8\t80\t1e-8\t30\tSynthetic organism\tNA\tToy protein\n"
        "toy_query\ttoy_target_b\t25\t4\t40\t1e-6\t20\tNA\tNA\tAlternative toy protein\n"
    )
    hits = read_hits(output, method="blastp")
    assert [hit["saccver"] for hit in hits] == ["toy_target_a", "toy_target_b"]
    assert hits[0]["sscinames"] == "Synthetic organism"
    assert hits[1]["stitle"] == "Alternative toy protein"


def test_mmseqs_reader_preserves_coordinate_fields(tmp_path: Path) -> None:
    output = tmp_path / "hits.tsv"
    output.write_text("toy_query\ttoy_target\t50\t8\t2\t9\t3\t10\t1e-6\t20\n")
    hits = read_hits(output, method="mmseqs")
    assert hits[0]["qstart"] == "2"
    assert hits[0]["tend"] == "10"
    assert hits[0]["bits"] == "20"


def test_malformed_rows_are_rejected(tmp_path: Path) -> None:
    output = tmp_path / "malformed.tsv"
    output.write_text("toy_query\ttoy_target\n")
    with pytest.raises(ValueError, match="expected 10 fields, got 2"):
        read_hits(output, method="mmseqs")


def test_blast_exec_keeps_spaced_paths_and_database_environment(tmp_path: Path) -> None:
    query = tmp_path / "toy queries.faa"
    query.write_text(">toy_query\nACDEFGHIK\n")
    output = tmp_path / "results" / "hits.tsv"
    database = tmp_path / "protein database"
    tool = fake_tool(tmp_path, '''import json, os, sys
from pathlib import Path
args = sys.argv[1:]
assert "-remote" not in args
assert Path(args[args.index("-query") + 1]).read_text().startswith(">toy_query")
output = Path(args[args.index("-out") + 1])
output.write_text("")
output.with_suffix(".json").write_text(json.dumps({"args": args, "blastdb": os.environ["BLASTDB"]}))
''')
    assert run_blastp(query, database, output, executable=tool) == output
    captured = json.loads(output.with_suffix(".json").read_text())
    assert captured["args"][captured["args"].index("-db") + 1] == str(database)
    assert captured["blastdb"].split(":")[0] == str(tmp_path)


def test_mmseqs_exec_uses_explicit_target_and_temporary_paths(tmp_path: Path) -> None:
    query, target = tmp_path / "query.faa", tmp_path / "target.faa"
    query.write_text(">toy_query\nACDEFGHIK\n")
    target.write_text(">toy_target\nACDEFGHIK\n")
    output = tmp_path / "results" / "hits.tsv"
    temporary = tmp_path / "temporary files"
    tool = fake_tool(tmp_path, '''import sys
from pathlib import Path
assert sys.argv[1] == "easy-search"
assert Path(sys.argv[2]).read_text().startswith(">toy_query")
assert Path(sys.argv[3]).read_text().startswith(">toy_target")
assert Path(sys.argv[5]).is_dir()
Path(sys.argv[4]).write_text("toy_query\\ttoy_target\\t100\\t9\\t1\\t9\\t1\\t9\\t1e-8\\t30\\n")
''')
    run_mmseqs(query, target, output, temporary, executable=tool)
    assert read_hits(output, method="mmseqs")[0]["pident"] == "100"


def test_tool_failure_is_not_a_successful_empty_result(tmp_path: Path) -> None:
    query = tmp_path / "query.faa"
    query.write_text(">toy_query\nACDEFGHIK\n")
    tool = fake_tool(tmp_path, "raise SystemExit(7)\n")
    with pytest.raises(subprocess.CalledProcessError) as error:
        run_blastp(query, tmp_path / "database", tmp_path / "hits.tsv", executable=tool)
    assert error.value.returncode == 7


@pytest.mark.parametrize("evalue", [0, -1, float("nan"), float("inf")])
def test_invalid_threshold_rejected_before_search(evalue: float) -> None:
    with pytest.raises(ValueError, match="E-value"):
        blastp_command("query.faa", "database", "hits.tsv", evalue=evalue)
    with pytest.raises(ValueError, match="E-value"):
        mmseqs_command("query.faa", "target.faa", "hits.tsv", "tmp", evalue=evalue)


def test_search_cannot_overwrite_query(tmp_path: Path) -> None:
    query = tmp_path / "query.faa"
    original = ">toy_query\nACDEFGHIK\n"
    query.write_text(original)
    with pytest.raises(ValueError, match="differ"):
        run_blastp(query, "database", query, executable="unused")
    assert query.read_text() == original
