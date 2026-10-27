from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

import betlas.provenance as provenance
from betlas.provenance import source_tree_state


def test_source_tree_state_records_only_tracked_files(tmp_path: Path, monkeypatch) -> None:
    if shutil.which("git") is None:
        pytest.skip("Git is required for tracked-source provenance")
    repo = tmp_path / "project"
    repo.mkdir()
    (repo / "pyproject.toml").write_text("[project]\nname = 'toy'\nversion = '0.0.0'\n")
    source = repo / "src"
    source.mkdir()
    (source / "sample.py").write_text("VALUE = 3\n")
    (source / "scratch.py").write_text("VALUE = 7\n")
    (repo / ".gitignore").write_text("src/scratch.py\n")
    subprocess.run(["git", "init", "--quiet", str(repo)], check=True)
    subprocess.run(["git", "add", "--", "src/sample.py"], cwd=repo, check=True)
    monkeypatch.chdir(repo)
    monkeypatch.setattr(provenance, "__file__", str(source / "betlas" / "provenance.py"))

    state = source_tree_state(paths=("src",))

    assert state["basis"] == "git_ls_files"
    assert state["paths"] == ["src/sample.py"]
    assert state["file_count"] == 1
    assert state["total_bytes"] == len("VALUE = 3\n")


def test_nested_export_stops_repository_discovery_at_package_boundary(
    tmp_path: Path, monkeypatch
) -> None:
    parent = tmp_path / "parent"
    (parent / ".git").mkdir(parents=True)
    export = parent / "export"
    package = export / "src" / "betlas"
    package.mkdir(parents=True)
    (export / "pyproject.toml").write_text("[project]\nname = 'toy'\nversion = '0.0.0'\n")
    source = package / "sample.py"
    source.write_text("VALUE = 3\n")
    monkeypatch.chdir(export)
    monkeypatch.setattr(provenance, "__file__", str(package / "provenance.py"))

    assert provenance.source_repo_root() is None
    assert provenance.source_repo_root(package) is None
    assert provenance.git_state()["available"] is False
    state = source_tree_state(paths=("src",))
    assert state["basis"] == "recursive_files"
    assert state["paths"] == ["src/betlas/sample.py"]


def test_target_install_does_not_inherit_parent_repository(tmp_path: Path, monkeypatch) -> None:
    parent = tmp_path / "parent"
    (parent / ".git").mkdir(parents=True)
    target = parent / "custom-target"
    package = target / "betlas"
    package.mkdir(parents=True)
    monkeypatch.chdir(target)
    monkeypatch.setattr(provenance, "__file__", str(package / "provenance.py"))

    assert provenance.source_repo_root() is None
    state = provenance.git_state()
    assert state["available"] is False
    assert state["root"] == ""
    assert state["commit"] == ""
