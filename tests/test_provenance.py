from __future__ import annotations

from pathlib import Path

from betlas.provenance import source_tree_state

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_source_tree_state_uses_tracked_files_not_ignored_private_scripts(monkeypatch) -> None:
    monkeypatch.chdir(REPO_ROOT)

    state = source_tree_state(paths=("scripts",))

    assert state["basis"] == "git_ls_files"
    assert all(not str(path).startswith("scripts/internal/") for path in state["paths"])
    assert all(not str(path).startswith("scripts/archive") for path in state["paths"])
