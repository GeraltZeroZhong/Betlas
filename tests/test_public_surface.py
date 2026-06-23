from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from betlas import cli
from betlas.grammars import list_grammars

REPO_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_SCRIPT_ROOTS = [
    REPO_ROOT / "scripts" / "external_baselines",
    REPO_ROOT / "scripts" / "reproducibility",
]
PUBLIC_SCRIPT_SUFFIXES = {".py", ".sh", ".cs", ".csproj"}
PUBLIC_SOURCE_SUFFIXES = PUBLIC_SCRIPT_SUFFIXES | {".yaml", ".yml", ".json"}
BANNED_PATTERNS = [
    re.compile(pattern)
    for pattern in [
        r"cz_",
        r"\bLegacy\b",
        r"\blegacy\b",
        r"Cooper-Zoo",
        r"cooper_zoo",
        r"\b[Cc]ooper\b",
        r"\bmanuscript\b",
        r"(?<![A-Za-z])Fig\.?(?![A-Za-z])",
        r"\bfigures?\b",
        r"\bpublication\b",
        r"\binternal\b",
    ]
]
SOURCE_BANNED_PATTERNS = [
    re.compile(pattern)
    for pattern in [
        r"\bLegacy\b",
        r"\blegacy\b",
        r"LEGACY_",
        r"\b[Aa]rchive(?:d|s)?\b",
        r"old-data",
        r"archived_20[0-9]{6}",
        r"Cooper-Zoo",
        r"cooper_zoo",
        r"\b[Cc]ooper\b",
        r"\bmanuscript\b",
        r"(?<![A-Za-z])Fig\.?(?![A-Za-z])",
        r"\bfigures?\b",
        r"\bpublication\b",
        r"\binternal\b",
    ]
]


def _public_docs() -> list[Path]:
    paths = [
        REPO_ROOT / "README.md",
        REPO_ROOT / "scripts" / "public_script_inventory.yaml",
    ]
    paths.extend(sorted((REPO_ROOT / "assets").rglob("*.md")))
    paths.extend(sorted((REPO_ROOT / "assets").rglob("*.yaml")))
    for root in PUBLIC_SCRIPT_ROOTS:
        paths.extend(sorted(root.rglob("*.md")))
    return [path for path in paths if path.exists()]


def _public_source_files() -> list[Path]:
    paths: list[Path] = []
    paths.extend(
        path
        for path in sorted((REPO_ROOT / "src" / "betlas").rglob("*"))
        if path.is_file() and path.suffix in {".py", ".yaml", ".yml"}
    )
    for root in PUBLIC_SCRIPT_ROOTS:
        paths.extend(
            path
            for path in sorted(root.rglob("*"))
            if path.is_file() and path.suffix in PUBLIC_SOURCE_SUFFIXES
        )
    return paths


def _assert_clean_public_text(name: str, text: str) -> None:
    for pattern in BANNED_PATTERNS:
        assert pattern.search(text) is None, f"{name} contains banned public-surface term {pattern.pattern!r}"


def _assert_clean_source_text(name: str, text: str) -> None:
    for pattern in SOURCE_BANNED_PATTERNS:
        assert pattern.search(text) is None, f"{name} contains boundary-leaking source term {pattern.pattern!r}"


def _inventory_paths() -> set[str]:
    inventory = yaml.safe_load((REPO_ROOT / "scripts" / "public_script_inventory.yaml").read_text(encoding="utf-8"))
    assert isinstance(inventory, dict)
    entries = inventory.get("scripts", [])
    assert isinstance(entries, list)
    assert all(entry.get("stable_python_api") is False for entry in entries)
    assert all(entry.get("pypi_api") is False for entry in entries)
    return {str(entry["path"]) for entry in entries}


def _inventory_entries() -> list[dict[str, object]]:
    inventory = yaml.safe_load((REPO_ROOT / "scripts" / "public_script_inventory.yaml").read_text(encoding="utf-8"))
    entries = inventory.get("scripts", [])
    assert isinstance(entries, list)
    return list(entries)


def _public_script_paths() -> set[str]:
    paths = {str((REPO_ROOT / "scripts" / "run_full_pipeline.py").relative_to(REPO_ROOT))}
    for root in PUBLIC_SCRIPT_ROOTS:
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix not in PUBLIC_SCRIPT_SUFFIXES:
                continue
            if path.name == "__init__.py":
                continue
            paths.add(str(path.relative_to(REPO_ROOT)))
    return paths


def _tracked_public_script_paths() -> set[str]:
    result = subprocess.run(
        [
            "git",
            "ls-files",
            "--",
            "scripts/run_full_pipeline.py",
            "scripts/external_baselines",
            "scripts/reproducibility",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    paths: set[str] = set()
    for rel_path in result.stdout.splitlines():
        path = Path(rel_path)
        if path.suffix not in PUBLIC_SCRIPT_SUFFIXES or path.name == "__init__.py":
            continue
        paths.add(rel_path)
    return paths


def _untracked_public_script_paths() -> set[str]:
    result = subprocess.run(
        [
            "git",
            "ls-files",
            "--others",
            "--exclude-standard",
            "--",
            "scripts/run_full_pipeline.py",
            "scripts/external_baselines",
            "scripts/reproducibility",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    paths: set[str] = set()
    for rel_path in result.stdout.splitlines():
        path = Path(rel_path)
        if path.suffix in PUBLIC_SCRIPT_SUFFIXES and path.name != "__init__.py":
            paths.add(rel_path)
    return paths


def _ignored_public_script_paths() -> set[str]:
    result = subprocess.run(
        [
            "git",
            "ls-files",
            "--others",
            "--ignored",
            "--exclude-standard",
            "--",
            "scripts/run_full_pipeline.py",
            "scripts/external_baselines",
            "scripts/reproducibility",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    allowed_parts = {"__pycache__", ".conda", "downloads", "runs", "tools", ".pytest_cache", ".ruff_cache"}
    paths: set[str] = set()
    for rel_path in result.stdout.splitlines():
        parts = set(Path(rel_path).parts)
        if parts & allowed_parts:
            continue
        paths.add(rel_path)
    return paths


def test_public_docs_and_inventory_use_betlas_public_terms_only() -> None:
    for path in _public_docs():
        _assert_clean_public_text(str(path.relative_to(REPO_ROOT)), path.read_text(encoding="utf-8"))


def test_public_source_uses_release_boundary_terms_only() -> None:
    for path in _public_source_files():
        _assert_clean_source_text(str(path.relative_to(REPO_ROOT)), path.read_text(encoding="utf-8"))


def test_all_tracked_public_scripts_are_in_inventory() -> None:
    assert _public_script_paths() == _inventory_paths()


def test_public_script_inventory_matches_git_index() -> None:
    assert _untracked_public_script_paths() == set()
    assert _ignored_public_script_paths() == set()
    assert _tracked_public_script_paths() == _inventory_paths()


def test_public_shell_wrappers_provide_help() -> None:
    for rel_path in sorted(_inventory_paths()):
        path = REPO_ROOT / rel_path
        if path.suffix != ".sh":
            continue
        result = subprocess.run(
            ["bash", str(path), "--help"],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert "Usage:" in result.stdout
        _assert_clean_public_text(rel_path, result.stdout + result.stderr)


def test_external_baseline_wrappers_fail_fast_without_required_inputs() -> None:
    wrappers = [
        "scripts/external_baselines/beta_barrel_staves/stave_count_wrappers/scripts/run_betaware.sh",
        "scripts/external_baselines/beta_barrel_staves/stave_count_wrappers/scripts/run_juchmme_pred_tmbb2.sh",
        "scripts/external_baselines/beta_barrel_staves/stave_count_wrappers/scripts/run_polarbearal.sh",
        "scripts/external_baselines/beta_barrel_staves/stave_count_wrappers/scripts/run_proftmb.sh",
        "scripts/external_baselines/beta_barrel_staves/stave_count_wrappers/scripts/run_tmbed.sh",
    ]
    for rel_path in wrappers:
        result = subprocess.run(
            ["bash", str(REPO_ROOT / rel_path)],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 2
        assert result.stdout == ""
        assert result.stderr.startswith("Error: required arguments:"), rel_path


def test_public_cli_scripts_provide_help() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT / 'src'}:{REPO_ROOT}"
    for entry in _inventory_entries():
        rel_path = str(entry["path"])
        entry_type = str(entry.get("entry", ""))
        if entry_type not in {"cli", "module_cli"}:
            continue
        path = REPO_ROOT / rel_path
        if path.suffix != ".py":
            continue
        result = subprocess.run(
            [sys.executable, str(path), "--help"],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
            timeout=20,
        )
        assert result.returncode == 0, result.stderr
        assert "usage:" in (result.stdout + result.stderr).lower()
        _assert_clean_public_text(rel_path, result.stdout + result.stderr)


def test_public_companion_scripts_do_not_top_level_import_optional_ml_dependencies() -> None:
    optional_imports = re.compile(r"(?m)^(?:from catboost\b|import catboost\b)")
    for rel_path in sorted(_public_script_paths()):
        path = REPO_ROOT / rel_path
        if path.suffix != ".py":
            continue
        assert optional_imports.search(path.read_text(encoding="utf-8")) is None, (
            f"{rel_path} imports optional ML dependencies at module load time"
        )


def test_public_cli_help_uses_release_surface_terms(capsys) -> None:
    help_commands = [
        ["--help"],
        ["grammar", "--help"],
        ["slice", "--help"],
        ["assets", "--help"],
        ["assets", "download", "--help"],
        ["assets", "check-esmc", "--help"],
        ["readout", "beta-barrel-detection", "--help"],
        ["readout", "beta-barrel-staves", "--help"],
        ["readout", "topology-diagnostics", "--help"],
        ["readout", "fold-continuous-scores", "--help"],
        ["readout", "topology-ambiguity", "--help"],
        ["readout", "mixed-topology", "--help"],
    ]
    for command in help_commands:
        with pytest.raises(SystemExit) as exc:
            cli.main(command)
        assert exc.value.code == 0
        _assert_clean_public_text("betlas " + " ".join(command), capsys.readouterr().out)


def test_topology_alias_help_is_mode_specific(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["readout", "fold-continuous-scores", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "continuous fold-organization scores" in out
    assert "--mode ambiguity" not in out

    with pytest.raises(SystemExit) as exc:
        cli.main(["readout", "fold-continuous-scores", "--mode", "ambiguity"])
    assert exc.value.code == 2
    assert "fixed topology mode" in capsys.readouterr().err


def test_public_cli_user_errors_do_not_print_tracebacks(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["grammar", "describe", "does-not-exist"])
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert "Error:" in captured.err
    assert "Traceback" not in captured.err

    with pytest.raises(SystemExit) as exc:
        cli.main(["readout", "does-not-exist"])
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert "Error:" in captured.err
    assert "Traceback" not in captured.err


def test_grammar_descriptions_are_complete_and_public(capsys) -> None:
    for spec in list_grammars():
        assert spec.math_summary
        assert spec.inputs
        assert spec.outputs
        cli.main(["grammar", "describe", spec.name])
        _assert_clean_public_text(f"grammar {spec.name}", capsys.readouterr().out)
