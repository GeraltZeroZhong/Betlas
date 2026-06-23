from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

from betlas.readouts import list_readouts

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_BETLAS = REPO_ROOT / "src" / "betlas"


def test_non_core_modules_are_not_public_betlas_imports() -> None:
    forbidden_specs = [
        "betlas.publication",
        "betlas.external_baselines",
        "betlas.science_report",
        "betlas.workflows",
        "betlas.readouts.bfvd_scan",
        "betlas.readouts.beta_barrel_detection.evaluation",
        "betlas.readouts.beta_barrel_detection.betlas_readout",
        "betlas.readouts.beta_barrel_staves.algorithm_ablations",
        "betlas.readouts.beta_barrel_staves.ml_baseline",
        "betlas.readouts.beta_barrel_staves.publication",
        "betlas.readouts.beta_barrel_staves.external_baselines",
        "betlas.readouts.beta_barrel_staves.schema",
    ]

    for module_name in forbidden_specs:
        assert importlib.util.find_spec(module_name) is None


def test_public_readout_registry_excludes_bfvd_reproducibility_workflow() -> None:
    names = {spec.name for spec in list_readouts()}

    assert "bfvd-viral-beta-fold-scan" not in names
    assert {"beta-barrel-detection", "beta-barrel-staves", "topology-diagnostics"} <= names


def test_src_package_does_not_import_reproducibility_or_plotting_companions() -> None:
    source_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(SRC_BETLAS.rglob("*.py"))
    )

    assert "scripts." not in source_text
    for import_name in ["matplotlib", "networkx", "anndata", "h5py"]:
        assert import_name not in source_text


def test_import_betlas_does_not_set_thread_environment() -> None:
    env = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env.pop(name, None)
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import os, betlas; print(os.environ.get('OMP_NUM_THREADS'), os.environ.get('OPENBLAS_NUM_THREADS'), os.environ.get('MKL_NUM_THREADS'))",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    assert result.stdout.strip() == "None None None"
