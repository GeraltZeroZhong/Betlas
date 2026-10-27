from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


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
