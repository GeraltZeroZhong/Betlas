from __future__ import annotations

import shutil
from importlib import resources
from pathlib import Path

EXAMPLE_NAMES = ("mini",)
_RESOURCE_PACKAGE = "betlas.example_data"


def list_examples() -> tuple[str, ...]:
    """Return packaged Betlas example identifiers."""

    return EXAMPLE_NAMES


def example_resource_name(name: str) -> str:
    """Return the packaged resource filename for one example id."""

    if name not in EXAMPLE_NAMES:
        available = ", ".join(EXAMPLE_NAMES)
        raise ValueError(f"unknown Betlas example {name!r}; available examples: {available}")
    return f"{name}.cif"


def copy_example(name: str, out_dir: str | Path = ".") -> Path:
    """Copy a packaged Betlas example file into a user-visible directory."""

    resource_name = example_resource_name(name)
    target_dir = Path(out_dir).expanduser()
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / resource_name
    source = resources.files(_RESOURCE_PACKAGE).joinpath(resource_name)
    with source.open("rb") as src, target.open("wb") as dst:
        shutil.copyfileobj(src, dst)
    return target


__all__ = ["copy_example", "example_resource_name", "list_examples"]
