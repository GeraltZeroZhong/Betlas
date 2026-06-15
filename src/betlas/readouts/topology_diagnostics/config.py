from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

CONFIG_RESOURCE = "conf/default.yaml"


def default_config_path() -> Path:
    return Path(str(files(__package__).joinpath(CONFIG_RESOURCE)))


def _plain(container: Any) -> dict[str, Any]:
    return dict(OmegaConf.to_container(container, resolve=True))


def load_config(config_path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    base = OmegaConf.load(default_config_path())
    if config_path is not None:
        base = OmegaConf.merge(base, OmegaConf.load(Path(config_path)))
    if overrides:
        base = OmegaConf.merge(base, OmegaConf.create(overrides))
    return _plain(base)


def cfg_get(config: dict[str, Any], dotted_key: str, default: Any) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current
