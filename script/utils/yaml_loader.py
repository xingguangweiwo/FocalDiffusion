"""Utility helpers for reading and writing YAML configuration files."""

from __future__ import annotations

from pathlib import Path

import yaml


def load_yaml_file(path: Path) -> dict:
    """Read and parse a YAML file using required PyYAML."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def dump_yaml_file(path: Path, data: dict) -> None:
    """Persist configuration data to disk as YAML using required PyYAML."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, default_flow_style=False, sort_keys=False)
