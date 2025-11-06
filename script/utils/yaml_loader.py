"""Utility helpers for reading and writing YAML configuration files.

This module mirrors the functionality provided by :mod:`yaml` but falls back
on a very small pure Python parser when PyYAML is unavailable.  The parser is
sufficient for the configuration files used by the training scripts (nested
mappings, scalars and simple lists) which allows ``python -m script.train --dry-run``
to execute in lean environments that do not bundle PyYAML or Torch.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, List, Tuple

try:  # pragma: no cover - executed only when PyYAML is available
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - exercised when PyYAML is missing
    yaml = None  # type: ignore

Scalar = Any


def load_yaml_file(path: Path) -> dict:
    """Read and parse a YAML file."""

    with Path(path).open("r", encoding="utf-8") as handle:
        text = handle.read()

    if yaml is not None:
        return yaml.safe_load(text) or {}

    return _MiniYAMLParser(text).parse()


def dump_yaml_file(path: Path, data: dict) -> None:
    """Persist configuration data to disk in YAML (or JSON) format."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if yaml is not None:  # pragma: no cover
        with path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(data, handle, default_flow_style=False)
        return

    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


@dataclass
class _Line:
    indent: int
    content: str


class _MiniYAMLParser:
    """Minimal indentation-aware YAML subset parser."""

    def __init__(self, text: str) -> None:
        self.lines: List[_Line] = self._preprocess(text)

    @staticmethod
    def _preprocess(text: str) -> List[_Line]:
        processed: List[_Line] = []
        for raw_line in text.splitlines():
            if not raw_line.strip():
                continue
            stripped = raw_line.lstrip()
            if stripped.startswith("#"):
                continue
            indent = len(raw_line) - len(stripped)
            processed.append(_Line(indent=indent, content=stripped))
        return processed

    def parse(self) -> dict:
        value, next_index = self._parse_block(0, 0)
        if next_index != len(self.lines):
            raise ValueError("Unexpected trailing content in YAML file")
        if not isinstance(value, dict):
            raise ValueError("Top level YAML structure must be a mapping")
        return value

    def _parse_block(self, start: int, indent: int) -> Tuple[Any, int]:
        items: List[Any] = []
        mapping: dict[str, Any] = {}
        index = start
        mode: str | None = None

        while index < len(self.lines):
            line = self.lines[index]
            if line.indent < indent:
                break
            if line.indent > indent:
                raise ValueError(
                    f"Invalid indentation at line {index + 1}: '{line.content}'"
                )

            if line.content.startswith("- "):
                if mode is None:
                    mode = "list"
                elif mode != "list":
                    raise ValueError("Cannot mix list items with mapping entries")
                value_text = line.content[2:].strip()
                if value_text:
                    index += 1
                    if ": " in value_text or value_text.endswith(":"):
                        inline_key, inline_remainder = value_text.split(":", 1)
                        inline_mapping: dict[str, Any] = {
                            inline_key.strip(): self._parse_scalar(inline_remainder.strip())
                        }
                        if index < len(self.lines) and self.lines[index].indent > indent:
                            nested_value, index = self._parse_block(index, indent + 2)
                            if not isinstance(nested_value, dict):
                                raise ValueError(
                                    "List item with inline mapping must be followed by mapping entries"
                                )
                            inline_mapping.update(nested_value)
                        value = inline_mapping
                    else:
                        value = self._parse_scalar(value_text)
                else:
                    value, index = self._parse_block(index + 1, indent + 2)
                items.append(value)
                continue

            if mode is None:
                mode = "dict"
            elif mode != "dict":
                raise ValueError("Cannot mix mapping entries with list items")

            if ":" not in line.content:
                raise ValueError(
                    f"Expected ':' in mapping entry at line {index + 1}: {line.content}"
                )

            key, remainder = line.content.split(":", 1)
            key = key.strip()
            remainder = remainder.strip()

            if not key:
                raise ValueError(f"Empty key at line {index + 1}")

            if remainder:
                mapping[key] = self._parse_scalar(remainder)
                index += 1
            else:
                value, index = self._parse_block(index + 1, indent + 2)
                mapping[key] = value

        if mode == "list":
            return items, index
        return mapping, index

    def _parse_scalar(self, token: str) -> Any:
        token_lower = token.lower()
        if token_lower in {"null", "none", "~"}:
            return None
        if token_lower == "true":
            return True
        if token_lower == "false":
            return False

        try:
            return json.loads(token)
        except json.JSONDecodeError:
            pass

        try:
            if token.startswith("0") and token != "0" and not token.startswith("0."):
                raise ValueError
            return int(token)
        except ValueError:
            try:
                return float(token)
            except ValueError:
                return token


__all__ = ["load_yaml_file", "dump_yaml_file"]
