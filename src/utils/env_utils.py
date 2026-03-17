"""Environment dependency helpers."""

from __future__ import annotations

import importlib.util


def ensure_sentencepiece_installed() -> None:
    """Raise a clear error when `sentencepiece` is unavailable.

    SD3/T5 tokenizers rely on sentencepiece in many environments. We check this
    explicitly so failures happen early with an actionable message.
    """

    if importlib.util.find_spec("sentencepiece") is None:
        raise ModuleNotFoundError(
            "Missing optional dependency `sentencepiece`. "
            "Install it with `pip install sentencepiece` and retry."
        )
