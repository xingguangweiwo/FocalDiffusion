"""Compatibility shim for the renamed task-output decoder.

New code should import :class:`TaskOutputDecoder` from
``src.models.task_output_decoder``.
"""

from .task_output_decoder import TaskOutputDecoder

DualOutputDecoder = TaskOutputDecoder

__all__ = ["TaskOutputDecoder", "DualOutputDecoder"]
