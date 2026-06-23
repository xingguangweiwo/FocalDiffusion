"""FocalStackGeneration training utilities."""

from .losses import FocalStackGenerationLoss, SelectiveViolationLoss, VerificationTraceLoss
from .optimizers import get_optimizer

__all__ = [
    "FocalStackGenerationTrainer",
    "FocalStackGenerationLoss",
    "VerificationTraceLoss",
    "SelectiveViolationLoss",
    "get_optimizer",
]


def __getattr__(name: str):
    """Lazily import trainer dependencies such as accelerate only when needed."""
    if name == "FocalStackGenerationTrainer":
        from .trainer import FocalStackGenerationTrainer

        return FocalStackGenerationTrainer
    raise AttributeError(name)
