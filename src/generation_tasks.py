"""Canonical task names for task-conditioned focal-stack image generation."""

GENERATION_TASKS = {
    "all_in_focus": 0,
    "depth": 1,
    "uncertainty": 2,
    "focal_evidence": 3,
    "refocus": 4,
}


def normalize_generation_task(generation_task: str) -> str:
    """Validate and return a canonical generation-task name."""
    if generation_task not in GENERATION_TASKS:
        valid = ", ".join(sorted(GENERATION_TASKS))
        raise ValueError(f"Unsupported generation_task {generation_task!r}; expected one of: {valid}")
    return generation_task
