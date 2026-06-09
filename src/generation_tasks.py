"""Canonical task names for task-conditioned focal-stack image generation."""

GENERATION_TASKS = {
    "all_in_focus": 0,
    "depth": 1,
    "uncertainty": 2,
    "focal_evidence": 3,
    "refocus": 4,
}

TASK_ALIASES = {
    "aif": "all_in_focus",
}


def normalize_generation_task(generation_task: str) -> str:
    """Return the canonical generation-task name, accepting supported legacy aliases."""
    canonical = TASK_ALIASES.get(generation_task, generation_task)
    if canonical not in GENERATION_TASKS:
        valid = ", ".join(sorted([*GENERATION_TASKS, *TASK_ALIASES]))
        raise ValueError(f"Unsupported generation_task {generation_task!r}; expected one of: {valid}")
    return canonical
