"""Progress bar management for the read-generation pipeline.

Provide a global inner-progress display that sub-tasks can register
with via the ``progress_task`` context manager.  Only ``cli.py``
mutates the global state through ``set_inner_progress``.
"""

from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Generator

from rich.progress import Progress

_inner_progress: Progress | None = None


def set_inner_progress(progress: Progress | None) -> None:
    """Set the global inner-progress instance."""
    global _inner_progress  # noqa: PLW0603
    _inner_progress = progress


def get_inner_progress() -> Progress | None:
    """Return the current inner-progress instance."""
    return _inner_progress


@contextmanager
def progress_task(
    total: int, description: str,
) -> Generator[object, None, None]:
    """Add a sub-task to the inner progress display.

    Yield a callable that advances the task by one step.
    """
    if _inner_progress is None:
        yield lambda: None
        return
    task_id = _inner_progress.add_task(description, total=total)
    try:
        yield lambda: _inner_progress.advance(task_id)
    finally:
        _inner_progress.remove_task(task_id)
