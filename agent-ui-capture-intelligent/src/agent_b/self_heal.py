"""Self-healing helpers for resilient UI actions."""

from __future__ import annotations

from pathlib import Path

from .memory import MEMORY


def configure_memory(path: Path | None, threshold: int = 2) -> None:
    if path is not None:
        MEMORY.configure(path, threshold=threshold)


def mark_selector_failure(selector: str) -> None:
    """Record that a selector failed so we avoid repeating it endlessly."""
    MEMORY.mark_failure(selector)


def mark_selector_success(selector: str) -> None:
    """Reset failure memory for a selector after a successful interaction."""
    MEMORY.mark_success(selector)


def selector_is_unreliable(selector: str) -> bool:
    """Return True when a selector has failed too many times."""
    return MEMORY.is_unreliable(selector)
