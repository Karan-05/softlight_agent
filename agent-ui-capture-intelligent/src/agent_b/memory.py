"""Session memory for selectors and outcomes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict


class SelectorMemory:
    def __init__(self) -> None:
        self.failures: Dict[str, int] = {}
        self.successes: Dict[str, int] = {}
        self.threshold = 2
        self._path: Path | None = None

    def configure(self, path: Path, threshold: int = 2) -> None:
        self._path = path
        self.threshold = threshold
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                self.failures = data.get("failures", {})
                self.successes = data.get("successes", {})
            except Exception:
                self.failures = {}
                self.successes = {}

    def _persist(self) -> None:
        if not self._path:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"failures": self.failures, "successes": self.successes}
            self._path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception:
            pass

    def mark_failure(self, selector: str) -> None:
        if not selector:
            return
        self.failures[selector] = self.failures.get(selector, 0) + 1
        self._persist()

    def mark_success(self, selector: str) -> None:
        if not selector:
            return
        self.successes[selector] = self.successes.get(selector, 0) + 1
        if selector in self.failures:
            del self.failures[selector]
        self._persist()

    def is_unreliable(self, selector: str) -> bool:
        if not selector:
            return False
        return self.failures.get(selector, 0) >= self.threshold


MEMORY = SelectorMemory()
