"""Lightweight retrieval store for prior successful plans."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from .models import PlannerAction

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> List[str]:
    return [token for token in text.lower().split() if token.isalnum() and len(token) > 2]


def _normalize_actions(actions: Sequence[PlannerAction]) -> List[dict]:
    cleaned: List[dict] = []
    for action in actions:
        if action.action in {"capture"}:
            continue
        cleaned.append(action.model_dump())
    return cleaned


@dataclass
class RetrievedPlan:
    actions: List[PlannerAction]
    similarity: float
    source_task: str


class ExperienceStore:
    """Store successful plans and retrieve similar ones for new tasks."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def add(self, app: Optional[str], task: str, actions: Sequence[PlannerAction]) -> None:
        normalized = _normalize_actions(actions)
        if not normalized:
            logger.debug("Skipping experience for %s; no actionable steps.", task)
            return
        record = {
            "app": (app or "generic").lower(),
            "task": task,
            "keywords": _tokenize(task),
            "actions": normalized,
        }
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        logger.info("Stored experience for task '%s' (%s actions)", task, len(normalized))

    def retrieve(self, app: Optional[str], task: str, *, min_similarity: float = 0.4) -> Optional[RetrievedPlan]:
        if not self.path.exists():
            return None
        target_tokens = set(_tokenize(task))
        best: Optional[RetrievedPlan] = None
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("app") != (app or "generic").lower():
                    continue
                candidate_tokens = set(record.get("keywords") or [])
                if not candidate_tokens or not target_tokens:
                    continue
                intersection = len(target_tokens & candidate_tokens)
                union = len(target_tokens | candidate_tokens)
                similarity = intersection / union if union else 0.0
                if similarity < min_similarity:
                    continue
                try:
                    actions = [PlannerAction.model_validate(payload) for payload in record.get("actions", [])]
                except Exception:
                    continue
                if not actions:
                    continue
                if best is None or similarity > best.similarity:
                    best = RetrievedPlan(actions=actions, similarity=similarity, source_task=record.get("task", ""))
        return best

