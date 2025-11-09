"""Simple semantic index based on task keyword TF-IDF embeddings."""

from __future__ import annotations

import json
import logging
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from .models import PlannerAction

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> List[str]:
    return [token for token in text.lower().split() if token.isalnum() and len(token) > 2]


@dataclass
class IndexedPlan:
    actions: List[PlannerAction]
    similarity: float
    source_task: str


class SemanticIndex:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def add(self, app: Optional[str], task: str, actions: Sequence[PlannerAction]) -> None:
        tokens = _tokenize(task)
        if not tokens:
            return
        payload = {
            "app": (app or "generic").lower(),
            "task": task,
            "tokens": tokens,
            "actions": [action.model_dump() for action in actions if action.action not in {"capture"}],
        }
        if not payload["actions"]:
            return
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload) + "\n")
        logger.info("Semantic index stored plan for %s", task)

    def retrieve(self, app: Optional[str], task: str, *, min_similarity: float = 0.35) -> Optional[IndexedPlan]:
        if not self.path.exists():
            return None
        target_tokens = _tokenize(task)
        if not target_tokens:
            return None
        target_counter = Counter(target_tokens)
        best: Optional[IndexedPlan] = None
        with self.path.open("r", encoding="utf-8") as fh:
            for raw_line in fh:
                try:
                    record = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue
                if record.get("app") != (app or "generic").lower():
                    continue
                tokens = record.get("tokens") or []
                if not tokens:
                    continue
                counter = Counter(tokens)
                similarity = _cosine_similarity(target_counter, counter)
                if similarity < min_similarity:
                    continue
                try:
                    actions = [PlannerAction.model_validate(payload) for payload in record.get("actions", [])]
                except Exception:
                    continue
                if not actions:
                    continue
                if best is None or similarity > best.similarity:
                    best = IndexedPlan(actions=actions, similarity=similarity, source_task=record.get("task", ""))
        return best


def _cosine_similarity(lhs: Counter[str], rhs: Counter[str]) -> float:
    dot = sum(lhs[token] * rhs[token] for token in lhs.keys() & rhs.keys())
    lhs_norm = math.sqrt(sum(value * value for value in lhs.values()))
    rhs_norm = math.sqrt(sum(value * value for value in rhs.values()))
    if lhs_norm == 0 or rhs_norm == 0:
        return 0.0
    return dot / (lhs_norm * rhs_norm)

