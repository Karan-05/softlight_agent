"""Persistent storage for successful action plans."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .intent import parse_intent
from .models import PlannerAction, UISnapshot
from .planning_validation import build_snapshot_affordances, validate_action_against_snapshot_and_intent

logger = logging.getLogger(__name__)


@dataclass
class StoredPlan:
    actions: List[PlannerAction]
    source: str
    failures: int = 0


class ActionPlanMemory:
    """Stores and retrieves deterministic action plans keyed by (app, task)."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _path_for(self, app: Optional[str], task_slug: str) -> Path:
        app_folder = (app or "generic").lower()
        return self.root / app_folder / f"{task_slug}.json"

    def load(self, app: Optional[str], task_slug: str) -> Optional[StoredPlan]:
        path = self._path_for(app, task_slug)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            logger.debug("Plan memory file %s is invalid JSON: %s", path, exc)
            return None
        actions_raw: Iterable[dict] = payload.get("actions") or []
        actions: List[PlannerAction] = []
        for entry in actions_raw:
            try:
                actions.append(PlannerAction.model_validate(entry))
            except Exception as exc:  # noqa: BLE001
                logger.debug("Skipping invalid cached action in %s: %s", path, exc)
                continue
        if not actions:
            return None
        if not any(step.action not in {"goto", "capture", "done"} for step in actions):
            logger.debug("Cached plan for %s lacks interactive steps; ignoring.", path)
            return None
        failures = int(payload.get("failures") or 0)
        source = str(payload.get("source") or "memory")
        return StoredPlan(actions=actions, source=source, failures=failures)

    def save(
        self,
        app: Optional[str],
        task_slug: str,
        actions: Iterable[PlannerAction],
        *,
        source: str = "memory",
        snapshot: Optional[UISnapshot] = None,
        task: Optional[str] = None,
    ) -> None:
        path = self._path_for(app, task_slug)
        path.parent.mkdir(parents=True, exist_ok=True)
        materialised = [action for action in actions if action.action not in {"capture"}]
        serialisable = [action.model_dump() for action in materialised]
        if not serialisable:
            return
        if not any(entry.get("action") not in {"goto", "capture", "done"} for entry in serialisable):
            logger.debug("Skipping plan save for %s/%s because it contains no interactive actions.", app or "generic", task_slug)
            return
        if snapshot is not None and task:
            plan = StoredPlan(actions=materialised, source=source)
            validated, reason = self.validate_for_snapshot(task=task, snapshot=snapshot, plan=plan)
            if not validated:
                logger.info(
                    "Not saving plan for %s/%s because it failed validation: %s",
                    app or "generic",
                    task_slug,
                    reason or "invalid selectors",
                )
                return
        payload: Dict[str, object] = {
            "actions": serialisable,
            "source": source,
            "failures": 0,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("Stored deterministic plan for %s/%s with %s steps", app or "generic", task_slug, len(serialisable))

    def validate_for_snapshot(
        self,
        *,
        task: str,
        snapshot: UISnapshot,
        plan: StoredPlan,
    ) -> Tuple[Optional[List[PlannerAction]], Optional[str]]:
        intent_data = parse_intent(task)
        affordances = build_snapshot_affordances(snapshot)
        destructive_task = _task_requests_destruction(intent_data)
        validated: List[PlannerAction] = []
        for action in plan.actions:
            valid, reason = validate_action_against_snapshot_and_intent(
                task,
                snapshot,
                action,
                affordances=affordances,
                intent_data=intent_data,
            )
            if not valid:
                return None, reason or "action invalid for current snapshot"
            if _action_conflicts_with_intent(action, destructive_task):
                return None, "cached plan contains dismiss/destructive steps"
            validated.append(action)
        return validated, None

    def mark_failure(self, app: Optional[str], task_slug: str) -> None:
        path = self._path_for(app, task_slug)
        if not path.exists():
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            path.unlink(missing_ok=True)
            return
        failures = int(payload.get("failures") or 0) + 1
        payload["failures"] = failures
        if failures >= 3:
            logger.info("Pruning unreliable plan for %s/%s after %s failures", app or "generic", task_slug, failures)
            path.unlink(missing_ok=True)
            return
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _action_conflicts_with_intent(action: PlannerAction, destructive_task: bool) -> bool:
    if destructive_task:
        return False
    if action.action not in {"click", "press", "fill"}:
        return False
    haystack = " ".join(
        str(part).lower()
        for part in (
            action.selector or "",
            action.reason or "",
            action.capture_name or "",
            action.value or "",
        )
        if part
    )
    if not haystack:
        return False
    destructive_tokens = ("discard", "delete", "remove", "cancel", "close", "dismiss", "keep editing", "stay")
    return any(token in haystack for token in destructive_tokens)


def _task_requests_destruction(intent_data: Dict[str, object]) -> bool:
    verbs: Sequence[str] = tuple(
        str(verb).lower()
        for verb in (intent_data.get("verbs") or [])
        if isinstance(verb, str)
    )
    return any(verb in {"delete", "remove", "discard", "close"} for verb in verbs)
