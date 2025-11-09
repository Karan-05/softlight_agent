"""Hierarchical planner that replays scripts, uses the LLM, then falls back to heuristics."""

from __future__ import annotations

import logging
from collections import deque
from typing import Callable, Deque, Dict, List, Optional, Sequence, Set, Tuple

from .heuristic_planner import HeuristicPlanner
from .intent import parse_intent
from .modal_submit import select_modal_submit_candidate
from .llm_planner import InvalidPlannerActionError, LLMPlanner
from .models import PlannerAction, UIElement, UISnapshot
from .planning_validation import (
    SnapshotAffordances,
    build_snapshot_affordances,
    validate_action_against_snapshot_and_intent,
)
from .self_heal import selector_is_unreliable

logger = logging.getLogger(__name__)


class HierarchicalPlanner:
    """Coordinator that replays queued scripts, tries LLM intent, then falls back to heuristics."""

    def __init__(
        self,
        llm_planner: Optional[LLMPlanner],
        heuristic_planner: HeuristicPlanner,
    ) -> None:
        self.llm_planner = llm_planner
        self.heuristic_planner = heuristic_planner
        self._pending_actions: Deque[PlannerAction] = deque()
        self._pending_source: Optional[str] = None
        self._event_logger: Optional[Callable[[Dict[str, object]], None]] = None
        self._invalidated_sources: Set[str] = set()
        self._pending_modal_submit_owner: Optional[str] = None

    def set_event_logger(self, callback: Optional[Callable[[Dict[str, object]], None]]) -> None:
        """Attach a telemetry callback for planner-specific events."""
        self._event_logger = callback

    def prime_actions(self, actions: Sequence[PlannerAction], source: str = "memory") -> None:
        """Inject a predetermined script that should execute before new planning."""
        filtered: List[PlannerAction] = []
        for action in actions:
            if action.selector and selector_is_unreliable(action.selector):
                logger.debug("Skipping unreliable selector while priming %s: %s", source, action.selector)
                continue
            filtered.append(action)
        if not filtered:
            return
        self._pending_actions = deque(filtered)
        self._pending_source = source

    def clear_pending(self) -> None:
        """Drop any queued actions, forcing a fresh plan next turn."""
        if self._pending_actions:
            logger.debug("Clearing %s queued actions from source %s", len(self._pending_actions), self._pending_source)
        self._pending_actions.clear()
        self._pending_source = None

    async def plan_next(
        self,
        task: str,
        snapshot: UISnapshot,
        history: Sequence[Dict[str, object]],
        *,
        avoid_selectors: Sequence[str] | None = None,
    ) -> Tuple[PlannerAction, str]:
        intent_info = parse_intent(task)
        affordances = build_snapshot_affordances(snapshot)

        forced_action = self._maybe_force_modal_submit(snapshot, history, intent_info)
        if forced_action:
            self.clear_pending()
            return forced_action, "modal-submit"

        if self._pending_actions:
            queued = self._dequeue_valid_action(task, snapshot, intent_info, affordances, history)
            if queued:
                action, source = queued
                self._update_modal_fill_hint(snapshot, intent_info, action)
                return action, source
        avoid = {selector for selector in (avoid_selectors or []) if selector}
        if self.llm_planner:
            llm_action = await self._try_llm(task, snapshot, history, avoid, intent_info, affordances)
            if llm_action:
                action, source = llm_action
                self._update_modal_fill_hint(snapshot, intent_info, action)
                return action, source

        # Fallback to heuristic planner
        action = self.heuristic_planner.plan_next(task, snapshot, history, avoid_selectors=list(avoid))
        self._update_modal_fill_hint(snapshot, intent_info, action)
        return action, "heuristic"

    def _queue_actions(self, actions: Sequence[PlannerAction], source: str) -> None:
        filtered: List[PlannerAction] = []
        for action in actions:
            if action.selector and any(action.selector == existing.selector for existing in filtered):
                continue
            if action.selector and selector_is_unreliable(action.selector):
                logger.debug("Skipping unreliable selector from %s: %s", source, action.selector)
                continue
            if not self._action_is_valid(action):
                logger.debug("Skipping invalid action from %s: %s", source, action)
                continue
            filtered.append(action)
        if filtered:
            self._pending_actions.extend(filtered)
            self._pending_source = source

    def _log_planner_reject(self, source: str, action: PlannerAction, reason: Optional[str]) -> None:
        logger.debug("Rejecting %s action %s: %s", source, action, reason)
        if self._event_logger:
            payload: Dict[str, object] = {
                "event": "planner_reject",
                "source": source,
                "reason": reason or "unspecified",
                "action": action.model_dump(),
            }
            self._event_logger(payload)

    def _filter_llm_actions(
        self,
        actions: Sequence[PlannerAction],
        *,
        task: str,
        snapshot: UISnapshot,
        intent_data: Dict[str, object],
        affordances: SnapshotAffordances,
        source: str,
    ) -> List[PlannerAction]:
        validated: List[PlannerAction] = []
        for action in actions:
            valid, reason = validate_action_against_snapshot_and_intent(
                task,
                snapshot,
                action,
                affordances=affordances,
                intent_data=intent_data,
            )
            if not valid:
                self._log_planner_reject(source, action, reason)
                continue
            validated.append(action)
        return validated

    async def _try_llm(
        self,
        task: str,
        snapshot: UISnapshot,
        history: Sequence[Dict[str, object]],
        avoid: set[str],
        intent_info: Dict[str, object],
        affordances: SnapshotAffordances,
    ) -> Optional[Tuple[PlannerAction, str]]:
        assert self.llm_planner is not None

        try:
            script = await self.llm_planner.plan_script(task, snapshot, list(history))
        except InvalidPlannerActionError as exc:
            logger.debug("LLM script discarded: %s", exc)
            script = []
        except Exception as exc:  # noqa: BLE001
            logger.debug("LLM script planning failed: %s", exc)
            script = []

        script = [step for step in script if not step.selector or step.selector not in avoid]
        if script:
            script = self._filter_llm_actions(
                script,
                task=task,
                snapshot=snapshot,
                intent_data=intent_info,
                affordances=affordances,
                source="llm-script",
            )
            self._queue_actions(script, source="llm-script")
            queued = self._dequeue_valid_action(task, snapshot, intent_info, affordances, history)
            if queued:
                return queued

        try:
            action = await self.llm_planner.plan_next(task, snapshot, list(history))
        except InvalidPlannerActionError as exc:
            logger.debug("LLM action discarded: %s", exc)
            return None
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM planner failed: %s", exc)
            return None

        if not self._action_is_valid(action):
            logger.debug("LLM produced invalid action; ignoring.")
            return None
        if action.selector and action.selector in avoid:
            logger.debug("LLM selected previously failed selector %s; ignoring.", action.selector)
            return None
        if action.selector and selector_is_unreliable(action.selector):
            logger.debug("LLM selected unreliable selector %s; ignoring.", action.selector)
            return None
        valid, reason = validate_action_against_snapshot_and_intent(
            task,
            snapshot,
            action,
            affordances=affordances,
            intent_data=intent_info,
        )
        if not valid:
            self._log_planner_reject("llm", action, reason)
            return None

        return action, "llm"

    def consume_invalidated_sources(self) -> Set[str]:
        sources = set(self._invalidated_sources)
        self._invalidated_sources.clear()
        return sources

    def _maybe_force_modal_submit(
        self,
        snapshot: UISnapshot,
        history: Sequence[Dict[str, object]],
        intent_info: Dict[str, object],
    ) -> Optional[PlannerAction]:
        target_nouns: Sequence[str] = tuple(intent_info.get("nouns") or ())
        if not target_nouns:
            self._pending_modal_submit_owner = None
            return None
        if not snapshot.modals:
            self._pending_modal_submit_owner = None
            return None
        explicit_title = (intent_info.get("explicit_title") or "").strip()
        modal_selector = self._pending_modal_submit_owner or self._recent_modal_name_fill(
            snapshot, history, target_nouns, explicit_title
        )
        if not modal_selector:
            self._pending_modal_submit_owner = None
            return None
        destructive_modal = _find_destructive_modal(snapshot)
        if destructive_modal and destructive_modal != modal_selector:
            exit_button = _find_modal_exit_button(snapshot, destructive_modal)
            if exit_button and exit_button.selector:
                logger.debug("Destructive modal detected; exiting via %s", exit_button.selector)
                return PlannerAction(
                    action="click",
                    selector=exit_button.selector,
                    reason="Dismiss destructive confirmation before submitting",
                    capture_name="dismiss-destructive-modal",
                )
        submit_element, choice = select_modal_submit_candidate(
            snapshot,
            target_nouns=target_nouns,
            owner_selector=modal_selector,
        )
        if not submit_element or not submit_element.selector:
            self._pending_modal_submit_owner = None
            return None
        selector = submit_element.selector
        if any(
            entry.get("action") == "click" and entry.get("selector") == selector and entry.get("success")
            for entry in history[-3:]
        ):
            self._pending_modal_submit_owner = None
            return None
        descriptor = choice.descriptor if choice else "primary button"
        score = choice.score if choice else None
        reason = f"Submit the creation dialog via '{descriptor}'"
        if score is not None:
            reason = f"{reason} (score {score})"
        logger.debug(
            "Forcing modal submit selector=%s descriptor=%r score=%s owner=%s",
            selector,
            descriptor,
            score,
            modal_selector,
        )
        if self._event_logger:
            self._event_logger(
                {
                    "event": "planner_force_submit",
                    "selector": selector,
                    "modal": modal_selector,
                    "descriptor": descriptor,
                    "score": score,
                }
            )
        self._pending_modal_submit_owner = None
        return PlannerAction(action="click", selector=selector, reason=reason, capture_name="modal-submit")

    def _recent_modal_name_fill(
        self,
        snapshot: UISnapshot,
        history: Sequence[Dict[str, object]],
        target_nouns: Sequence[str],
        explicit_title: str,
    ) -> Optional[str]:
        explicit_normalised = explicit_title.lower()
        for entry in reversed(history):
            if not isinstance(entry, dict):
                continue
            action = str(entry.get("action") or "").lower()
            if action == "click" and entry.get("success"):
                if _history_entry_is_neutral(entry):
                    continue
                break
            if action != "fill" or not entry.get("success"):
                continue
            selector = (entry.get("selector") or "").strip()
            if not selector:
                continue
            element = _lookup_input(snapshot, selector)
            if not element:
                continue
            owner_selector = _owner_selector_for_element(element, snapshot)
            if not owner_selector:
                continue
            if not _input_looks_like_name_field(element, target_nouns):
                continue
            value = (entry.get("value") or "").strip()
            if not value:
                continue
            if explicit_normalised and explicit_normalised not in value.lower():
                continue
            return owner_selector
        return None


    def _action_is_valid(self, action: PlannerAction) -> bool:
        if action.action == "press" and not action.value:
            return False
        if action.action in {"click", "fill", "wait_for"} and not (action.selector or "").strip():
            return False
        if action.action == "fill":
            if action.value is None:
                return False
            if isinstance(action.value, str) and not action.value.strip():
                return False
        if action.action == "goto" and not action.url:
            return False
        return True

    def _dequeue_valid_action(
        self,
        task: str,
        snapshot: UISnapshot,
        intent_info: Dict[str, object],
        affordances: SnapshotAffordances,
        history: Sequence[Dict[str, object]],
    ) -> Optional[Tuple[PlannerAction, str]]:
        if not self._pending_actions:
            return None
        source = self._pending_source or "script"
        invalidated = False
        while self._pending_actions:
            action = self._pending_actions.popleft()
            if not self._action_is_valid(action):
                self._log_planner_reject(source, action, "action missing required fields")
                invalidated = True
                continue
            if action.selector and selector_is_unreliable(action.selector):
                self._log_planner_reject(source, action, "selector marked unreliable")
                invalidated = True
                continue
            if _fill_already_satisfied(history, action):
                self._log_planner_reject(source, action, "fill already satisfied earlier this run")
                invalidated = True
                continue
            valid, reason = validate_action_against_snapshot_and_intent(
                task,
                snapshot,
                action,
                affordances=affordances,
                intent_data=intent_info,
            )
            if not valid:
                self._log_planner_reject(source, action, reason)
                invalidated = True
                continue
            self._update_modal_fill_hint(snapshot, intent_info, action)
            return action, source
        self._pending_source = None
        if invalidated and source:
            self._invalidated_sources.add(source)
        return None

    def _update_modal_fill_hint(
        self,
        snapshot: UISnapshot,
        intent_info: Dict[str, object],
        action: PlannerAction,
    ) -> None:
        if action.action == "click":
            self._pending_modal_submit_owner = None
            return
        if action.action != "fill":
            return
        selector = (action.selector or "").strip()
        if not selector:
            return
        element = _lookup_input(snapshot, selector)
        if not element:
            return
        target_nouns: Sequence[str] = tuple(intent_info.get("nouns") or ())
        if not target_nouns:
            return
        if not _input_looks_like_name_field(element, target_nouns):
            return
        owner = _owner_selector_for_element(element, snapshot)
        if owner:
            self._pending_modal_submit_owner = owner


def _lookup_input(snapshot: UISnapshot, selector: str) -> Optional[UIElement]:
    for element in snapshot.inputs:
        if element.selector == selector:
            return element
    return None


def _input_looks_like_name_field(element: UIElement, target_nouns: Sequence[str]) -> bool:
    metadata = element.metadata or {}
    descriptor_parts = [
        element.text,
        metadata.get("labelText"),
        metadata.get("placeholder"),
        metadata.get("ariaLabel"),
        metadata.get("ownerDialogTitle"),
    ]
    descriptor = _collapse_text(descriptor_parts)
    if not descriptor:
        return False
    if "summary" in descriptor or "description" in descriptor:
        return False
    if any(token in descriptor for token in ("name", "title")):
        return True
    return any(noun and noun in descriptor for noun in target_nouns)


def _owner_selector_for_element(element: UIElement, snapshot: UISnapshot) -> Optional[str]:
    metadata = element.metadata or {}
    owner_dialog = metadata.get("ownerDialog") or {}
    selector = (owner_dialog.get("selector") or "").strip()
    if selector:
        return selector
    title_hint = _collapse_text([(metadata.get("ownerDialogTitle") or "")])
    if title_hint:
        for modal in reversed(snapshot.modals):
            modal_selector = (modal.selector or "").strip()
            modal_texts = [
                modal.text,
                (modal.metadata or {}).get("ariaLabel"),
                (modal.metadata or {}).get("labelText"),
            ]
            for text in modal_texts:
                value = _collapse_text([text])
                if value and (value in title_hint or title_hint in value):
                    if modal_selector:
                        return modal_selector
    if snapshot.modals:
        fallback = (snapshot.modals[-1].selector or "").strip()
        if fallback:
            return fallback
    return None


def _history_entry_is_neutral(entry: Dict[str, object]) -> bool:
    haystack = " ".join(
        str(entry.get(field) or "")
        for field in ("selector", "reason", "capture_name")
    ).lower()
    neutral_tokens = ("dismiss", "cancel", "close", "discard", "keep editing", "stay")
    return any(token in haystack for token in neutral_tokens)


def _fill_already_satisfied(history: Sequence[Dict[str, object]], action: PlannerAction) -> bool:
    if action.action != "fill":
        return False
    selector = (action.selector or "").strip()
    value = (action.value or "").strip()
    if not selector or not value:
        return False
    for entry in reversed(history):
        if not isinstance(entry, dict):
            continue
        entry_action = str(entry.get("action") or "").lower()
        if entry_action == "fill" and entry.get("selector") == selector and entry.get("success"):
            entry_value = str(entry.get("value") or "").strip()
            if entry_value == value:
                return True
        if entry_action == "click" and entry.get("success"):
            break
    return False


def _collapse_text(parts: Sequence[object]) -> str:
    chunks: List[str] = []
    for part in parts:
        if part is None:
            continue
        text = str(part).strip()
        if text:
            chunks.append(text)
    if not chunks:
        return ""
    return " ".join(chunks).strip().lower()


def _find_destructive_modal(snapshot: UISnapshot) -> Optional[str]:
    tokens = ("discard", "delete", "remove", "are you sure", "confirmation")
    for modal in snapshot.modals:
        text = _collapse_text(
            [
                modal.text,
                (modal.metadata or {}).get("ariaLabel"),
                (modal.metadata or {}).get("labelText"),
            ]
        )
        if text and any(token in text for token in tokens):
            selector = (modal.selector or "").strip()
            if selector:
                return selector
    return None


def _find_modal_exit_button(snapshot: UISnapshot, modal_selector: str) -> Optional[UIElement]:
    safe_tokens = ("cancel", "keep", "stay", "close", "never mind", "go back")
    for element in snapshot.clickables:
        if not element.selector:
            continue
        metadata = element.metadata or {}
        owner = ((metadata.get("ownerDialog") or {}).get("selector") or "").strip()
        if owner != modal_selector:
            continue
        descriptor = _collapse_text(
            [
                element.text,
                metadata.get("ariaLabel"),
                metadata.get("labelText"),
            ]
        )
        if not descriptor:
            continue
        if any(token in descriptor for token in safe_tokens):
            return element
    return None


__all__ = ["HierarchicalPlanner"]
