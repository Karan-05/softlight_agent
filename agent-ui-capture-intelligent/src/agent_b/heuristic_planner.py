"""Simple fallback planner when the LLM is not available."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from .intent import (
    GENERIC_OBJECT_KEYWORDS,
    TaskIntent,
    determine_artifact_title,
    get_or_create_task_intent,
    parse_intent,
    text_has_conflicting_noun,
    text_mentions_target_noun,
)
from .models import PlannerAction, UIElement, UISnapshot
from .modal_submit import select_modal_submit_candidate
from .planning_validation import (
    SnapshotAffordances,
    build_snapshot_affordances,
    validate_action_against_snapshot_and_intent,
)

logger = logging.getLogger(__name__)

PREFERRED = [
    "new",
    "create",
    "add",
    "save",
    "submit",
    "start",
    "continue",
    "next",
    "apply",
    "filter",
    "settings",
    "edit",
    "confirm",
    "share",
    "search",
    "command",
    "palette",
    "find",
    "toggle",
    "enable",
    "disable",
    "capture",
    "done",
    "finish",
]
STOP_WORDS = {token for phrase in PREFERRED for token in phrase.split() if len(token) > 2}
STOP_WORDS.update({"linear", "notion"})

_DESTRUCTIVE_KEYWORDS = (
    "discard",
    "delete",
    "remove",
    "are you sure",
    "permanently",
    "abandon",
    "trash",
)
_SAFE_EXIT_KEYWORDS = ("cancel", "keep", "stay", "close", "never mind", "go back", "dismiss")


@dataclass
class _IntentHints:
    create: bool = False
    filter: bool = False
    toggle: bool = False
    search: bool = False
    capture_requested: bool = False
    expect_modal: bool = False
    command_palette: bool = False
    share: bool = False
    settings: bool = False


class HeuristicPlanner:
    """Greedy planner that keeps the loop alive without LLM access."""

    def plan_next(
        self,
        task: str,
        snapshot: UISnapshot,
        history: Sequence[Mapping],
        avoid_selectors: Sequence[str] | None = None,
    ) -> PlannerAction:
        affordances = build_snapshot_affordances(snapshot)
        intent = _infer_intent(task)
        intent_info = parse_intent(task)
        task_intent = get_or_create_task_intent(task)
        verbs = [str(verb) for verb in task_intent.verbs if verb]
        primary = task_intent.primary_verb
        artifact_title = determine_artifact_title(task_intent)
        verbs_lower = {verb.lower() for verb in task_intent.verbs if isinstance(verb, str)}
        if (primary or "").lower():
            verbs_lower.add((primary or "").lower())
        if "create" in verbs_lower:
            intent.create = True
        if "search" in verbs_lower:
            intent.search = True
        if "filter" in verbs_lower:
            intent.filter = True
        if "toggle" in verbs_lower:
            intent.toggle = True
        if "share" in verbs_lower:
            intent.share = True

        avoided = {selector for selector in (avoid_selectors or []) if selector}
        task_keywords = _task_keywords(task)
        if task_intent.target_nouns:
            task_keywords.update(token for token in task_intent.target_nouns if token)
        semantic_terms = _semantic_terms(task)
        modal_present = bool(snapshot.modals)
        recent_creation_click = _recent_creation_click(history)
        active_modal_selector = _active_modal_selector(snapshot.modals)
        destructive_modal_open = bool(
            active_modal_selector
            and (
                _surface_has_destructive_language(snapshot.modals)
                or _surface_has_destructive_language(snapshot.overlays)
            )
        )
        task_requests_destruction = any(verb in {"delete", "remove", "discard"} for verb in verbs_lower)
        destructive_context = destructive_modal_open and not task_requests_destruction

        def finalize(action: PlannerAction) -> PlannerAction:
            return self._finalize_action(action, task, snapshot, intent_info, affordances)

        if destructive_context:
            exit_target = _find_safe_modal_exit(snapshot.clickables, avoided, owner_selector=active_modal_selector)
            if exit_target:
                return finalize(
                    PlannerAction(
                        action="click",
                        selector=exit_target.selector,
                        capture_name=_capture_name(exit_target, prefix="heuristic-dismiss"),
                        reason="Exiting destructive modal before continuing",
                    )
                )
            return finalize(
                PlannerAction(
                    action="capture",
                    capture_name="heuristic-capture-destructive-modal",
                    reason="Destructive confirm modal blocking progress",
                )
            )

        onboarding_target = _find_onboarding_target(snapshot, avoided)
        if onboarding_target:
            return finalize(
                PlannerAction(
                    action="click",
                    selector=onboarding_target.selector,
                    capture_name=_capture_name(onboarding_target, prefix="heuristic-click"),
                    reason="Heuristic planner progressing onboarding surface",
                )
            )

        if (intent.command_palette or intent.settings) and not modal_present:
            shortcut = _next_palette_shortcut(history)
            if shortcut:
                return finalize(
                    PlannerAction(
                        action="press",
                        value=shortcut,
                        capture_name="heuristic-press-command-palette",
                        reason="Open the command palette shortcut.",
                    )
                )

        if _last_action_was_successful_fill(history):
            submit_target = _find_submit_target(
                snapshot,
                avoided,
                intent,
                task_intent,
                prefer_modal=modal_present,
                active_modal_selector=active_modal_selector,
            )
            if submit_target:
                return finalize(
                    PlannerAction(
                        action="click",
                        selector=submit_target.selector,
                        capture_name=_capture_name(submit_target, prefix="heuristic-submit"),
                        reason="Heuristic planner submitting the current modal.",
                    )
                )

        if intent.capture_requested and _modal_matches_request(snapshot, intent):
            return finalize(
                PlannerAction(
                    action="capture",
                    capture_name="requested-capture",
                    reason="Task requests capturing the currently visible modal or menu.",
                )
            )

        if intent.settings:
            settings_target = _find_settings_target(snapshot.clickables, avoided)
            if settings_target:
                return finalize(
                    PlannerAction(
                        action="click",
                        selector=settings_target.selector,
                        capture_name=_capture_name(settings_target, prefix="heuristic-click"),
                        reason="Heuristic planner opening workspace/settings surface",
                    )
                )

        fill_target = _select_fill_target(
            snapshot.inputs,
            avoided,
            intent,
            history,
            task_keywords,
            task_intent,
            modal_present=modal_present,
            active_modal_selector=active_modal_selector,
            prefer_name_field=bool(task_intent.explicit_title),
        )
        if fill_target and (intent.search or intent.settings or modal_present or recent_creation_click or intent.create):
            fill_value = _fill_value_for_element(fill_target, artifact_title, task)
            descriptor = _input_descriptor(fill_target)
            reason = (
                "Fill the title field with the chosen name."
                if any(token in descriptor for token in ("name", "title"))
                else "Fill supporting details for the requested artifact."
            )
            return finalize(
                PlannerAction(
                    action="fill",
                    selector=fill_target.selector,
                    value=fill_value,
                    capture_name=_capture_name(fill_target, prefix="heuristic-fill"),
                    reason=reason,
                )
            )

        if intent.share:
            share_target = _find_share_target(snapshot.clickables, avoided)
            if share_target:
                return finalize(
                    PlannerAction(
                        action="click",
                        selector=share_target.selector,
                        capture_name=_capture_name(share_target, prefix="heuristic-click"),
                        reason="Heuristic planner opening share surface",
                    )
                )

        prefer_modals = modal_present or intent.expect_modal or intent.command_palette or intent.share
        candidates = _filter_candidates(snapshot.clickables, avoided, prefer_modals=prefer_modals, task_keywords=task_keywords)
        if candidates:
            best = _choose_best(
                candidates,
                task_keywords,
                semantic_terms,
                intent,
                history,
                task_intent,
                avoid_destructive=destructive_context,
            )
            if best and not _conflicts_with_target(best, task_intent.target_nouns, intent.create):
                return finalize(
                    PlannerAction(
                        action="click",
                        selector=best.selector,
                        capture_name=_capture_name(best, prefix="heuristic-click"),
                        reason="Heuristic planner click",
                    )
                )

        for element in snapshot.inputs:
            if not element.selector or element.selector in avoided:
                continue
            reason = "Heuristic planner input focus"
            if intent.search and "search" in (element.text or "").lower():
                reason = "Heuristic planner focusing search input"
            return finalize(
                PlannerAction(
                    action="click",
                    selector=element.selector,
                    capture_name=_capture_name(element, prefix="heuristic-input"),
                    reason=reason,
                )
            )

        for element in snapshot.breadcrumbs:
            if element.selector and element.selector not in avoided:
                return finalize(
                    PlannerAction(
                        action="click",
                        selector=element.selector,
                        capture_name=_capture_name(element, prefix="heuristic-nav"),
                        reason="Heuristic planner breadcrumb navigation",
                    )
                )

        return finalize(
            PlannerAction(
                action="capture",
                capture_name="heuristic-capture",
                reason="Heuristic planner unable to identify action",
            )
        )

    def _finalize_action(
        self,
        action: PlannerAction,
        task: str,
        snapshot: UISnapshot,
        intent_info: Dict[str, object],
        affordances: SnapshotAffordances,
    ) -> PlannerAction:
        try:
            valid, reason = validate_action_against_snapshot_and_intent(
                task,
                snapshot,
                action,
                affordances=affordances,
                intent_data=intent_info,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Heuristic validation error: %s", exc)
            valid = False
            reason = str(exc)
        if valid:
            return action
        logger.debug("Heuristic action replaced due to validation failure: %s", reason)
        return PlannerAction(
            action="capture",
            capture_name="heuristic-invalid-action",
            reason=reason or "Heuristic action rejected by validator",
        )


def _filter_candidates(
    elements: Sequence[UIElement],
    avoided: set[str],
    prefer_modals: bool,
    task_keywords: set[str],
) -> list[UIElement]:
    filtered: list[UIElement] = []
    modal_like: list[UIElement] = []
    for element in elements:
        if not element.selector or element.selector in avoided:
            continue
        if element.metadata.get("state", {}).get("disabled") is True:
            continue
        owner_dialog = element.metadata.get("ownerDialog")
        if owner_dialog:
            modal_like.append(element)
        else:
            filtered.append(element)
    modal_like.sort(key=lambda el: _element_matches_keywords(el, task_keywords), reverse=True)
    filtered.sort(key=lambda el: _element_matches_keywords(el, task_keywords), reverse=True)
    if prefer_modals and modal_like:
        return modal_like
    if modal_like:
        return modal_like + filtered
    return filtered


def _element_matches_keywords(element: UIElement, keywords: set[str]) -> int:
    if not keywords:
        return 0
    text = (element.text or element.metadata.get("ariaLabel") or "").lower()
    return 1 if any(keyword in text for keyword in keywords) else 0


def _choose_best(
    elements: Sequence[UIElement],
    keywords: set[str],
    semantic_terms: set[str],
    intent: _IntentHints,
    history: Sequence[Mapping],
    task_intent: TaskIntent,
    *,
    avoid_destructive: bool,
) -> UIElement | None:
    best: UIElement | None = None
    best_score = -1
    for element in elements:
        score = _score_element(
            element,
            keywords,
            semantic_terms,
            intent,
            history,
            task_intent,
            avoid_destructive=avoid_destructive,
        )
        if score > best_score:
            best_score = score
            best = element
    return best


def _score_element(
    element: UIElement,
    keywords: set[str],
    semantic_terms: set[str],
    intent: _IntentHints,
    history: Sequence[Mapping],
    task_intent: TaskIntent,
    *,
    avoid_destructive: bool,
) -> int:
    text = element.text or element.metadata.get("ariaLabel") or ""
    lowered_text = text.lower()
    score = _synonym_score(text or element.selector)

    if avoid_destructive and _contains_destructive_text(lowered_text):
        return -1000
    if intent.create and _contains_destructive_text(lowered_text):
        return -1000

    for kw in keywords:
        if kw in lowered_text:
            score += 4
    for phrase in semantic_terms:
        if phrase in lowered_text:
            score += 6
    if element.role in {"button", "link"}:
        score += 2

    metadata = element.metadata or {}
    state = metadata.get("state") or {}
    if state.get("selected") is True:
        score -= 4
    if state.get("disabled") is True or state.get("readonly") is True:
        score -= 10
    if metadata.get("ownerDialog"):
        score += 8
        if intent.expect_modal:
            score += 4

    target_nouns = task_intent.target_nouns
    noun_hit = text_mentions_target_noun(lowered_text, target_nouns)
    conflicting_noun = text_has_conflicting_noun(lowered_text, target_nouns)
    creation_tokens = ("new", "create", "add", "start", "compose")
    has_creation_word = any(token in lowered_text for token in creation_tokens)
    create_task = intent.create or task_intent.primary_verb == "create"

    if target_nouns:
        if noun_hit:
            score += 18
            if create_task and has_creation_word:
                score += 18
            elif create_task:
                score += 8
        elif conflicting_noun:
            penalty = 60 if create_task else 20
            score -= penalty
            if create_task:
                score -= 20

    if intent.share and "share" in lowered_text:
        score += 12
    if intent.filter and "filter" in lowered_text:
        score += 10
    if intent.command_palette and any(keyword in lowered_text for keyword in ("command", "palette", "quick find", "search")):
        score += 10
    if intent.toggle and (state.get("checked") is not None or any(token in lowered_text for token in ("toggle", "switch", "enable", "disable"))):
        score += 8
    if intent.settings and any(token in lowered_text for token in ("setting", "preference", "notification")):
        score += 8
    if intent.settings and any(token in lowered_text for token in ("workspace", "account", "profile")):
        score += 4
    if intent.search and "search" in lowered_text:
        score += 6
    if intent.capture_requested and "capture" in lowered_text:
        score += 5
    if intent.create and has_creation_word:
        if noun_hit:
            score += 8
        else:
            score -= 8
    elif intent.create and any(token in lowered_text for token in ("save", "done", "finish", "submit")):
        score += 2

    ancestry = metadata.get("ancestry") or []
    if ancestry:
        for idx, ancestor in enumerate(ancestry[:3]):
            label = (ancestor.get("label") or "").lower()
            if any(keyword in label for keyword in keywords):
                score += max(4 - idx, 1)
            if intent.expect_modal and "dialog" in (ancestor.get("role") or "").lower():
                score += 2
    owner_menu = metadata.get("ownerMenu")
    if owner_menu and keywords:
        label = (owner_menu.get("label") or "").lower()
        if any(keyword in label for keyword in keywords):
            score += 5

    score -= _history_penalty(element.selector, history)
    return score


def _task_keywords(task: str) -> set[str]:
    base = re.sub(r"[^a-z0-9\s]", " ", task.lower())
    return {token for token in base.split() if len(token) > 2 and token not in STOP_WORDS}


def _semantic_terms(task: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9]+", task.lower())
    phrases: set[str] = set()
    for size in (3, 2):
        for idx in range(len(tokens) - size + 1):
            window = tokens[idx : idx + size]
            if all(token in STOP_WORDS for token in window):
                continue
            if any(token in {"linear", "notion"} for token in window):
                continue
            phrase = " ".join(window)
            phrases.add(phrase)
    return phrases


def _synonym_score(text: str | None) -> int:
    if not text:
        return 0
    candidate = text.lower()
    for idx, pattern in enumerate(PREFERRED):
        if pattern in candidate:
            return 100 - idx
    return 0


def _capture_name(element: UIElement, prefix: str) -> str:
    raw = element.text or element.selector or "element"
    slug = re.sub(r"[^a-z0-9]+", "-", raw.lower()).strip("-")
    return f"{prefix}-{slug or 'element'}"


def _history_penalty(selector: str | None, history: Sequence[Mapping]) -> int:
    if not selector:
        return 0
    penalty = 0
    depth = 0
    for entry in reversed(history):
        if not isinstance(entry, Mapping):
            continue
        depth += 1
        if depth > 4:
            break
        if entry.get("selector") == selector:
            penalty += max(5 - depth, 1)
    return penalty


def _infer_intent(task: str) -> _IntentHints:
    lowered = task.lower()
    command_palette = "command palette" in lowered or "quick find" in lowered
    expect_modal = command_palette or "modal" in lowered or "dialog" in lowered or "menu" in lowered
    capture_requested = "capture" in lowered
    if capture_requested and any(token in lowered for token in ("modal", "dialog", "menu", "palette")):
        expect_modal = True
    return _IntentHints(
        create=any(token in lowered for token in ("create", "add", "new", "start", "compose")),
        filter="filter" in lowered or "show only" in lowered,
        toggle=any(token in lowered for token in ("toggle", "enable", "disable", "turn on", "turn off", "switch")),
        search=any(token in lowered for token in ("search", "find", "look for")),
        capture_requested=capture_requested,
        expect_modal=expect_modal,
        command_palette=command_palette,
        share="share" in lowered,
        settings=any(token in lowered for token in ("setting", "settings", "preference", "notification")),
    )


def _select_fill_target(
    inputs: Sequence[UIElement],
    avoided: set[str],
    intent: _IntentHints,
    history: Sequence[Mapping],
    task_keywords: set[str],
    task_intent: TaskIntent,
    *,
    modal_present: bool,
    active_modal_selector: Optional[str] = None,
    prefer_name_field: bool = False,
) -> UIElement | None:
    recent_fills: set[str] = {
        str(entry.get("selector"))
        for entry in history[-4:]
        if isinstance(entry, Mapping) and entry.get("action") == "fill" and entry.get("selector")
    }
    candidates: List[UIElement] = [element for element in inputs if element.selector and element.selector not in avoided]
    if modal_present:
        modal_scoped = [element for element in candidates if (element.metadata or {}).get("ownerDialog")]
        if modal_scoped:
            candidates = modal_scoped
    if active_modal_selector:
        scoped = [
            element
            for element in candidates
            if ((element.metadata or {}).get("ownerDialog") or {}).get("selector") == active_modal_selector
        ]
        if scoped:
            candidates = scoped
    if prefer_name_field:
        name_matched = [
            element
            for element in candidates
            if _input_matches_title_hint(element, task_intent.target_nouns)
        ]
        if name_matched:
            candidates = name_matched

    best: UIElement | None = None
    best_score = -1
    for element in candidates:
        if element.selector in recent_fills:
            continue
        if not _is_fillable_input(element):
            continue
        metadata = element.metadata or {}
        label = (element.text or metadata.get("labelText") or "").lower()
        placeholder = (metadata.get("placeholder") or "").lower()
        dialog_title = (metadata.get("ownerDialogTitle") or "").lower()
        owner_dialog = metadata.get("ownerDialog")
        input_type = (metadata.get("type") or "").lower()
        tag = (metadata.get("tag") or "").lower()
        content_editable = bool(metadata.get("contentEditable"))
        score = 0
        if owner_dialog:
            score += 6
            if modal_present:
                score += 4
        elif modal_present:
            score -= 2
        if content_editable or tag in {"textarea"}:
            score += 4
        elif tag == "input" and input_type in {"text", "search", "", "email", "url"}:
            score += 4
        elif tag == "input":
            score += 1
        if any(keyword in label for keyword in ("name", "title")):
            score += 6
        if any(keyword in placeholder for keyword in ("name", "title", "summary", "description")):
            score += 4
        if any(keyword in label for keyword in task_keywords):
            score += 3
        descriptor = f"{label} {placeholder} {dialog_title}"
        if task_intent.target_nouns:
            if text_mentions_target_noun(descriptor, task_intent.target_nouns):
                score += 10
            elif text_has_conflicting_noun(descriptor, task_intent.target_nouns):
                score -= 6
        if prefer_name_field and not _input_matches_title_hint(element, task_intent.target_nouns):
            score -= 6
        if intent.search and any(keyword in (label + placeholder) for keyword in ("search", "find", "filter")):
            score += 6
        if intent.settings and "setting" in (label + placeholder):
            score += 4
        if intent.create and score == 0 and owner_dialog:
            score += 2
        if score > best_score:
            best = element
            best_score = score
    return best


def _recent_creation_click(history: Sequence[Mapping]) -> bool:
    checked = 0
    for entry in reversed(history):
        if not isinstance(entry, Mapping):
            continue
        checked += 1
        if checked > 4:
            break
        if entry.get("action") != "click":
            continue
        selector = str(entry.get("selector") or "").lower()
        reason = str(entry.get("reason") or "").lower()
        if any(token in selector for token in ("create", "new", "add")) or any(token in reason for token in ("create", "new", "add")):
            return True
    return False


def _next_palette_shortcut(history: Sequence[Mapping]) -> str | None:
    used = {
        str(entry.get("value"))
        for entry in history
        if isinstance(entry, Mapping) and entry.get("action") == "press" and entry.get("value")
    }
    for shortcut in ("Meta+K", "Control+K"):
        if shortcut not in used:
            return shortcut
    return None


def _find_settings_target(elements: Sequence[UIElement], avoided: set[str]) -> UIElement | None:
    keywords = ("notification", "settings", "workspace", "preferences", "account")
    for element in elements:
        if not element.selector or element.selector in avoided:
            continue
        text = (element.text or element.metadata.get("ariaLabel") or "").lower()
        if any(keyword in text for keyword in keywords):
            return element
    return None


def _find_onboarding_target(snapshot: UISnapshot, avoided: set[str]) -> UIElement | None:
    url = snapshot.url.lower()
    if "onboarding" not in url:
        return None
    keywords = ("continue", "skip", "back to previous workspace", "use an existing account", "enter workspace")
    for element in snapshot.clickables:
        if not element.selector or element.selector in avoided:
            continue
        text = (element.text or element.metadata.get("ariaLabel") or "").lower()
        if any(keyword in text for keyword in keywords):
            return element
    return None


def _find_share_target(elements: Sequence[UIElement], avoided: set[str]) -> UIElement | None:
    keywords = ("share", "invite", "copy link")
    for element in elements:
        if not element.selector or element.selector in avoided:
            continue
        text = (element.text or element.metadata.get("ariaLabel") or "").lower()
        if any(keyword in text for keyword in keywords):
            return element
    return None


def _modal_matches_request(snapshot: UISnapshot, intent: _IntentHints) -> bool:
    candidates = snapshot.modals or []
    if not candidates:
        return False
    keywords = []
    if intent.share:
        keywords.extend(("share", "invite"))
    if intent.command_palette or intent.search:
        keywords.extend(("command", "palette", "quick find", "search"))
    if intent.settings:
        keywords.extend(("settings", "preferences", "notifications"))
    if intent.capture_requested:
        keywords.extend(("modal", "dialog"))
    for element in candidates:
        text = (element.text or element.metadata.get("ariaLabel") or "").lower()
        if any(keyword in text for keyword in keywords if keyword):
            return True
    return False


def _is_fillable_input(element: UIElement) -> bool:
    metadata = element.metadata or {}
    tag = (metadata.get("tag") or "").lower()
    input_type = (metadata.get("type") or "").lower()
    role = (element.role or metadata.get("role") or "").lower()
    content_editable = bool(metadata.get("contentEditable"))
    if tag == "select":
        return False
    if role in {"checkbox", "radio", "switch"}:
        return False
    if input_type in {"checkbox", "radio", "submit", "button", "file"}:
        return False
    if tag in {"button"}:
        return False
    return True if content_editable or tag in {"input", "textarea"} or role in {"textbox"} else False


def _last_action_was_successful_fill(history: Sequence[Mapping]) -> bool:
    if not history:
        return False
    entry = history[-1]
    return (
        isinstance(entry, Mapping)
        and entry.get("action") == "fill"
        and entry.get("success") is True
    )


def _find_submit_target(
    snapshot: UISnapshot,
    avoided: set[str],
    intent: _IntentHints,
    task_intent: TaskIntent,
    *,
    prefer_modal: bool,
    active_modal_selector: Optional[str] = None,
) -> UIElement | None:
    target_nouns: Sequence[str] = tuple(task_intent.target_nouns or ())
    submit_element, _choice = (None, None)
    if prefer_modal or active_modal_selector:
        submit_element, _choice = select_modal_submit_candidate(
            snapshot,
            target_nouns=target_nouns,
            owner_selector=active_modal_selector,
        )
        if submit_element and submit_element.selector and submit_element.selector not in avoided:
            return submit_element

    elements: List[UIElement] = []
    seen: set[str] = set()
    for group in (snapshot.primary_actions, snapshot.clickables):
        for element in group or []:
            selector = (element.selector or "").strip()
            if not selector or selector in seen:
                continue
            seen.add(selector)
            elements.append(element)

    submit_keywords = ("create", "add", "save", "done", "submit", "finish", "publish", "launch", "apply", "confirm", "ok", "continue")
    scoped_matches: List[Tuple[int, UIElement]] = []
    modal_matches: List[Tuple[int, UIElement]] = []
    ambient_matches: List[Tuple[int, UIElement]] = []

    for element in elements:
        if not element.selector or element.selector in avoided:
            continue
        metadata = element.metadata or {}
        text = (element.text or metadata.get("ariaLabel") or "").lower()
        owner_dialog = metadata.get("ownerDialog")
        owner_selector = ((owner_dialog or {}).get("selector") or "").strip()
        if not text:
            continue
        score = 0
        if owner_dialog:
            score += 5
        if prefer_modal and owner_dialog:
            score += 3
        if active_modal_selector and owner_selector == active_modal_selector:
            score += 6
        if any(keyword in text for keyword in submit_keywords):
            score += 8
        if intent.create and task_intent.target_nouns:
            if text_mentions_target_noun(text, task_intent.target_nouns):
                score += 6
            elif text_has_conflicting_noun(text, task_intent.target_nouns):
                score -= 12
        if intent.create and any(keyword in text for keyword in ("create", "add", "new", "save")):
            score += 4
        if intent.create and task_intent.target_nouns and text_has_conflicting_noun(text, task_intent.target_nouns) and not text_mentions_target_noun(text, task_intent.target_nouns):
            continue
        if score <= 0:
            continue
        entry = (score, element)
        if active_modal_selector and owner_selector == active_modal_selector:
            scoped_matches.append(entry)
        elif owner_dialog:
            modal_matches.append(entry)
        else:
            ambient_matches.append(entry)

    def _pick(candidates: List[Tuple[int, UIElement]]) -> Optional[UIElement]:
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]

    if active_modal_selector:
        target = _pick(scoped_matches)
        if target:
            return target
        target = _pick(modal_matches)
        if target:
            return target

    if prefer_modal:
        target = _pick(modal_matches)
        if target:
            return target

    combined = modal_matches + ambient_matches
    return _pick(combined)


def _fill_value_for_element(element: UIElement, artifact_title: str, task: str) -> str:
    descriptor = _input_descriptor(element)
    if descriptor:
        if any(token in descriptor for token in ("name", "title")):
            return artifact_title
        if any(token in descriptor for token in ("summary", "description", "details", "notes")):
            return _derive_summary_text(task, artifact_title)
    return artifact_title


def _derive_summary_text(task: str, artifact_title: str) -> str:
    concise = " ".join(task.strip().split())
    if len(concise) > 160:
        concise = concise[:157] + "..."
    return f"{artifact_title} captures: {concise}"


def _contains_destructive_text(text: str) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in _DESTRUCTIVE_KEYWORDS)


def _conflicts_with_target(element: UIElement, target_nouns: Sequence[str], create_intent: bool) -> bool:
    if not create_intent or not target_nouns:
        return False
    metadata = element.metadata or {}
    label = (element.text or metadata.get("ariaLabel") or "").lower()
    if not label:
        return False
    return text_has_conflicting_noun(label, target_nouns) and not text_mentions_target_noun(label, target_nouns)


def _surface_has_destructive_language(elements: Sequence[UIElement]) -> bool:
    for element in elements:
        metadata = element.metadata or {}
        text = (element.text or metadata.get("ariaLabel") or "").lower()
        if _contains_destructive_text(text):
            return True
    return False


def _find_safe_modal_exit(
    elements: Sequence[UIElement],
    avoided: set[str],
    *,
    owner_selector: Optional[str],
) -> Optional[UIElement]:
    preferred: List[UIElement] = []
    fallback: List[UIElement] = []
    for element in elements:
        if not element.selector or element.selector in avoided:
            continue
        metadata = element.metadata or {}
        owner = ((metadata.get("ownerDialog") or {}).get("selector") or "").strip()
        label = (element.text or metadata.get("ariaLabel") or "").lower()
        if not label:
            continue
        if any(keyword in label for keyword in _SAFE_EXIT_KEYWORDS):
            if owner_selector and owner == owner_selector:
                preferred.append(element)
            else:
                fallback.append(element)
    if preferred:
        return preferred[0]
    if fallback:
        return fallback[0]
    return None


def _active_modal_selector(modals: Sequence[UIElement]) -> Optional[str]:
    if not modals:
        return None
    modal = modals[-1]
    return (modal.selector or "").strip() or None


def _input_matches_title_hint(element: UIElement, target_nouns: Sequence[str]) -> bool:
    descriptor = _input_descriptor(element) or ""
    if any(token in descriptor for token in ("name", "title")):
        return True
    return text_mentions_target_noun(descriptor, target_nouns)


def _input_descriptor(element: UIElement) -> str:
    metadata = element.metadata or {}
    pieces = [
        element.text,
        metadata.get("labelText"),
        metadata.get("ownerDialogTitle"),
        metadata.get("placeholder"),
        metadata.get("ariaLabel"),
    ]
    collapsed = " ".join(part for part in pieces if part)
    return collapsed.strip().lower()
