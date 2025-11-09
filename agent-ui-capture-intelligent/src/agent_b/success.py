"""Generic success detection for Agent B.

Root cause summary (from datasets captured on 2025-11-04 … 2025-11-07):
    • datasets/linear/create-a-project-in-linear/run.jsonl shows the agent
      clicking the same “New view” control until max steps with no done action.
    • datasets/notion/open-workspace-settings-and-capture-the-modal/run.jsonl
      captures the correct modal repeatedly but never emits a completion signal.

Those runs demonstrate that the loop never reasons about generic UI success
patterns. This module fixes that by spotting common intents (open, create,
filter, toggle, command palette, share modal, etc.) using only on-screen text,
ARIA roles, and action history — no app-specific selectors or URLs.
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence

from .intent import parse_intent, text_has_conflicting_noun, text_mentions_target_noun
from .modal_submit import history_entry_looks_like_submit
from .models import UIElement, UISnapshot

# Phrases that typically describe UI state rather than task entities.
FILLER_TOKENS = {
    "the",
    "a",
    "an",
    "to",
    "into",
    "and",
    "for",
    "with",
    "please",
    "kindly",
    "current",
    "page",
    "pages",
    "workspace",
    "app",
    "apps",
    "tab",
    "tabs",
    "view",
    "panel",
    "section",
    "modal",
    "dialog",
    "menu",
    "screen",
    "state",
    "task",
    "agent",
    "ui",
    "open",
    "create",
    "add",
    "new",
    "make",
    "set",
    "show",
    "only",
    "capture",
    "toggle",
    "enable",
    "disable",
    "search",
    "find",
    "filter",
    "page",
    "pages",
    "project",
    "projects",
    "database",
    "databases",
    "priority",
    "linear",
    "notion",
}

VERB_HINTS = {
    "open",
    "navigate",
    "go",
    "switch",
    "show",
    "visit",
    "create",
    "add",
    "new",
    "start",
    "compose",
    "filter",
    "search",
    "find",
    "toggle",
    "enable",
    "disable",
    "turn",
    "set",
    "capture",
    "share",
    "invite",
}

CREATION_SUCCESS_WORDS = {"created", "added", "saved", "done", "success", "created!", "added!"}
FILTER_WORDS = {"filter", "filtered", "showing", "show only", "sorted", "grouped"}
MODAL_WORDS = {"modal", "dialog", "palette", "quick find", "command", "menu", "settings", "preferences", "share"}
DESTRUCTIVE_MODAL_TOKENS = {"discard", "delete", "remove", "are you sure", "permanently", "abandon"}
HIGH_SALIENCE_ROLES = {
    "heading",
    "link",
    "button",
    "menuitem",
    "listitem",
    "gridcell",
    "row",
    "treeitem",
    "option",
    "tab",
    "article",
    "alert",
    "status",
}

logger = logging.getLogger(__name__)


@dataclass
class SuccessSignal:
    satisfied: bool
    reason: Optional[str] = None
    category: Optional[str] = None


@dataclass
class _IntentProfile:
    open_view: bool = False
    create: bool = False
    filter: bool = False
    toggle: bool = False
    search: bool = False
    share: bool = False
    command_palette: bool = False
    capture_modal: bool = False
    expect_modal: bool = False
    expected_toggle_state: Optional[bool] = None
    search_terms: tuple[str, ...] = ()


@dataclass
class _SuccessContext:
    task: str
    intents: _IntentProfile
    target_terms: tuple[str, ...]
    target_phrases: tuple[str, ...]
    target_nouns: tuple[str, ...]
    explicit_title: Optional[str]

    def match_text(self, text: str) -> Optional[str]:
        normalized = _normalize_text(text)
        if not normalized:
            return None
        for phrase in self.target_phrases:
            if phrase and phrase in normalized:
                return phrase
        for term in self.target_terms:
            if term and term in normalized:
                return term
        return None


@dataclass
class _FillEvent:
    index: int
    value: str
    descriptor: str


def evaluate_success(
    task: str,
    snapshot: UISnapshot,
    history: Sequence[Mapping[str, object]],
) -> SuccessSignal:
    """
    Return a SuccessSignal when the UI reflects the requested task based on generic cues.
    """

    intent_info = parse_intent(task)
    intents = _infer_intents(task)
    target_terms = _extract_target_terms(task)
    target_phrases = _extract_target_phrases(task, target_terms)
    context = _SuccessContext(
        task=task,
        intents=intents,
        target_terms=target_terms,
        target_phrases=target_phrases,
        target_nouns=tuple(intent_info.get("nouns") or ()),
        explicit_title=intent_info.get("explicit_title"),
    )

    checks = (
        _modal_goal_met,
        _command_palette_goal_met,
        _navigation_goal_met,
        _filter_goal_met,
        _toggle_goal_met,
        _creation_goal_met,
        _search_goal_met,
    )

    for check in checks:
        signal = check(context, snapshot, history)
        if signal.satisfied:
            return signal

    return SuccessSignal(satisfied=False)


def _modal_goal_met(context: _SuccessContext, snapshot: UISnapshot, history: Sequence[Mapping[str, object]]) -> SuccessSignal:
    if not (context.intents.expect_modal or context.intents.capture_modal or context.intents.share):
        return SuccessSignal(False)
    modal_texts = list(_iter_texts(snapshot.modals))
    if not modal_texts:
        return SuccessSignal(False)
    keywords = set()
    if context.intents.share:
        keywords.add("share")
    if context.intents.command_palette:
        keywords.update({"command", "palette", "quick find"})
    if context.intents.capture_modal:
        keywords.update(context.target_terms)
    keywords.update(token for phrase in context.target_phrases for token in phrase.split())
    for text in modal_texts:
        normalized = _normalize_text(text)
        if not normalized:
            continue
        if (context.intents.create or context.intents.open_view) and _text_contains_destructive_language(normalized):
            continue
        if keywords and not any(keyword in normalized for keyword in keywords):
            continue
        match = context.match_text(text) or next((word for word in keywords if word in normalized), None)
        if match:
            category = "modal" if "palette" not in normalized else "palette"
            return SuccessSignal(True, reason=f"Modal visible with '{match}'", category=category)
    return SuccessSignal(False)


def _command_palette_goal_met(context: _SuccessContext, snapshot: UISnapshot, history: Sequence[Mapping[str, object]]) -> SuccessSignal:
    if not context.intents.command_palette:
        return SuccessSignal(False)
    palette_keywords = {"command palette", "quick find", "search"}
    for text in _iter_texts(snapshot.modals) or _iter_texts(snapshot.overlays):
        normalized = _normalize_text(text)
        if not normalized:
            continue
        if any(keyword in normalized for keyword in palette_keywords):
            return SuccessSignal(True, reason="Command palette / quick find modal is open", category="palette")
    return SuccessSignal(False)


def _navigation_goal_met(context: _SuccessContext, snapshot: UISnapshot, history: Sequence[Mapping[str, object]]) -> SuccessSignal:
    if not context.intents.open_view:
        return SuccessSignal(False)

    for element in _iter_elements(snapshot.clickables, snapshot.breadcrumbs, snapshot.primary_actions):
        text = element.text or element.metadata.get("ariaLabel") or ""
        match = context.match_text(text)
        if not match:
            continue
        if _is_element_active(element):
            return SuccessSignal(True, reason=f"Active navigation element matches '{match}'", category="navigation")

    title = snapshot.title or ""
    title_match = context.match_text(title)
    if title_match:
        return SuccessSignal(True, reason=f"Document title references '{title_match}'", category="navigation")

    url_match = context.match_text(snapshot.url or "")
    if url_match:
        return SuccessSignal(True, reason=f"URL contains '{url_match}'", category="navigation")
    return SuccessSignal(False)


def _filter_goal_met(context: _SuccessContext, snapshot: UISnapshot, history: Sequence[Mapping[str, object]]) -> SuccessSignal:
    if not context.intents.filter:
        return SuccessSignal(False)
    history_mentions_filter = any(
        "filter" in str(entry.get("reason", "")).lower() or "filter" in str(entry.get("selector", "")).lower()
        for entry in history[-5:]
    )
    texts = list(_iter_texts(snapshot.clickables, snapshot.primary_actions, snapshot.modals))
    for text in texts:
        normalized = _normalize_text(text)
        if not normalized:
            continue
        if not any(keyword in normalized for keyword in FILTER_WORDS):
            continue
        match = context.match_text(text)
        if match or history_mentions_filter:
            return SuccessSignal(True, reason=f"Filter indicator found with '{match or 'filter'}'", category="filter")
    return SuccessSignal(False)


def _toggle_goal_met(context: _SuccessContext, snapshot: UISnapshot, history: Sequence[Mapping[str, object]]) -> SuccessSignal:
    if not context.intents.toggle:
        return SuccessSignal(False)

    desired = context.intents.expected_toggle_state
    for element in _iter_elements(snapshot.clickables, snapshot.inputs):
        label = element.text or element.metadata.get("ariaLabel") or ""
        match = context.match_text(label)
        if not match:
            continue
        state = _coerce_bool(element.metadata.get("state", {}).get("checked"))
        if state is None:
            continue
        if desired is not None and state != desired:
            continue
        human_state = "on" if state else "off"
        return SuccessSignal(True, reason=f"Toggle '{match}' is {human_state}", category="toggle")
    return SuccessSignal(False)


def _creation_goal_met(context: _SuccessContext, snapshot: UISnapshot, history: Sequence[Mapping[str, object]]) -> SuccessSignal:
    if not context.intents.create:
        return SuccessSignal(False)
    if snapshot.modals:
        return SuccessSignal(False)
    if _destructive_dialog_visible(snapshot) and not _task_mentions_destruction(context.task):
        return SuccessSignal(False)
    fill_event = _recent_name_fill_event(history, context.explicit_title, context.target_nouns)
    if not fill_event:
        return SuccessSignal(False)
    submit_entry = _recent_submit_click_after(history, fill_event.index, context.target_nouns)
    if not submit_entry:
        return SuccessSignal(False)

    primary_value = (context.explicit_title or fill_event.value or "").strip()
    if not primary_value:
        return SuccessSignal(False)
    if _value_visible_outside_modal(
        snapshot,
        primary_value,
        context.target_nouns,
        strict=bool(context.explicit_title),
    ):
        trimmed = primary_value.strip()
        logger.debug(
            "Create success via fill value %r and submit selector=%s",
            trimmed,
            submit_entry.get("selector"),
        )
        return SuccessSignal(True, reason=f"Created artifact '{trimmed}' is visible", category="create")

    elements = list(_iter_elements(snapshot.clickables, snapshot.primary_actions))
    for element in elements:
        text = element.text or element.metadata.get("ariaLabel") or ""
        normalized = _normalize_text(text)
        if not normalized or not any(word in normalized for word in CREATION_SUCCESS_WORDS):
            continue
        if context.target_nouns:
            if text_has_conflicting_noun(normalized, context.target_nouns):
                continue
            if not text_mentions_target_noun(normalized, context.target_nouns):
                continue
        match = context.match_text(text)
        if match:
            logger.debug(
                "Create success inferred from UI copy '%s' after fill value %r",
                text,
                primary_value,
            )
            return SuccessSignal(True, reason=f"Success copy references '{match}'", category="create")

    return SuccessSignal(False)


def _search_goal_met(context: _SuccessContext, snapshot: UISnapshot, history: Sequence[Mapping[str, object]]) -> SuccessSignal:
    if not context.intents.search:
        return SuccessSignal(False)

    if not context.intents.search_terms:
        return SuccessSignal(False)

    texts = list(_iter_texts(snapshot.modals, snapshot.clickables, snapshot.inputs, snapshot.primary_actions))
    for text in texts:
        normalized = _normalize_text(text)
        if not normalized:
            continue
        for term in context.intents.search_terms:
            if term in normalized:
                return SuccessSignal(True, reason=f"Search surface shows '{term}'", category="search")
    return SuccessSignal(False)


def _value_visible_outside_modal(
    snapshot: UISnapshot,
    value: str,
    target_nouns: Sequence[str],
    *,
    strict: bool,
) -> bool:
    normalized_value = _normalize_text(value)
    if not normalized_value:
        return False
    search_groups = (snapshot.clickables, snapshot.primary_actions, snapshot.breadcrumbs)
    for element in _iter_elements(*search_groups):
        if _element_matches_value(element, normalized_value, strict=strict, target_nouns=target_nouns):
            return True
    title = _normalize_text(snapshot.title or "")
    if title and normalized_value in title and _text_length_reasonable(title, normalized_value, strict=strict):
        if target_nouns:
            if text_has_conflicting_noun(title, target_nouns):
                return False
            if not text_mentions_target_noun(title, target_nouns):
                return False
        if strict and not title.startswith(normalized_value):
            return False
        return True
    return False


def _recent_submit_click_after(
    history: Sequence[Mapping[str, object]],
    start_index: int,
    target_nouns: Sequence[str],
) -> Optional[Mapping[str, object]]:
    for idx in range(start_index + 1, len(history)):
        entry = history[idx]
        if not isinstance(entry, Mapping):
            continue
        action = str(entry.get("action") or "").lower()
        if action == "click":
            if not entry.get("success"):
                continue
            descriptor = _normalize_text(f"{entry.get('selector', '')} {entry.get('reason', '')}")
            if _text_contains_neutral_language(descriptor):
                continue
            if history_entry_looks_like_submit(entry, target_nouns):
                return entry
        if action == "done":
            break
    return None


def _text_length_reasonable(text: str, value: str, *, strict: bool) -> bool:
    baseline = len(value)
    limit = baseline + 10 if strict else max(baseline + 30, 60)
    return len(text) <= limit


def _element_matches_value(
    element: UIElement,
    normalized_value: str,
    *,
    strict: bool,
    target_nouns: Sequence[str],
) -> bool:
    metadata = element.metadata or {}
    if metadata.get("ownerDialog"):
        return False
    text = _normalize_text(element.text or metadata.get("ariaLabel"))
    if not text or normalized_value not in text:
        return False
    if not _is_high_salience_element(element):
        return False
    if not _text_length_reasonable(text, normalized_value, strict=strict):
        return False
    if strict and not text.startswith(normalized_value):
        return False
    if strict and len(text) > len(normalized_value):
        remainder = text[len(normalized_value) :].strip(" -–—:·")
        if remainder and len(remainder) > 10:
            return False
    if target_nouns:
        if text_has_conflicting_noun(text, target_nouns):
            return False
        if not text_mentions_target_noun(text, target_nouns):
            return False
    return True


def _is_high_salience_element(element: UIElement) -> bool:
    role = (element.role or "").lower()
    if role in HIGH_SALIENCE_ROLES:
        return True
    metadata = element.metadata or {}
    state = metadata.get("state") or {}
    if state.get("selected") is True:
        return True
    ancestry = metadata.get("ancestry") or []
    for ancestor in ancestry:
        ancestor_role = (ancestor.get("role") or "").lower()
        if ancestor_role in {"listitem", "row", "article"}:
            return True
    return False


def _text_contains_destructive_language(text: str) -> bool:
    lowered = text.lower()
    return any(token in lowered for token in DESTRUCTIVE_MODAL_TOKENS)


def _text_contains_neutral_language(text: str) -> bool:
    lowered = text.lower()
    for token in ("dismiss", "cancel", "close", "discard", "keep editing", "stay"):
        if token in lowered:
            return True
    return False


def _destructive_dialog_visible(snapshot: UISnapshot) -> bool:
    surfaces = list(snapshot.modals or []) + list(snapshot.overlays or [])
    for element in surfaces:
        metadata = element.metadata or {}
        text = _normalize_text(element.text or metadata.get("ariaLabel"))
        if text and _text_contains_destructive_language(text):
            return True
    return False


def _task_mentions_destruction(task: str) -> bool:
    lowered = task.lower()
    return any(keyword in lowered for keyword in ("delete", "remove", "discard", "trash", "close without saving"))


def _iter_elements(*groups: Iterable[UIElement]) -> Iterable[UIElement]:
    for group in groups:
        for element in group:
            if element:
                yield element


def _iter_texts(*groups: Iterable[UIElement]) -> Iterable[str]:
    for element in _iter_elements(*groups):
        if element.text:
            yield element.text
        aria_label = element.metadata.get("ariaLabel")
        if aria_label:
            yield aria_label


def _normalize_text(value: Optional[str]) -> str:
    if not value:
        return ""
    collapsed = re.sub(r"\s+", " ", value).strip().lower()
    return collapsed


def _text_contains_phrase(value: Optional[str], phrase: Optional[str]) -> bool:
    if not value or not phrase:
        return False
    return _normalize_text(phrase) in _normalize_text(value)


def _infer_intents(task: str) -> _IntentProfile:
    lowered = task.lower()
    intent = _IntentProfile()
    intent.open_view = any(token in lowered for token in ("open", "navigate", "switch", "go to", "view", "show"))
    intent.create = any(token in lowered for token in ("create", "add", "new", "start", "compose"))
    intent.filter = "filter" in lowered or "show only" in lowered or "only show" in lowered
    intent.toggle = any(token in lowered for token in ("toggle", "enable", "disable", "turn on", "turn off", "switch on", "switch off"))
    intent.search = any(token in lowered for token in ("search", "look for", "find"))
    intent.share = "share" in lowered
    intent.command_palette = "command palette" in lowered or "quick find" in lowered or "command menu" in lowered
    intent.capture_modal = "modal" in lowered or "dialog" in lowered or ("capture" in lowered and "menu" in lowered)
    intent.expect_modal = intent.command_palette or intent.capture_modal or "modal" in lowered or "dialog" in lowered
    intent.expected_toggle_state = None
    if any(token in lowered for token in ("enable", "turn on", "switch on", "check")):
        intent.expected_toggle_state = True
    elif any(token in lowered for token in ("disable", "turn off", "switch off", "uncheck")):
        intent.expected_toggle_state = False
    intent.search_terms = _extract_search_terms(lowered)
    return intent


def _extract_target_terms(task: str) -> tuple[str, ...]:
    tokens = re.findall(r"[a-z0-9]+", task.lower())
    terms = []
    for token in tokens:
        if len(token) <= 2:
            continue
        if token in VERB_HINTS:
            continue
        if token in FILLER_TOKENS:
            continue
        if token not in terms:
            terms.append(token)
    return tuple(terms)


def _extract_target_phrases(task: str, target_terms: tuple[str, ...]) -> tuple[str, ...]:
    tokens = re.findall(r"[a-z0-9]+", task.lower())
    phrases = []
    token_count = len(tokens)
    for size in (3, 2):
        for idx in range(token_count - size + 1):
            window = tokens[idx : idx + size]
            if not any(token in target_terms for token in window):
                continue
            phrase = " ".join(window)
            if phrase not in phrases:
                phrases.append(phrase)
    return tuple(phrases[:10])


def _recent_name_fill_event(
    history: Sequence[Mapping[str, object]],
    explicit_title: Optional[str],
    target_nouns: Sequence[str],
) -> Optional[_FillEvent]:
    explicit_normalized = _normalize_text(explicit_title) if explicit_title else None
    for index in range(len(history) - 1, -1, -1):
        entry = history[index]
        if not isinstance(entry, Mapping):
            continue
        if entry.get("action") != "fill" or not entry.get("success"):
            continue
        descriptor = " ".join(
            part
            for part in (
                str(entry.get("selector") or ""),
                str(entry.get("capture_name") or ""),
                str(entry.get("reason") or ""),
            )
            if part
        ).lower()
        if not _looks_like_name_field(descriptor, target_nouns):
            continue
        value = str(entry.get("value") or "").strip()
        if not value:
            continue
        if explicit_normalized and explicit_normalized not in _normalize_text(value):
            continue
        return _FillEvent(index=index, value=value, descriptor=descriptor)
    return None


def _looks_like_name_field(descriptor: str, target_nouns: Sequence[str]) -> bool:
    if not descriptor:
        return False
    if "summary" in descriptor or "description" in descriptor:
        return False
    if any(token in descriptor for token in ("name", "title")):
        return True
    return any(noun and noun in descriptor for noun in target_nouns)


def _is_element_active(element: UIElement) -> bool:
    state = element.metadata.get("state") or {}
    if state.get("selected") is True:
        return True
    classes = (element.metadata.get("classes") or "").lower()
    if any(keyword in classes for keyword in ("active", "selected", "current", "is-active")):
        return True
    aria_current = element.metadata.get("ariaCurrent") or element.metadata.get("aria-current")
    if isinstance(aria_current, str) and aria_current.lower() in {"page", "true", "step"}:
        return True
    owner_dialog = element.metadata.get("ownerDialog") or {}
    if owner_dialog and owner_dialog.get("role") in {"menuitemradio", "menuitemcheckbox"} and state.get("checked") is True:
        return True
    return False


def _coerce_bool(value: object) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.lower()
        if lowered in {"true", "on", "checked"}:
            return True
        if lowered in {"false", "off", "unchecked"}:
            return False
    return None


def _extract_search_terms(task_lower: str) -> tuple[str, ...]:
    patterns = [
        r"search(?:\s+for)?\s+([a-z0-9\s]+?)(?:\s+in|\s+on|\s+for|\s+within|$)",
        r"find\s+([a-z0-9\s]+?)(?:\s+in|\s+on|\s+for|\s+within|$)",
        r"look\s+for\s+([a-z0-9\s]+?)(?:\s+in|\s+on|\s+for|\s+within|$)",
    ]
    terms = []
    for pattern in patterns:
        match = re.search(pattern, task_lower)
        if not match:
            continue
        group = match.group(1).strip()
        if not group:
            continue
        candidate = re.sub(r"\s+", " ", group)
        if candidate and candidate not in terms:
            terms.append(candidate)
    normalized_terms = []
    for term in terms:
        normalized_terms.append(term.lower())
    return tuple(normalized_terms[:5])


__all__ = ["SuccessSignal", "evaluate_success"]
