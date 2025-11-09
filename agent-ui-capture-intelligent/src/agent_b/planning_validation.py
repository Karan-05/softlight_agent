"""Shared helpers for validating planner actions against the UISnapshot and task intent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .intent import parse_intent, text_has_conflicting_noun, text_mentions_target_noun
from .models import PlannerAction, UIElement, UISnapshot

_NAME_HINTS = ("name", "title")
_CREATE_VERBS = {"create", "add", "new", "make", "start", "launch", "build", "compose"}
_DESTRUCTIVE_HINTS = ("discard", "delete", "remove", "trash", "close without saving", "abandon", "archive")


@dataclass
class SnapshotAffordances:
    all_selectors: Set[str]
    all_selectors_lower: Set[str]
    input_selectors: Set[str]
    input_selectors_lower: Set[str]
    element_lookup: Dict[str, UIElement]
    element_lookup_lower: Dict[str, UIElement]


def build_snapshot_affordances(snapshot: UISnapshot) -> SnapshotAffordances:
    selectors: Set[str] = set()
    lowered: Set[str] = set()
    element_lookup: Dict[str, UIElement] = {}
    element_lookup_lower: Dict[str, UIElement] = {}
    input_selectors: Set[str] = set()
    input_lower: Set[str] = set()

    collections = (
        snapshot.clickables,
        snapshot.inputs,
        snapshot.modals,
        snapshot.primary_actions,
        snapshot.breadcrumbs,
        snapshot.overlays,
    )
    for elements in collections:
        for element in elements:
            selector = (element.selector or "").strip()
            if selector:
                _register_selector(selectors, lowered, element_lookup, element_lookup_lower, selector, element)
            for derived in _derived_selectors(element):
                _register_selector(selectors, lowered, element_lookup, element_lookup_lower, derived, element)

    for element in snapshot.inputs:
        selector = (element.selector or "").strip()
        if not selector:
            continue
        input_selectors.add(selector)
        input_lower.add(selector.lower())

    return SnapshotAffordances(
        all_selectors=selectors,
        all_selectors_lower=lowered,
        input_selectors=input_selectors,
        input_selectors_lower=input_lower,
        element_lookup=element_lookup,
        element_lookup_lower=element_lookup_lower,
    )


def collect_snapshot_selectors(snapshot: UISnapshot) -> Tuple[Set[str], Set[str]]:
    affordances = build_snapshot_affordances(snapshot)
    return affordances.all_selectors, affordances.input_selectors


def selector_is_known(selector: str, allowed: Set[str], allowed_lower: Set[str]) -> bool:
    candidate = selector.strip()
    if not candidate:
        return True
    if candidate in allowed:
        return True
    return candidate.lower() in allowed_lower


def validate_action_against_snapshot_and_intent(
    task: str,
    snapshot: UISnapshot,
    action: PlannerAction,
    *,
    affordances: Optional[SnapshotAffordances] = None,
    intent_data: Optional[Dict[str, object]] = None,
) -> Tuple[bool, Optional[str]]:
    affordance_index = affordances or build_snapshot_affordances(snapshot)
    selector = (action.selector or "").strip()
    if selector and not selector_is_known(selector, affordance_index.all_selectors, affordance_index.all_selectors_lower):
        return False, f"Selector '{selector}' not present in snapshot"
    if action.action == "fill":
        if not selector:
            return False, "Fill action missing selector"
        if not selector_is_known(selector, affordance_index.input_selectors, affordance_index.input_selectors_lower):
            return False, f"Fill selector '{selector}' not present in snapshot inputs"
        value = action.value
        if not isinstance(value, str) or not value.strip():
            return False, "Fill action must include a non-empty value"
    element = _lookup_element(selector, affordance_index)

    if action.action == "fill":
        if not element:
            return False, f"Fill selector '{selector}' not found in snapshot inputs"
        if not _element_is_fillable(element):
            return False, f"Fill selector '{selector}' resolves to a non-fillable element"

    intent = intent_data or parse_intent(task)
    verbs = [
        str(verb).lower()
        for verb in (intent.get("verbs") or [])
        if isinstance(verb, str)
    ]
    primary_verb = (verbs[0] if verbs else intent.get("verb")) or ""
    primary_verb = str(primary_verb).lower()
    verb_candidates = verbs or ([primary_verb] if primary_verb else [])
    target_nouns: Sequence[str] = tuple(intent.get("nouns") or ())
    explicit_title = intent.get("explicit_title") if intent else None
    is_create_intent = any(verb in _CREATE_VERBS for verb in verb_candidates)
    modal_present = bool(snapshot.modals)
    if not is_create_intent:
        return True, None

    if action.action == "click" and selector and element and target_nouns:
        descriptor = _element_descriptor(element)
        if descriptor:
            mentions_target = text_mentions_target_noun(descriptor, target_nouns)
            conflicts = text_has_conflicting_noun(descriptor, target_nouns)
            if conflicts and not mentions_target:
                return False, f"Selector '{selector}' references conflicting object text for create-intent"
            if _descriptor_is_destructive(descriptor) and not mentions_target:
                return False, "Creation intent should not click destructive controls"
            if _descriptor_is_neutral_or_destructive(descriptor):
                return False, "Creation intent should not click dismiss/cancel controls"
            if "search" in descriptor and not mentions_target:
                return False, "Creation intent should not click search controls"
            if modal_present and not _element_in_modal(element) and _looks_like_creation_submit(descriptor, target_nouns):
                return False, "Creation modal is open; submit inside the modal instead of background controls"

    if action.action == "fill" and element:
        label_text = _input_descriptor(element)
        lowered_label = label_text.lower()
        has_name_hint = any(token in lowered_label for token in _NAME_HINTS)
        if modal_present and not _element_in_modal(element) and (has_name_hint or text_mentions_target_noun(lowered_label, target_nouns)):
            return False, "Creation modal is open; fill the requested field inside that modal"
        if explicit_title:
            prioritized = _preferred_title_inputs(snapshot.inputs, target_nouns, require_name_hint=True)
            fallback = _preferred_title_inputs(snapshot.inputs, target_nouns, require_name_hint=False)
            preferred_selectors = prioritized or fallback
            if preferred_selectors and selector not in preferred_selectors:
                return False, "Explicit title should be entered into the name/title field for the requested artifact"
            if not _value_contains_title(action.value, explicit_title):
                return False, "Fill value does not contain the requested explicit title"
        if target_nouns:
            mentions_target = text_mentions_target_noun(lowered_label, target_nouns)
            conflicts = text_has_conflicting_noun(lowered_label, target_nouns)
            if conflicts and not mentions_target:
                return False, "Input label references a conflicting object type"
            if not has_name_hint and not mentions_target:
                return False, "Input label does not mention name/title or the target noun"
        return True, None

    return True, None


def _register_selector(
    selectors: Set[str],
    lowered: Set[str],
    lookup: Dict[str, UIElement],
    lookup_lower: Dict[str, UIElement],
    selector: str,
    element: UIElement,
) -> None:
    selectors.add(selector)
    lowered.add(selector.lower())
    lookup.setdefault(selector, element)
    lookup_lower.setdefault(selector.lower(), element)


def _derived_selectors(element: UIElement) -> Tuple[str, ...]:
    text = (element.text or "").strip()
    role = (element.role or "").strip()
    derived: list[str] = []
    if text:
        derived.append(_text_selector(text))
        if role:
            derived.append(_role_selector(role, text))
    return tuple(derived)


def _lookup_element(selector: Optional[str], affordances: SnapshotAffordances) -> Optional[UIElement]:
    if not selector:
        return None
    element = affordances.element_lookup.get(selector)
    if element:
        return element
    return affordances.element_lookup_lower.get(selector.lower())


def _element_descriptor(element: UIElement) -> str:
    metadata = element.metadata or {}
    candidates = [
        element.text,
        metadata.get("ariaLabel"),
        metadata.get("labelText"),
        metadata.get("placeholder"),
    ]
    descriptor = " ".join(value for value in candidates if value)
    return descriptor.lower().strip()


def _input_descriptor(element: UIElement) -> str:
    metadata = element.metadata or {}
    pieces = [
        element.text,
        metadata.get("labelText"),
        metadata.get("ownerDialogTitle"),
        metadata.get("placeholder"),
        metadata.get("ariaLabel"),
    ]
    descriptor = " ".join(value for value in pieces if value)
    return descriptor.strip().lower()


def _escape_selector_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _text_selector(value: str) -> str:
    return f'text="{_escape_selector_value(value[:80])}"'


def _role_selector(role: str, value: str) -> str:
    return f'role={role}[name="{_escape_selector_value(value[:80])}"]'


def _value_contains_title(value: Optional[str], title: Optional[str]) -> bool:
    if not value or not title:
        return False
    return title.strip().lower() in value.strip().lower()


def _element_is_fillable(element: UIElement) -> bool:
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
    if not content_editable and tag not in {"input", "textarea"} and role not in {"textbox"}:
        return False
    return True


def _preferred_title_inputs(
    inputs: Sequence[UIElement],
    target_nouns: Sequence[str],
    *,
    require_name_hint: bool = False,
) -> List[str]:
    strong: List[str] = []
    fallback: List[str] = []
    for element in inputs:
        selector = (element.selector or "").strip()
        if not selector:
            continue
        descriptor = _input_descriptor(element)
        if not descriptor:
            continue
        mentions_name = any(token in descriptor for token in _NAME_HINTS)
        if require_name_hint and not mentions_name:
            continue
        mentions_target = text_mentions_target_noun(descriptor, target_nouns)
        if mentions_name and mentions_target:
            strong.append(selector)
        elif mentions_name or (not require_name_hint and mentions_target):
            fallback.append(selector)
    if strong:
        return strong
    return fallback


def _element_in_modal(element: UIElement) -> bool:
    metadata = element.metadata or {}
    owner_dialog = metadata.get("ownerDialog") or {}
    return bool(owner_dialog)


def _looks_like_creation_submit(text: str, target_nouns: Sequence[str]) -> bool:
    lowered = text.lower()
    if any(token in lowered for token in ("create", "add", "save", "start", "new", "submit", "finish", "done")):
        return True
    return text_mentions_target_noun(lowered, target_nouns)


def _descriptor_is_destructive(text: str) -> bool:
    lowered = text.lower()
    return any(token in lowered for token in _DESTRUCTIVE_HINTS)


def _descriptor_is_neutral_or_destructive(text: str) -> bool:
    lowered = text.lower()
    neutral = ("dismiss", "cancel", "close", "keep editing", "stay")
    return any(token in lowered for token in neutral) or _descriptor_is_destructive(text)


__all__ = [
    "SnapshotAffordances",
    "build_snapshot_affordances",
    "collect_snapshot_selectors",
    "selector_is_known",
    "validate_action_against_snapshot_and_intent",
]
