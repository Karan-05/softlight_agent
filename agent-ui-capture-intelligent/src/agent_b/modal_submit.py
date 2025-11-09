"""Shared heuristics for identifying modal submit buttons and related history signals."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List, Mapping, Optional, Sequence, Tuple

from .intent import text_has_conflicting_noun, text_mentions_target_noun
from .models import UIElement, UISnapshot

logger = logging.getLogger(__name__)

SUBMIT_VERBS = (
    "create",
    "add",
    "save",
    "done",
    "confirm",
    "submit",
    "continue",
    "ok",
    "start",
    "finish",
    "publish",
    "launch",
    "apply",
)
DESTRUCTIVE_TOKENS = ("discard", "delete", "remove", "cancel", "close", "dismiss")
NEUTRAL_TOKENS = ("dismiss", "cancel", "close", "keep editing", "stay")
FOOTER_HINTS = ("footer", "modal-footer", "dialog-footer", "actions", "button-row")


@dataclass
class ModalSubmitChoice:
    element: UIElement
    score: int
    descriptor: str
    owner_selector: Optional[str]


def select_modal_submit_candidate(
    snapshot: UISnapshot,
    *,
    target_nouns: Sequence[str],
    owner_selector: Optional[str] = None,
    intent_is_destructive: bool = False,
) -> Tuple[Optional[UIElement], Optional[ModalSubmitChoice]]:
    """Return the highest scoring submit control inside the active modal/drawer."""

    active_modal_selector = owner_selector or _active_modal_selector(snapshot.modals)
    if not active_modal_selector:
        return None, None

    modal_titles = _modal_titles(snapshot, active_modal_selector)
    pools = list(snapshot.primary_actions or []) + list(snapshot.clickables or [])
    seen: set[str] = set()
    choices: List[ModalSubmitChoice] = []
    for element in pools:
        selector = (element.selector or "").strip()
        if not selector or selector in seen:
            continue
        seen.add(selector)
        choice = _score_candidate(
            element,
            target_nouns=target_nouns,
            active_modal_selector=active_modal_selector,
            owner_selector=owner_selector,
            intent_is_destructive=intent_is_destructive,
            modal_titles=modal_titles,
        )
        if choice:
            choices.append(choice)

    if not choices:
        logger.debug(
            "No modal submit candidates found for modal=%s. Top modal controls: %s",
            active_modal_selector,
            _summarize_modal_buttons(snapshot, active_modal_selector)[:5],
        )
        return None, None

    choices.sort(key=lambda entry: entry.score, reverse=True)
    best = choices[0]
    logger.debug(
        "Modal submit choice selector=%s owner=%s score=%s descriptor=%r",
        (best.element.selector or "").strip(),
        best.owner_selector,
        best.score,
        best.descriptor,
    )
    return best.element, best


def history_entry_looks_like_submit(
    entry: Mapping[str, object],
    target_nouns: Sequence[str],
    *,
    intent_is_destructive: bool = False,
) -> bool:
    """Heuristic that checks if a successful click entry resembles a modal submit action."""

    if str(entry.get("action") or "").lower() != "click":
        return False
    if not entry.get("success"):
        return False

    descriptor_parts = [
        str(entry.get("selector") or ""),
        str(entry.get("reason") or ""),
        str(entry.get("capture_name") or ""),
    ]
    descriptor = " ".join(part for part in descriptor_parts if part).strip()
    normalized = descriptor.lower()
    if descriptor:
        if not intent_is_destructive and _contains_any(normalized, DESTRUCTIVE_TOKENS):
            return False
        if _contains_any(normalized, NEUTRAL_TOKENS):
            return False
        if _text_has_submit_verb(normalized):
            return True
        if target_nouns and text_mentions_target_noun(normalized, target_nouns) and "submit" in normalized:
            return True

    selector_hint = _selector_hint(str(entry.get("selector") or ""))
    if selector_hint:
        lowered_hint = selector_hint.lower()
        if not intent_is_destructive and _contains_any(lowered_hint, DESTRUCTIVE_TOKENS):
            return False
        if _text_has_submit_verb(lowered_hint):
            return True
        if target_nouns and text_mentions_target_noun(lowered_hint, target_nouns) and "submit" in lowered_hint:
            return True

    return False


def _score_candidate(
    element: UIElement,
    *,
    target_nouns: Sequence[str],
    active_modal_selector: str,
    owner_selector: Optional[str],
    intent_is_destructive: bool,
    modal_titles: Sequence[str],
) -> Optional[ModalSubmitChoice]:
    metadata = element.metadata or {}
    owner = ((metadata.get("ownerDialog") or {}).get("selector") or "").strip()
    if not _within_modal_scope(metadata, owner, owner_selector, active_modal_selector):
        return None

    role_value = (element.role or "").lower()
    if role_value and role_value not in {"button"}:
        return None

    descriptor = _element_descriptor(element)
    if not descriptor:
        descriptor = _selector_hint(element.selector or "") or ""
    normalized = descriptor.lower()
    if not normalized:
        return None
    if not intent_is_destructive and _contains_any(normalized, DESTRUCTIVE_TOKENS):
        return None
    if _contains_any(normalized, NEUTRAL_TOKENS):
        return None
    if not _text_has_submit_verb(normalized):
        return None

    score = 0
    if owner_selector and owner == owner_selector:
        score += 18
    elif owner == active_modal_selector:
        score += 12
    elif _ancestry_indicates_modal(metadata):
        score += 6

    if _element_in_footer(metadata):
        score += 2
    if _looks_primary(metadata):
        score += 3
    score += 8  # base submit verb bonus

    mention_target = text_mentions_target_noun(normalized, target_nouns)
    conflicts_target = text_has_conflicting_noun(normalized, target_nouns)
    if mention_target:
        score += 6
    elif conflicts_target:
        return None

    if mention_target:
        score += 3  # encourage verb+noun combinations

    if "title" in normalized or "name" in normalized:
        score -= 8
    if _looks_like_modal_title(normalized, modal_titles):
        score -= 6

    if (metadata.get("state") or {}).get("disabled") is True:
        return None

    if score <= 0:
        return None

    return ModalSubmitChoice(element=element, score=score, descriptor=descriptor.strip(), owner_selector=owner)


def _element_descriptor(element: UIElement) -> str:
    metadata = element.metadata or {}
    parts = [
        element.text,
        metadata.get("ariaLabel"),
        metadata.get("labelText"),
        metadata.get("ownerDialogTitle"),
    ]
    return " ".join(part.strip() for part in parts if isinstance(part, str) and part.strip())


def _selector_hint(selector: str) -> str:
    if not selector:
        return ""
    name_match = re.search(r"name=/(.+?)/i", selector)
    if name_match:
        return name_match.group(1)
    direct_match = re.search(r'name="([^"]+)"', selector)
    if direct_match:
        return direct_match.group(1)
    text_match = re.search(r'text="?([^"]+)"?', selector)
    if text_match:
        return text_match.group(1)
    return ""


def _text_has_submit_verb(text: str) -> bool:
    return _contains_any(text, SUBMIT_VERBS)


def _contains_any(text: str, tokens: Sequence[str]) -> bool:
    lowered = text.lower()
    return any(token in lowered for token in tokens if token)


def _looks_primary(metadata: Mapping[str, object]) -> bool:
    classes = str(metadata.get("classes") or "").lower()
    return any(token in classes for token in ("primary", "confirm", "submit", "cta", "button--primary"))


def _element_in_footer(metadata: Mapping[str, object]) -> bool:
    ancestry = metadata.get("ancestry") or []
    for ancestor in ancestry:
        classes = str(ancestor.get("classes") or "").lower()
        tag = str(ancestor.get("tag") or "").lower()
        role = str(ancestor.get("role") or "").lower()
        if any(hint in classes for hint in FOOTER_HINTS):
            return True
        if tag == "footer" or role == "contentinfo":
            return True
    return False


def _active_modal_selector(modals: Sequence[UIElement]) -> Optional[str]:
    if not modals:
        return None
    modal = modals[-1]
    selector = (modal.selector or "").strip()
    return selector or None


def _within_modal_scope(
    metadata: Mapping[str, object],
    owner: Optional[str],
    owner_selector: Optional[str],
    active_modal_selector: str,
) -> bool:
    if owner_selector:
        if owner == owner_selector:
            return True
    else:
        if owner == active_modal_selector:
            return True
    return _ancestry_indicates_modal(metadata)


def _ancestry_indicates_modal(metadata: Mapping[str, object]) -> bool:
    ancestry = metadata.get("ancestry") or []
    for entry in ancestry:
        role = str(entry.get("role") or "").lower()
        classes = str(entry.get("classes") or "").lower()
        tag = str(entry.get("tag") or "").lower()
        if role in {"dialog", "alertdialog"}:
            return True
        if tag in {"dialog"}:
            return True
        if any(token in classes for token in ("modal", "dialog", "drawer", "sheet", "layer")):
            return True
    return False


def _modal_titles(snapshot: UISnapshot, selector: Optional[str]) -> List[str]:
    titles: List[str] = []
    if not selector:
        return titles
    for modal in snapshot.modals or []:
        modal_selector = (modal.selector or "").strip()
        if not modal_selector:
            continue
        if modal_selector != selector:
            continue
        candidates = [
            modal.text,
            (modal.metadata or {}).get("ariaLabel"),
            (modal.metadata or {}).get("labelText"),
        ]
        for candidate in candidates:
            normalized = _normalize(candidate)
            if normalized:
                titles.append(normalized)
    return titles


def _looks_like_modal_title(text: str, modal_titles: Sequence[str]) -> bool:
    normalized = _normalize(text)
    if not normalized:
        return False
    return any(normalized == title or normalized.startswith(title) for title in modal_titles)


def _normalize(value: Optional[str]) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", value).strip().lower()


def _summarize_modal_buttons(snapshot: UISnapshot, modal_selector: Optional[str]) -> List[str]:
    summaries: List[str] = []
    if not modal_selector:
        return summaries
    for element in snapshot.clickables[:40]:
        metadata = element.metadata or {}
        owner = ((metadata.get("ownerDialog") or {}).get("selector") or "").strip()
        if owner and owner != modal_selector:
            continue
        if not owner and not _ancestry_indicates_modal(metadata):
            continue
        descriptor = _element_descriptor(element) or _selector_hint(element.selector or "")
        if not descriptor:
            continue
        summaries.append(f"{descriptor} [{element.selector}]")
    return summaries


__all__ = [
    "ModalSubmitChoice",
    "select_modal_submit_candidate",
    "history_entry_looks_like_submit",
]
