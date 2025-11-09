"""Lightweight task-intent parsing helpers shared across planners and evaluators."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
import secrets
from typing import Dict, List, Optional, Sequence

GENERIC_OBJECT_KEYWORDS = {
    "project",
    "projects",
    "issue",
    "issues",
    "task",
    "tasks",
    "ticket",
    "tickets",
    "story",
    "stories",
    "epic",
    "epics",
    "page",
    "pages",
    "doc",
    "docs",
    "document",
    "documents",
    "database",
    "view",
    "board",
    "list",
    "feed",
    "notification",
    "notifications",
    "settings",
    "command palette",
    "palette",
    "command",
    "menu",
    "modal",
    "share",
    "workspace",
    "team",
    "project roadmap",
    "table",
    "tables",
    "record",
    "records",
    "page view",
    "kanban",
}

_OBJECT_STOP_WORDS = {
    "a",
    "an",
    "the",
    "new",
    "to",
    "for",
    "with",
    "named",
    "called",
    "title",
    "of",
    "in",
    "on",
    "by",
    "about",
    "current",
    "this",
    "that",
    "my",
    "your",
    "workspace",
    "item",
    "items",
    "thing",
    "things",
    "stuff",
    "app",
    "apps",
    "linear",
    "notion",
    "agent",
}

_APP_HINT_WORDS = {
    "linear",
    "notion",
    "asana",
    "slack",
    "github",
    "figma",
    "jira",
    "trello",
    "clickup",
    "monday",
    "salesforce",
    "zendesk",
    "intercom",
    "confluence",
}

_VERB_PATTERNS = {
    "create": r"(?:create|make|add|start|launch|build|new)",
    "open": r"(?:open|view|show|switch|navigate|go to|access)",
    "filter": r"(?:filter|show only|narrow|limit)",
    "search": r"(?:search|find|look for)",
    "toggle": r"(?:toggle|switch on|switch off|enable|disable)",
    "capture": r"(?:capture|screenshot|record)",
    "share": r"(?:share|invite)",
}

_OBJECT_PATTERN = re.compile(
    r"(?:create|make|add|start|launch|build|open|view|show|switch|navigate|filter|search|toggle|capture|share)\s+(?:a|an|the|new)?\s*([a-z][a-z0-9\s\-]{1,40})",
    flags=re.IGNORECASE,
)


@dataclass
class TaskIntent:
    """Structured summary of a natural-language instruction."""

    primary_verb: Optional[str]
    verbs: List[str] = field(default_factory=list)
    target_nouns: List[str] = field(default_factory=list)
    explicit_title: Optional[str] = None
    raw: str = ""
    generated_title: Optional[str] = None


def parse_intent(task: str) -> Dict[str, Optional[str] | List[str]]:
    """
    Parse a natural-language task into canonical components.

    Returns dict with keys:
      - verb: canonical verb like "create", "open", etc.
      - nouns: ordered list of lower-case target nouns (most specific first).
      - explicit_title: optional literal title from 'named', 'title:', or quoted text.
    """

    normalized = task.strip()
    lowered = normalized.lower()
    verbs = _extract_verbs(lowered)
    primary_verb = verbs[0] if verbs else _detect_primary_verb(lowered)
    nouns = _extract_target_nouns(lowered)
    explicit_title = _extract_explicit_title(normalized)
    return {
        "verbs": verbs,
        "verb": primary_verb,
        "nouns": nouns,
        "explicit_title": explicit_title,
    }


def parse_task_intent(task: str) -> TaskIntent:
    """Legacy wrapper returning TaskIntent dataclass for existing call sites."""

    parsed = parse_intent(task)
    verbs = list(parsed.get("verbs") or [])
    primary = verbs[0] if verbs else parsed.get("verb")
    intent_obj = TaskIntent(
        primary_verb=primary,
        verbs=verbs,
        target_nouns=list(parsed.get("nouns") or []),
        explicit_title=parsed.get("explicit_title"),
        raw=task,
    )
    determine_artifact_title(intent_obj)
    return intent_obj


def text_mentions_target_noun(text: str, nouns: Sequence[str]) -> bool:
    if not text or not nouns:
        return False
    lowered = text.lower()
    return any(noun and noun in lowered for noun in nouns)


def text_has_conflicting_noun(text: str, nouns: Sequence[str]) -> bool:
    if not text or not nouns:
        return False
    lowered = text.lower()
    noun_set = set(nouns)
    for keyword in GENERIC_OBJECT_KEYWORDS:
        if keyword in noun_set:
            continue
        if keyword in lowered:
            return True
    return False


def _detect_primary_verb(task_lower: str) -> Optional[str]:
    for verb, pattern in _VERB_PATTERNS.items():
        if re.search(pattern, task_lower):
            return verb
    return None


def _extract_verbs(task_lower: str) -> List[str]:
    hits: List[tuple[int, str]] = []
    for verb, pattern in _VERB_PATTERNS.items():
        match = re.search(pattern, task_lower)
        if match:
            hits.append((match.start(), verb))
    hits.sort(key=lambda item: item[0])
    ordered: List[str] = []
    for _, verb in hits:
        if verb not in ordered:
            ordered.append(verb)
    return ordered


def _extract_target_nouns(task_lower: str) -> List[str]:
    nouns: List[str] = []
    seen: set[str] = set()
    for match in _OBJECT_PATTERN.finditer(task_lower):
        phrase = match.group(1) or ""
        candidate = _normalize_object_phrase(phrase)
        if not candidate:
            continue
        if candidate not in seen:
            nouns.append(candidate)
            seen.add(candidate)
    if nouns:
        return nouns

    fallback = _fallback_keyword_scan(task_lower)
    for candidate in fallback:
        if candidate not in seen:
            nouns.append(candidate)
            seen.add(candidate)
    return nouns


def _normalize_object_phrase(phrase: str) -> Optional[str]:
    tokens = _tokenize(phrase)
    if not tokens:
        return None
    candidate = _match_keyword_phrase(tokens)
    if candidate:
        return candidate
    for token in tokens:
        noun = _canonicalize_noun(token)
        if noun:
            return noun
    return None


def _extract_explicit_title(task: str) -> Optional[str]:
    patterns = (
        r"title\s*[:=]\s*([^\n,.;]+)",
        r"\bwith\s+(?:the\s+)?title\s+([\"'][^\"']+[\"']|[a-z0-9][a-z0-9\s\-\']{2,})",
        r"\btitle\s+([\"'][^\"']+[\"']|[a-z0-9][a-z0-9\s\-\']{2,})",
        r"\bnamed\s+([\"'][^\"']+[\"']|[a-z0-9][a-z0-9\s\-\']{2,})",
        r"\bname\s+([\"'][^\"']+[\"']|[a-z0-9][a-z0-9\s\-\']{2,})",
        r"\bcall(?:ed)?\s+([\"'][^\"']+[\"']|[a-z0-9][a-z0-9\s\-\']{2,})",
    )
    for pattern in patterns:
        match = re.search(pattern, task, flags=re.IGNORECASE)
        if match:
            raw = match.group(1).strip().strip('"').strip("'")
            candidate = _strip_trailing_context(raw)
            if candidate:
                return candidate
    quoted = re.findall(r'"([^"]{3,})"', task)
    if quoted:
        return _strip_trailing_context(quoted[0].strip())
    return None


def _tokenize(text: str) -> List[str]:
    return [token for token in re.findall(r"[a-z0-9]+", text.lower()) if token not in _OBJECT_STOP_WORDS]


def _match_keyword_phrase(tokens: Sequence[str]) -> Optional[str]:
    if not tokens:
        return None
    max_window = min(3, len(tokens))
    for size in range(max_window, 1, -1):
        for idx in range(len(tokens) - size + 1):
            phrase = " ".join(tokens[idx : idx + size])
            noun = _canonicalize_phrase(phrase)
            if noun:
                return noun
    return None


def _canonicalize_phrase(phrase: str) -> Optional[str]:
    normalized = phrase.strip()
    if not normalized:
        return None
    if normalized in _APP_HINT_WORDS:
        return None
    if normalized in GENERIC_OBJECT_KEYWORDS:
        for token in normalized.split():
            noun = _canonicalize_noun(token)
            if noun:
                return noun
        return normalized
    return None


def _canonicalize_noun(token: str) -> Optional[str]:
    cleaned = token.strip().lower()
    if not cleaned or len(cleaned) < 3:
        return None
    if cleaned in _APP_HINT_WORDS or cleaned in _OBJECT_STOP_WORDS:
        return None
    if cleaned in GENERIC_OBJECT_KEYWORDS:
        return _singularize(cleaned)
    singular = _singularize(cleaned)
    if singular in GENERIC_OBJECT_KEYWORDS:
        return singular
    return None


def _singularize(token: str) -> str:
    if len(token) <= 3:
        return token
    if token.endswith("ies") and len(token) > 3:
        return token[:-3] + "y"
    if token.endswith("ses") or token.endswith("xes"):
        return token[:-2]
    if token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _fallback_keyword_scan(task_lower: str) -> List[str]:
    tokens = _tokenize(task_lower)
    nouns: List[str] = []
    seen: set[str] = set()
    for token in tokens:
        noun = _canonicalize_noun(token)
        if noun and noun not in seen:
            nouns.append(noun)
            seen.add(noun)
    return nouns


def _strip_trailing_context(value: str) -> Optional[str]:
    if not value:
        return None
    parts = re.split(r"\b(?:in|on|for|at|with|using|to)\b", value, maxsplit=1)
    trimmed = parts[0].strip()
    return trimmed or None


def determine_artifact_title(task_intent: TaskIntent) -> str:
    if task_intent.generated_title:
        return task_intent.generated_title
    explicit = (task_intent.explicit_title or "").strip()
    if explicit:
        task_intent.generated_title = explicit
        return explicit
    noun = task_intent.target_nouns[0] if task_intent.target_nouns else "item"
    raw_lower = (task_intent.raw or "").lower()
    if _requests_random_title(raw_lower):
        title = _generate_random_title(noun)
    else:
        title = _generate_default_title(noun)
    task_intent.generated_title = title
    return title


def _generate_default_title(noun: Optional[str]) -> str:
    base = _trim_title_base(f"New {(noun or 'Item').strip().split()[0].title()}")
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    suffix = secrets.token_hex(2)
    return f"{base} {today}-{suffix}".strip()


def _generate_random_title(noun: Optional[str]) -> str:
    base_token = (noun or "Item").strip().split()[0].title()
    base = _trim_title_base(f"Automation {base_token}")
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    suffix = secrets.token_hex(2).upper()
    return f"{base} {today}-{suffix}".strip()


def _trim_title_base(value: str, limit: int = 28) -> str:
    value = value.strip()
    if len(value) <= limit:
        return value
    return value[:limit].rstrip()


def _requests_random_title(task_lower: str) -> bool:
    if not task_lower:
        return False
    patterns = (
        r"random\s+(?:title|name)",
        r"with\s+(?:a\s+)?random\s+(?:title|name)",
        r"random\s+(?:project|item)\s+(?:title|name)?",
    )
    return any(re.search(pattern, task_lower) for pattern in patterns)


_TASK_INTENT_CACHE: Dict[str, TaskIntent] = {}


def get_or_create_task_intent(task: str) -> TaskIntent:
    cached = _TASK_INTENT_CACHE.get(task)
    if cached:
        return cached
    intent_obj = parse_task_intent(task)
    _TASK_INTENT_CACHE[task] = intent_obj
    return intent_obj


def reset_task_intent(task: str) -> None:
    _TASK_INTENT_CACHE.pop(task, None)


__all__ = [
    "TaskIntent",
    "parse_intent",
    "parse_task_intent",
    "GENERIC_OBJECT_KEYWORDS",
    "text_mentions_target_noun",
    "text_has_conflicting_noun",
    "determine_artifact_title",
    "get_or_create_task_intent",
    "reset_task_intent",
]
