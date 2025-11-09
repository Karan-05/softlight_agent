"""Single-step LLM planner for Agent B."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

try:  # optional dependency
    from openai import AsyncOpenAI  # type: ignore
except ImportError:
    AsyncOpenAI = None  # type: ignore[assignment]
from pydantic import ValidationError

from .config import OPENAI_MODEL
from .intent import determine_artifact_title, parse_intent, get_or_create_task_intent
from .models import PlannerAction, UISnapshot
from .planning_validation import build_snapshot_affordances, validate_action_against_snapshot_and_intent

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are Agent B's planning assistant. Pick exactly one UI action for the executor based on the visible affordances.\n"
    "Return ONLY a JSON object with required keys for the chosen action. Include a short reason.\n"
    "Selectors MUST come verbatim from the snapshot lists (clickables, inputs, modals, breadcrumbs, overlays, primary actions). "
    "Copy the exact `selector` string; if nothing suitable exists, capture more context or wait for UI to update instead of guessing.\n"
    "When the task names a specific object (project, issue, page, database, view, etc.), interact only with controls whose visible text references that object and avoid elements mentioning different objects.\n"
    "For fill actions, you MUST pick a selector from the 'Visible inputs' section (true text inputs, textareas, or contenteditable textboxes). "
    "Never use selectors from buttons, links, checkboxes, radios, switches, or other non-input elements for a fill. "
    "If the task provides an explicit title or name, type that exact text into the input whose label/placeholder mentions 'name', 'title', or the target noun before submitting.\n"
    "When handling create/add/new tasks, avoid destructive controls (Discard/Delete/Remove) unless the instruction explicitly requests a destructive action—close or cancel those modals instead.\n"
    "If no suitable input is listed, emit a capture or wait_for rather than guessing.\n"
    "Reason over the task intent (open/navigate, create/add, filter/search, toggle, capture modal/share menu, command palette, etc.) "
    "and choose actions that move toward that state.\n"
    "Interact with visible modals/dialogs/menus before touching the background UI, and do not repeat the same selector+action more than twice—"
    "if nothing new is available, emit a capture to refresh context.\n"
    "Emit done only when generic success cues are visible (requested modal open, target view active, new artifact text rendered, toggle switched, filter active, etc.).\n"
)

SCRIPT_SYSTEM_PROMPT = (
    "You are Agent B's strategist. Produce a JSON object with keys 'actions' (array of steps) and 'confidence' (0-1)."
    "Each step must include action, reason, and any required selector/url/value, plus an optional expect field describing the postcondition."
    "Use selectors exactly as provided in the snapshot; do not invent new identifiers, and only use fill selectors from the 'Visible inputs' list."
    " If an explicit title is provided, plan to fill it into the name/title field for the target artifact before submitting."
    " For create/add/new intents never click destructive buttons (Discard/Delete/Remove) unless the user explicitly asked for deletion—prefer cancel/close actions instead."
    " Reason about intent (open/create/filter/toggle/search/capture) and focus on modals before background elements."
    " When unsure about a selector, prefer a capture action to gather context."
    " Avoid reusing selectors that failed previously or repeating the same selector more than twice."
)


class InvalidPlannerActionError(RuntimeError):
    """Raised when the LLM proposes an action that cannot be executed safely."""


class LLMPlanner:
    """Delegate planning to an OpenAI model using the provided UI snapshot."""

    def __init__(self, client: Optional[AsyncOpenAI] = None, model: str = OPENAI_MODEL):
        self.client = client
        self.model = model

    async def plan_next(self, task: str, snapshot: UISnapshot, history: List[Dict[str, Any]]) -> PlannerAction:
        if not self.client:
            raise RuntimeError("OpenAI client is not configured")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": self._build_prompt(task, snapshot, history)},
        ]

        logger.debug("Planner prompt: %s", messages[-1]["content"])
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.1,
        )

        raw = response.choices[0].message.content if response.choices else ""
        logger.debug("Planner raw response: %s", raw)
        payload_str, extracted_reason = _extract_json_and_reason(raw)
        logger.debug("Planner JSON payload repr: %r", payload_str)
        if not payload_str:
            raise ValueError("Planner produced empty response")

        sanitized = _sanitize_json_string(payload_str)
        try:
            payload = json.loads(sanitized)
        except json.JSONDecodeError as exc:
            logger.debug("Sanitized planner payload repr: %r", sanitized)
            raise ValueError(
                f"Planner returned invalid JSON: {payload_str}; error={exc}"
            ) from exc

        if not isinstance(payload, dict):
            raise ValueError("Planner response must be a JSON object")

        try:
            action = PlannerAction.model_validate(payload)
        except ValidationError as exc:
            raise ValueError(f"Planner response failed validation: {payload}") from exc

        if extracted_reason:
            action.reason = extracted_reason

        _guard_against_login_navigation(action)
        affordances = build_snapshot_affordances(snapshot)
        intent_info = parse_intent(task)
        valid, reason = validate_action_against_snapshot_and_intent(
            task,
            snapshot,
            action,
            affordances=affordances,
            intent_data=intent_info,
        )
        if not valid:
            logger.debug("Planner action rejected by validator: %s (%s)", action.model_dump(), reason)
            raise InvalidPlannerActionError(reason or "Planner action rejected by validator")

        return action

    async def plan_script(
        self,
        task: str,
        snapshot: UISnapshot,
        history: List[Dict[str, Any]],
        *,
        max_actions: int = 5,
    ) -> List[PlannerAction]:
        if not self.client:
            raise RuntimeError("OpenAI client is not configured")

        prompt = self._build_prompt(task, snapshot, history, include_guidance=True, max_actions=max_actions)
        messages = [
            {"role": "system", "content": SCRIPT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.1,
        )

        raw = response.choices[0].message.content if response.choices else ""
        payload_str, extracted_reason = _extract_json_and_reason(raw)
        if not payload_str:
            raise ValueError("Planner produced empty script response")

        sanitized = _sanitize_json_string(payload_str)
        try:
            payload = json.loads(sanitized)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Planner returned invalid JSON for script: {payload_str}") from exc

        actions_payload = payload.get("actions")
        if not isinstance(actions_payload, list):
            raise ValueError("Planner script response missing 'actions' array")

        steps: List[PlannerAction] = []
        affordances = build_snapshot_affordances(snapshot)
        intent_info = parse_intent(task)
        for idx, entry in enumerate(actions_payload):
            if not isinstance(entry, dict):
                continue
            try:
                candidate = PlannerAction.model_validate(entry)
            except ValidationError:
                logger.debug("Skipping invalid script entry: %s", entry)
                continue
            valid, reason = validate_action_against_snapshot_and_intent(
                task,
                snapshot,
                candidate,
                affordances=affordances,
                intent_data=intent_info,
            )
            if not valid:
                logger.debug("Discarding script entry (%s): %s", reason, entry)
                continue
            steps.append(candidate)
            if len(steps) >= max_actions:
                break
        for step in steps:
            _guard_against_login_navigation(step)
        return steps

    def _build_prompt(
        self,
        task: str,
        snapshot: UISnapshot,
        history: List[Dict[str, Any]],
        *,
        include_guidance: bool = False,
        max_actions: int = 5,
    ) -> str:
        intent = parse_intent(task)
        task_intent = get_or_create_task_intent(task)
        explicit_title = (task_intent.explicit_title or "").strip()
        keywords = _extract_keywords(task)
        suggested_name = determine_artifact_title(task_intent)
        suggested_summary = _suggest_artifact_summary(suggested_name, task)
        lines: List[str] = [
            f"Task: {task}",
            f"URL: {snapshot.url}",
            f"Title: {snapshot.title or ''}",
            "",
        ]

        if snapshot.detected_app:
            lines.append(f"Detected app: {snapshot.detected_app}")
            lines.append("")

        if keywords:
            lines.append(f"Task keywords: {', '.join(keywords)}")
            lines.append("")
            if suggested_name:
                lines.append(f"Suggested name: {suggested_name}")
            if suggested_summary:
                lines.append(f"Suggested summary: {suggested_summary}")
            lines.append("")

        intent_summary = _describe_task_intent(task)
        if intent_summary:
            lines.append(f"Intent focus: {intent_summary}")
            lines.append("")

        lines.extend(_format_elements("Visible clickables", snapshot.clickables[:40]))
        lines.extend(_format_elements("Visible inputs", snapshot.inputs[:20]))
        lines.extend(_format_elements("Visible modals", snapshot.modals[:5]))
        if snapshot.primary_actions:
            lines.extend(_format_elements("Primary action candidates", snapshot.primary_actions[:15]))
        if snapshot.breadcrumbs:
            lines.extend(_format_elements("Breadcrumb trail", snapshot.breadcrumbs[:8]))
        input_json = self._collect_inputs(snapshot)
        if input_json:
            lines.append("")
            lines.append("Input selectors (JSON):")
            lines.append(input_json)

        history_tail = history[-5:]
        lines.append("")
        lines.append("Recent history (most recent last):")
        if history_tail:
            for idx, entry in enumerate(history_tail, start=1):
                action = entry.get("action")
                selector = entry.get("selector")
                url = entry.get("url")
                success = entry.get("success")
                reason = entry.get("reason")
                lines.append(f"- step{idx}: action={action} selector={selector} url={url} success={success} reason={reason}")
        else:
            lines.append("- none")

        if keywords:
            lines.append("")
            lines.append("Elements matching task keywords:")
            hits = _collect_keyword_hits(keywords, snapshot)
            if hits:
                for hit in hits[:15]:
                    lines.append(f"- {hit}")
            else:
                lines.append("- none")

        modal_hint = _detect_creation_modal(keywords, snapshot, suggested_name, task)
        if modal_hint:
            lines.append("")
            lines.append(f"Modal guidance: {modal_hint}")

        success_hint = _detect_created_entries(snapshot, suggested_name)
        if success_hint:
            lines.append("")
            lines.append(f"Success cues: {success_hint}")

        if include_guidance:
            lines.append("")
            lines.append(f"Plan up to {max_actions} actions that progress the task. Avoid repeating selectors or actions that already failed in the history.")

        lines.append("")
        if include_guidance:
            lines.append('Respond ONLY with JSON: {"actions": [...]} where each entry is a step object.')
        else:
            lines.append("Respond with a single JSON object describing the next action.")
        return "\n".join(lines)

    def _collect_inputs(self, snapshot: UISnapshot) -> str:
        inputs: List[Dict[str, str]] = []
        for element in snapshot.inputs[:30]:
            selector = (element.selector or "").strip()
            if not selector:
                continue
            metadata = element.metadata or {}
            descriptor = {
                "selector": selector,
                "role": element.role or metadata.get("role") or "",
                "label": metadata.get("labelText") or element.text or metadata.get("ariaLabel") or "",
                "placeholder": metadata.get("placeholder") or "",
                "ownerDialogTitle": metadata.get("ownerDialogTitle") or "",
            }
            cleaned = {key: value for key, value in descriptor.items() if value}
            inputs.append(cleaned)
        if not inputs:
            return ""
        return json.dumps(inputs, ensure_ascii=False, indent=2)

def _extract_json_and_reason(content: str) -> tuple[str, Optional[str]]:
    if not content:
        return "", None
    trimmed = content.strip()
    reason = None
    # capture chain-of-thought if model provided 'reason' outside JSON
    if "Reason:" in trimmed:
        pre_reason, post_reason = trimmed.split("Reason:", 1)
        trimmed = pre_reason.strip()
        reason_candidate = post_reason.strip()
        if "{" in reason_candidate:
            before_json, after_json = reason_candidate.split("{", 1)
            reason = before_json.split("Result:", 1)[0].strip()
            trimmed += " {" + after_json
        else:
            reason = reason_candidate.split("Result:", 1)[0].strip()
    trimmed = _strip_json_prefix(trimmed)
    trimmed = _remove_code_fences(trimmed)
    trimmed = _strip_json_prefix(trimmed)
    trimmed = _remove_code_fences(trimmed)
    if trimmed.startswith("{") and trimmed.endswith("}"):
        return trimmed, reason
    start = trimmed.find("{")
    end = trimmed.rfind("}")
    if start != -1 and end != -1 and end > start:
        return trimmed[start : end + 1], reason
    return trimmed, reason


def _guard_against_login_navigation(action: PlannerAction) -> None:
    if action.action != "goto" or not action.url:
        return
    lowered = action.url.lower()
    login_tokens = ("login", "sign-in", "signin")
    if any(token in lowered for token in login_tokens):
        raise ValueError("Planner attempted to navigate to a login page; rejecting action")


def _remove_code_fences(text: str) -> str:
    if text.startswith("```"):
        fence = text.split("```")
        if len(fence) >= 3:
            return fence[1].strip()
        return text.lstrip("`")
    return text


def _strip_json_prefix(text: str) -> str:
    if not text:
        return ""
    lowered = text.lower()
    if lowered.startswith("json"):
        return text[4:].lstrip(": \n\t")
    return text


_INVALID_ESCAPE_FINDER = re.compile(r"\\([^\"\\/bfnrtu])")


def _sanitize_json_string(data: str) -> str:
    """Fix non-json escapes (Playwright CSS escaping like '\\ ') before parsing."""
    if not data or "\\" not in data:
        return data
    fixed = _INVALID_ESCAPE_FINDER.sub(r"\1", data)
    fixed = fixed.replace("\\ ", " ")
    return _escape_unquoted_quotes(fixed)


def _escape_unquoted_quotes(data: str) -> str:
    """Escape double quotes that appear inside string literals without backslashes."""
    result: List[str] = []
    in_string = False
    escaped = False
    bracket_depth = 0

    for idx, char in enumerate(data):
        if not in_string:
            if char == '"' and not escaped:
                in_string = True
                bracket_depth = 0
            result.append(char)
            escaped = char == "\\"
            continue

        if escaped:
            result.append(char)
            escaped = False
            continue

        if char == "\\":
            result.append(char)
            escaped = True
            continue

        if char == "[":
            bracket_depth += 1
            result.append(char)
            continue

        if char == "]" and bracket_depth:
            bracket_depth = max(0, bracket_depth - 1)
            result.append(char)
            continue

        if char == '"':
            # Look ahead to decide if this is the string terminator.
            next_idx = idx + 1
            while next_idx < len(data) and data[next_idx].isspace():
                next_idx += 1
            if bracket_depth == 0 and (next_idx >= len(data) or data[next_idx] in {",", "}", "]", ":"}):
                in_string = False
                result.append(char)
            else:
                result.append("\\\"")
        else:
            result.append(char)

    return "".join(result)


def _extract_keywords(task: str) -> List[str]:
    raw_tokens = re.findall(r"[a-zA-Z]+", task.lower())
    keywords = {token for token in raw_tokens if len(token) >= 4}
    return sorted(keywords)


def _describe_task_intent(task: str) -> str:
    lowered = task.lower()
    fragments: List[str] = []
    if any(word in lowered for word in ("open", "navigate", "view", "switch", "show", "go to")):
        fragments.append("open or switch a view")
    if any(word in lowered for word in ("create", "add", "new", "start", "compose")):
        fragments.append("create/add content")
    if "filter" in lowered or "show only" in lowered:
        fragments.append("apply a filter/search chip")
    if any(word in lowered for word in ("toggle", "enable", "disable", "turn on", "turn off", "switch")):
        fragments.append("toggle a control or setting")
    if any(word in lowered for word in ("search", "find", "look for")):
        fragments.append("type or search for text")
    if "command palette" in lowered or "quick find" in lowered:
        fragments.append("open the command palette / quick finder")
    if "share" in lowered:
        fragments.append("open/share a menu")
    if "capture" in lowered:
        fragments.append("capture the resulting modal/menu")
    ordered: List[str] = []
    for fragment in fragments:
        if fragment not in ordered:
            ordered.append(fragment)
    return ", ".join(ordered)


def _collect_keyword_hits(keywords: List[str], snapshot: UISnapshot) -> List[str]:
    hits: List[str] = []

    def add_hit(prefix: str, element) -> None:
        text = element.text or ""
        selector = element.selector or ""
        hits.append(f"{prefix}: text={text!r} selector={selector!r}")

    for idx, element in enumerate(snapshot.clickables):
        content = f"{element.text or ''} {element.selector or ''}".lower()
        if any(keyword in content for keyword in keywords):
            add_hit(f"clickable[{idx}]", element)

    for idx, element in enumerate(snapshot.inputs):
        content = f"{element.text or ''} {element.selector or ''}".lower()
        if any(keyword in content for keyword in keywords):
            add_hit(f"input[{idx}]", element)

    for idx, element in enumerate(snapshot.modals):
        content = f"{element.text or ''} {element.selector or ''}".lower()
        if any(keyword in content for keyword in keywords):
            add_hit(f"modal[{idx}]", element)

    return hits


def _format_elements(title: str, elements: List[Any]) -> List[str]:
    lines = ["", f"{title} (index | role | text -> selector):"]
    if not elements:
        lines.append("- none")
        return lines
    for idx, element in enumerate(elements):
        text = getattr(element, "text", "") or ""
        role = getattr(element, "role", "") or ""
        selector = getattr(element, "selector", "") or ""
        metadata = getattr(element, "metadata", {}) or {}
        highlights: List[str] = []
        state = metadata.get("state") or {}
        flagged = [key for key, value in state.items() if value and value is not False]
        if flagged:
            highlights.append("state=" + ",".join(str(item) for item in flagged[:3]))
        data_test = metadata.get("dataTestId")
        if data_test:
            highlights.append(f"data-testid={data_test}")
        owner_menu = metadata.get("ownerMenu")
        if owner_menu and owner_menu.get("label"):
            highlights.append(f"menu={owner_menu['label']}")
        if metadata.get("ancestry"):
            ancestor_labels = [ancestor.get("label") for ancestor in metadata["ancestry"] if ancestor.get("label")]
            if ancestor_labels:
                highlights.append("context=" + " > ".join(ancestor_labels[:2]))
        trail = f" ({'; '.join(highlights)})" if highlights else ""
        lines.append(f"- {idx}: {role} | {text} -> {selector}{trail}")
    return lines


def _suggest_artifact_name(task: str) -> Optional[str]:
    cleaned = _local_slugify(task)
    if not cleaned or cleaned == "task":
        return None
    words = [part.capitalize() for part in cleaned.split("-") if part]
    return " ".join(words[:5])


def _detect_creation_modal(
    keywords: List[str],
    snapshot: UISnapshot,
    suggested_name: Optional[str],
    task: str,
) -> Optional[str]:
    if not snapshot.modals:
        return None

    name_selector = None
    summary_selector = None
    create_selector = None

    modal_texts = " | ".join(filter(None, (modal.text for modal in snapshot.modals)))
    lower_modal = modal_texts.lower()
    if "project" in lower_modal:
        modal_type = "project"
    elif "issue" in lower_modal:
        modal_type = "issue"
    else:
        modal_type = None

    for element in snapshot.inputs:
        selector = (element.selector or "").lower()
        if not selector:
            continue
        if modal_type == "project" and "project" in selector and "name" in selector and not name_selector:
            name_selector = element.selector
        elif modal_type == "project" and "summary" in selector and not summary_selector:
            summary_selector = element.selector
        elif modal_type == "project" and "description" in selector and not summary_selector:
            summary_selector = element.selector
        elif modal_type == "issue" and "issue" in selector and "title" in selector and not name_selector:
            name_selector = element.selector
        elif modal_type == "issue" and "description" in selector and not summary_selector:
            summary_selector = element.selector

    for element in snapshot.clickables:
        content = f"{(element.text or '').lower()} {(element.selector or '').lower()}"
        if "create" in content and (
            (modal_type == "project" and "project" in content) or (modal_type == "issue" and "issue" in content)
        ):
            create_selector = element.selector
            break

    if modal_type is None:
        return None

    parts: List[str] = []
    if modal_type == "project":
        parts.append("Project creation modal detected.")
    elif modal_type == "issue":
        parts.append("Issue creation modal detected.")

    if name_selector and suggested_name:
        parts.append(f"Fill {name_selector!r} with '{suggested_name}'.")
    elif name_selector:
        parts.append(f"Fill {name_selector!r} with a descriptive name.")
    if summary_selector:
        summary_hint = _suggest_artifact_summary(suggested_name, task)
        if summary_hint:
            parts.append(f"Fill {summary_selector!r} with '{summary_hint}'.")
        else:
            parts.append(f"Describe the details via {summary_selector!r}.")
    if create_selector:
        parts.append(f"Submit using {create_selector!r} once fields are ready.")

    if len(parts) == 1:
        return None
    return " ".join(parts)


def _local_slugify(task: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", task.lower()).strip("-")
    parts = [part for part in cleaned.split("-") if part]
    return "-".join(parts) or "task"


def _suggest_artifact_summary(name: Optional[str], task: str) -> Optional[str]:
    if not name:
        name = "This item"
    short_task = " ".join(task.strip().split())
    if len(short_task) > 180:
        short_task = short_task[:177] + "..."
    return f"{name} captures: {short_task}"


def _detect_created_entries(snapshot: UISnapshot, suggested_name: Optional[str]) -> Optional[str]:
    candidates: List[str] = []
    if suggested_name:
        candidates.append(suggested_name.lower())
    candidates.extend(["automation project", "automation issue", "automation task"])

    for element in snapshot.clickables:
        text = (element.text or "").strip()
        if not text:
            continue
        lowered = text.lower()
        if any(keyword in lowered for keyword in candidates):
            return f"Found entry '{text}'. Consider verifying details and completing the task."
    return None


__all__ = ["LLMPlanner", "InvalidPlannerActionError"]
