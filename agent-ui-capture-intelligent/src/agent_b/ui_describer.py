"""Utilities to summarise the visible UI for the planner."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from playwright.async_api import Error as PlaywrightError, Frame, Locator, Page

from .models import UIElement, UISnapshot

CLICKABLE_LIMIT = 60
INPUT_LIMIT = 30
MODAL_LIMIT = 6
PRIMARY_LIMIT = 20
BREADCRUMB_LIMIT = 10
OVERLAY_LIMIT = 10
IFRAME_LIMIT = 4

ROLE_ORDER: Tuple[str, ...] = ("button", "link", "menuitem", "option")
CSS_CLICKABLE_SELECTOR = "button, [role=button], a[href], [tabindex], [aria-label], [data-testid], [data-test]"
CSS_MODAL_CLICKABLE_SELECTOR = (
    "[role='dialog'] button, [role='dialog'] [role='button'], "
    "[data-modal] button, [data-modal] [role='button'], dialog button"
)
CSS_INPUT_SELECTOR = "input, textarea, [contenteditable='true'], [role='textbox'], select"
CSS_MODAL_SELECTOR = "[role='dialog'], dialog, [data-modal], [data-overlay], [data-layer-kind='modal']"

PRIMARY_SELECTOR = (
    'button:has-text("New"), button:has-text("Create"), button:has-text("Add"), '
    "[data-testid*=\"create\"], [data-test-id*=\"create\"], "
    '[aria-label*="Create"], [aria-label*="New"], [aria-label*="Add"]'
)
OVERLAY_SELECTOR = "[data-animated-popover-backdrop], [data-layer-kind='modal-backdrop'], .ReactModal__Overlay"
BREADCRUMB_SELECTOR = "[aria-label*='breadcrumb'] a, nav[aria-label*='breadcrumb'] a, [data-testid*='breadcrumb'] a"

PREFERRED_LABELS: Tuple[str, ...] = (
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
)

_SERIALIZER_HELPERS = """
    const isElementVisible = (element) => {
        if (!element) return false;
        const style = window.getComputedStyle(element);
        if (!style) return false;
        if (style.visibility === "hidden" || style.display === "none" || Number(style.opacity) === 0) return false;
        const rect = element.getBoundingClientRect();
        if (!rect || rect.width < 1 || rect.height < 1) return false;
        return true;
    };

    const preferredDataAttr = (element) => {
        const attrs = ["data-testid", "data-test-id", "data-test", "data-qa"];
        for (const attr of attrs) {
            const value = element.getAttribute(attr);
            if (value) {
                return { attr, value };
            }
        }
        return null;
    };

    const collectAncestry = (element) => {
        const ancestry = [];
        let current = element ? element.parentElement : null;
        while (current && ancestry.length < 6) {
            const role = current.getAttribute("role");
            const label =
                current.getAttribute("aria-label") ||
                current.getAttribute("data-testid") ||
                current.getAttribute("data-test-id") ||
                current.getAttribute("data-qa");
            const classes = current.className || "";
            if (role || label || classes) {
                ancestry.push({
                    tag: current.tagName ? current.tagName.toLowerCase() : null,
                    role: role || null,
                    label: label || null,
                    classes: classes || null,
                });
            }
            current = current.parentElement;
        }
        return ancestry;
    };

    const getAriaBoolean = (value) => {
        if (value === null || value === undefined) return null;
        if (value === "mixed") return "mixed";
        if (value === "true" || value === true) return true;
        if (value === "false" || value === false) return false;
        return null;
    };

    const computeStateFlags = (element) => {
        const selectedAria = getAriaBoolean(element.getAttribute("aria-selected"));
        const pressedAria = getAriaBoolean(element.getAttribute("aria-pressed"));
        return {
            disabled: element.hasAttribute("disabled") ? true : getAriaBoolean(element.getAttribute("aria-disabled")),
            selected: selectedAria !== null ? selectedAria : pressedAria,
            expanded: getAriaBoolean(element.getAttribute("aria-expanded")),
            checked: getAriaBoolean(element.getAttribute("aria-checked")),
            busy: getAriaBoolean(element.getAttribute("aria-busy")),
            readonly: getAriaBoolean(element.getAttribute("aria-readonly")),
        };
    };

    const nearestContext = (element, selectorList) => {
        for (const selector of selectorList) {
            const owner = element.closest(selector);
            if (owner) {
                return {
                    selector,
                    role: owner.getAttribute("role") || null,
                    label: owner.getAttribute("aria-label") || null,
                    dataTestId: owner.getAttribute("data-testid") || owner.getAttribute("data-test-id") || null,
                };
            }
        }
        return null;
    };

    const escapeForAttr = (value) => {
        if (!value) return "";
        if (window.CSS && typeof window.CSS.escape === "function") {
            return window.CSS.escape(value);
        }
        return value.replace(/["\\\\]/g, "\\\\$&");
    };

    const collectLabelCandidates = (element) => {
        const texts = [];
        const aria = element.getAttribute("aria-label");
        if (aria) {
            texts.push(aria);
        }
        const labelledBy = element.getAttribute("aria-labelledby");
        if (labelledBy) {
            labelledBy
                .split(/\\s+/)
                .filter(Boolean)
                .forEach((id) => {
                    const ref = document.getElementById(id);
                    const text = ((ref && (ref.innerText || ref.textContent)) || "").replace(/\\s+/g, " ").trim();
                    if (text) {
                        texts.push(text);
                    }
                });
        }
        if (element.labels && typeof element.labels.length === "number") {
            Array.from(element.labels).forEach((label) => {
                const text = (label.innerText || label.textContent || "").replace(/\\s+/g, " ").trim();
                if (text) {
                    texts.push(text);
                }
            });
        }
        const elementId = element.id;
        if (elementId) {
            const escaped = escapeForAttr(elementId);
            if (escaped) {
                const matches = document.querySelectorAll(`label[for="${escaped}"]`);
                matches.forEach((label) => {
                    const text = (label.innerText || label.textContent || "").replace(/\\s+/g, " ").trim();
                    if (text) {
                        texts.push(text);
                    }
                });
            }
        }
        return texts;
    };

    const previousLabelText = (element) => {
        let current = element.previousElementSibling;
        let steps = 0;
        while (current && steps < 3) {
            const text = (current.innerText || current.textContent || "").replace(/\\s+/g, " ").trim();
            if (text) {
                return text;
            }
            current = current.previousElementSibling;
            steps += 1;
        }
        return null;
    };

    const ancestorLabelText = (element) => {
        const container = element.closest("div, section, form, article");
        if (!container) {
            return null;
        }
        const heading = container.querySelector("h1, h2, h3, h4, label, strong, span");
        if (!heading) {
            return null;
        }
        const text = (heading.innerText || heading.textContent || "").replace(/\\s+/g, " ").trim();
        return text || null;
    };

    const deriveOwnerDialogTitle = (element) => {
        const dialog = element.closest("[role='dialog'], [data-modal], dialog");
        if (!dialog) {
            return null;
        }
        const labelledBy = dialog.getAttribute("aria-labelledby");
        if (labelledBy) {
            for (const id of labelledBy.split(/\\s+/)) {
                const header = document.getElementById(id);
                const text = (header && (header.innerText || header.textContent) || "").replace(/\\s+/g, " ").trim();
                if (text) {
                    return text;
                }
            }
        }
        const aria = dialog.getAttribute("aria-label");
        if (aria) {
            return aria;
        }
        const heading = dialog.querySelector("h1, h2, h3, h4");
        if (heading) {
            const text = (heading.innerText || heading.textContent || "").replace(/\\s+/g, " ").trim();
            if (text) {
                return text;
            }
        }
        return null;
    };

    const deriveInputLabel = (element) => {
        const explicit = collectLabelCandidates(element).find(Boolean);
        if (explicit) {
            return explicit;
        }
        const sibling = previousLabelText(element);
        if (sibling) {
            return sibling;
        }
        const ancestor = ancestorLabelText(element);
        if (ancestor) {
            return ancestor;
        }
        const placeholder = element.getAttribute("placeholder");
        if (placeholder) {
            return placeholder;
        }
        const nameAttr = element.getAttribute("name");
        if (nameAttr) {
            return nameAttr;
        }
        return null;
    };

    const roleFor = (element) => {
        const explicit = element.getAttribute("role");
        if (explicit) {
            return explicit;
        }
        const tag = element.tagName ? element.tagName.toLowerCase() : "";
        if (tag === "button" || tag === "summary") {
            return "button";
        }
        if (tag === "a" && element.getAttribute("href")) {
            return "link";
        }
        if (tag === "input") {
            const typeAttr = element.getAttribute("type") || "text";
            if (["checkbox", "radio"].includes(typeAttr)) {
                return typeAttr;
            }
            return "textbox";
        }
        if (tag === "textarea") {
            return "textbox";
        }
        if (tag === "select") {
            return "listbox";
        }
        return null;
    };

    const serializeElement = (element) => {
        if (!(element instanceof HTMLElement)) {
            return null;
        }
        if (!isElementVisible(element)) {
            return null;
        }
        const tag = element.tagName ? element.tagName.toLowerCase() : null;
        let nth = 0;
        if (tag) {
            try {
                const matches = document.querySelectorAll(tag);
                for (let idx = 0; idx < matches.length; idx += 1) {
                    if (matches[idx] === element) {
                        nth = idx;
                        break;
                    }
                }
            } catch (err) {
                nth = 0;
            }
        }
        const metadata = {
            tag,
            id: element.id || null,
            href: element.getAttribute("href") || null,
            ariaLabel: element.getAttribute("aria-label") || null,
            placeholder: element.getAttribute("placeholder") || null,
            name: element.getAttribute("name") || null,
            type: element.getAttribute("type") || null,
            classes: element.className || null,
            dataTestId: element.getAttribute("data-testid") || element.getAttribute("data-test-id") || null,
            ariaDisabled: element.getAttribute("aria-disabled"),
            ariaPressed: element.getAttribute("aria-pressed"),
            ariaHasPopup: element.getAttribute("aria-haspopup"),
            contentEditable: element.isContentEditable || null,
            state: computeStateFlags(element),
            ancestry: collectAncestry(element),
            ownerMenu: nearestContext(element, ["[role='menu']", "[data-overlay-container='true'] [role='menu']"]),
            ownerDialog: nearestContext(element, ["[role='dialog']", "[data-modal]", "dialog"]),
            labelText: deriveInputLabel(element),
            ownerDialogTitle: deriveOwnerDialogTitle(element),
        };
        const dataset = element.dataset || {};
        const entries = Object.entries(dataset).slice(0, 10);
        if (entries.length) {
            metadata.dataset = Object.fromEntries(entries);
        }
        return {
            text: ((element.innerText || element.textContent || "").replace(/\\s+/g, " ").trim()) || null,
            role: roleFor(element),
            tag,
            nth,
            ariaLabel: element.getAttribute("aria-label") || null,
            dataAttr: preferredDataAttr(element),
            metadata,
        };
    };
"""

_SERIALIZE_SCRIPT = f"""
    (element) => {{
        {_SERIALIZER_HELPERS}
        return serializeElement(element);
    }}
"""

_JS_FALLBACK_SCRIPT = f"""
    (config) => {{
        {_SERIALIZER_HELPERS}
        const limit = Math.max(0, config?.limit || 0);
        const root = document.body || document.documentElement;
        if (!root || !limit) {{
            return [];
        }}
        const walker = document.createTreeWalker(root, NodeFilter.SHOW_ELEMENT);
        const results = [];
        const seenSelectors = new Set();
        while (results.length < limit) {{
            const node = walker.nextNode();
            if (!node) {{
                break;
            }}
            if (!(node instanceof HTMLElement)) {{
                continue;
            }}
            if (!node.matches("button, a[href], [role], [tabindex], summary, [data-testid], [data-test-id], [data-test], [aria-label]")) {{
                if (typeof node.onclick !== "function") {{
                    continue;
                }}
            }}
            if (node.getAttribute("role") === "presentation" || node.getAttribute("aria-hidden") === "true") {{
                continue;
            }}
            const info = serializeElement(node);
            if (!info) {{
                continue;
            }}
            const key = info.metadata?.id || `${{info.tag}}|${{info.text}}|${{info.ariaLabel}}`;
            if (seenSelectors.has(key)) {{
                continue;
            }}
            seenSelectors.add(key);
            results.push(info);
        }}
        return results.slice(0, limit);
    }}
"""

_SUPPLEMENTARY_SCRIPT = f"""
    (limits) => {{
        {_SERIALIZER_HELPERS}
        const collect = (selector, limit) => {{
            if (!selector || !limit) {{
                return [];
            }}
            const nodes = Array.from(document.querySelectorAll(selector));
            const output = [];
            for (const node of nodes) {{
                if (output.length >= limit) {{
                    break;
                }}
                const info = serializeElement(node);
                if (info) {{
                    output.push(info);
                }}
            }}
            return output;
        }};
        return {{
            primary: collect({json.dumps(PRIMARY_SELECTOR)}, limits?.primary || 0),
            overlays: collect({json.dumps(OVERLAY_SELECTOR)}, limits?.overlays || 0),
            breadcrumbs: collect({json.dumps(BREADCRUMB_SELECTOR)}, limits?.breadcrumbs || 0),
        }};
    }}
"""


def _rx_escape(value: str) -> str:
    return value.replace("/", r"\/").replace("[", r"\[").replace("]", r"\]")


def _attr_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _apply_prefix(selector: str, prefix: str) -> str:
    return f"{prefix}{selector}" if prefix else selector


async def describe_ui(page: Page) -> UISnapshot:
    """Return a structured snapshot of the current UI."""
    url = page.url
    try:
        title = await page.title()
    except PlaywrightError:
        title = None

    clickables: List[UIElement] = []
    inputs: List[UIElement] = []
    modals: List[UIElement] = []
    primary_actions: List[UIElement] = []
    overlays: List[UIElement] = []
    breadcrumbs: List[UIElement] = []

    frames: List[Frame] = []
    main_frame = page.main_frame
    frames.append(main_frame)
    iframe_candidates: List[Frame] = []
    for frame in page.frames:
        if frame is main_frame:
            continue
        try:
            parent = getattr(frame, "parent")
        except Exception:  # noqa: BLE001
            parent = None
        if parent is not main_frame:
            continue
        if frame not in iframe_candidates:
            iframe_candidates.append(frame)
        if len(iframe_candidates) >= IFRAME_LIMIT:
            break
    frames.extend(iframe_candidates)

    seen_clickables: Set[str] = set()
    seen_inputs: Set[str] = set()
    seen_modals: Set[str] = set()

    for idx, frame in enumerate(frames):
        prefix = "" if frame is main_frame else f"frame[{idx - 1}]:"

        remaining_clickables = CLICKABLE_LIMIT - len(clickables)
        if remaining_clickables > 0:
            clickables.extend(await _collect_clickables(frame, prefix, remaining_clickables, seen_clickables))

        remaining_inputs = INPUT_LIMIT - len(inputs)
        if remaining_inputs > 0:
            inputs.extend(await _collect_inputs(frame, prefix, remaining_inputs, seen_inputs))

        remaining_modals = MODAL_LIMIT - len(modals)
        if remaining_modals > 0:
            modals.extend(await _collect_modals(frame, prefix, remaining_modals, seen_modals))

        supplemental = await _collect_supplementary(frame, prefix)
        _extend_elements(primary_actions, supplemental.get("primary", []), prefix, "primary", PRIMARY_LIMIT)
        _extend_elements(overlays, supplemental.get("overlays", []), prefix, "overlay", OVERLAY_LIMIT)
        _extend_elements(breadcrumbs, supplemental.get("breadcrumbs", []), prefix, "breadcrumb", BREADCRUMB_LIMIT)

    _prioritize_preferred(clickables)
    detected_app = _detect_app(url, clickables, breadcrumbs)

    return UISnapshot(
        url=url,
        title=title,
        clickables=clickables[:CLICKABLE_LIMIT],
        inputs=inputs[:INPUT_LIMIT],
        modals=modals[:MODAL_LIMIT],
        primary_actions=primary_actions[:PRIMARY_LIMIT],
        overlays=overlays[:OVERLAY_LIMIT],
        breadcrumbs=breadcrumbs[:BREADCRUMB_LIMIT],
        detected_app=detected_app,
    )


async def _collect_clickables(frame: Frame, prefix: str, limit: int, seen: Set[str]) -> List[UIElement]:
    results: List[UIElement] = []
    remaining = limit

    if remaining > 0:
        modal_scoped = await _collect_modal_scoped_clickables(frame, prefix, remaining, seen)
        results.extend(modal_scoped)
        remaining = limit - len(results)

    for role_name in ROLE_ORDER:
        if remaining <= 0:
            break
        locator = frame.get_by_role(role_name)
        batch = await _collect_by_locator(
            locator,
            prefix,
            remaining,
            seen,
            kind="clickable",
            role_hint=role_name,
            require_enabled=False,
        )
        results.extend(batch)
        remaining = limit - len(results)

    if remaining > 0:
        locator = frame.locator(CSS_CLICKABLE_SELECTOR)
        batch = await _collect_by_locator(
            locator,
            prefix,
            remaining,
            seen,
            kind="clickable",
            role_hint=None,
            require_enabled=False,
        )
        results.extend(batch)
        remaining = limit - len(results)

    if remaining > 0:
        fallback = await _collect_clickable_fallback(frame, prefix, remaining, seen)
        results.extend(fallback)

    return results


async def _collect_inputs(frame: Frame, prefix: str, limit: int, seen: Set[str]) -> List[UIElement]:
    locator = frame.locator(CSS_INPUT_SELECTOR)
    return await _collect_by_locator(locator, prefix, limit, seen, kind="input", role_hint=None, require_enabled=False)


async def _collect_modals(frame: Frame, prefix: str, limit: int, seen: Set[str]) -> List[UIElement]:
    locator = frame.locator(CSS_MODAL_SELECTOR)
    return await _collect_by_locator(locator, prefix, limit, seen, kind="modal", role_hint="dialog", require_enabled=False)


async def _collect_modal_scoped_clickables(frame: Frame, prefix: str, limit: int, seen: Set[str]) -> List[UIElement]:
    if limit <= 0:
        return []
    locator = frame.locator(CSS_MODAL_CLICKABLE_SELECTOR)
    return await _collect_by_locator(
        locator,
        prefix,
        limit,
        seen,
        kind="clickable",
        role_hint=None,
        require_enabled=False,
    )


async def _collect_clickable_fallback(frame: Frame, prefix: str, limit: int, seen: Set[str]) -> List[UIElement]:
    if limit <= 0:
        return []
    try:
        payload = await frame.evaluate(_JS_FALLBACK_SCRIPT, {"limit": limit * 2})
    except PlaywrightError:
        return []
    results: List[UIElement] = []
    if not isinstance(payload, Iterable):
        return results
    for entry in payload:
        if len(results) >= limit:
            break
        if not isinstance(entry, dict):
            continue
        element = _create_ui_element(entry, entry.get("role"), prefix, "clickable")
        if not element or not element.selector:
            continue
        if element.selector in seen:
            continue
        seen.add(element.selector)
        results.append(element)
    return results


async def _collect_supplementary(frame: Frame, prefix: str) -> Dict[str, List[Dict[str, Any]]]:
    try:
        payload = await frame.evaluate(
            _SUPPLEMENTARY_SCRIPT,
            {"primary": PRIMARY_LIMIT, "overlays": OVERLAY_LIMIT, "breadcrumbs": BREADCRUMB_LIMIT},
        )
    except PlaywrightError:
        return {"primary": [], "overlays": [], "breadcrumbs": []}
    if not isinstance(payload, dict):
        return {"primary": [], "overlays": [], "breadcrumbs": []}
    return {
        "primary": payload.get("primary") or [],
        "overlays": payload.get("overlays") or [],
        "breadcrumbs": payload.get("breadcrumbs") or [],
    }


def _extend_elements(
    target: List[UIElement],
    payload: List[Dict[str, Any]],
    prefix: str,
    kind: str,
    limit: int,
) -> None:
    for entry in payload:
        if len(target) >= limit:
            break
        if not isinstance(entry, dict):
            continue
        element = _create_ui_element(entry, entry.get("role"), prefix, kind)
        if element:
            target.append(element)


async def _collect_by_locator(
    locator: Locator,
    prefix: str,
    limit: int,
    seen: Set[str],
    *,
    kind: str,
    role_hint: Optional[str],
    require_enabled: bool = False,
) -> List[UIElement]:
    results: List[UIElement] = []
    if limit <= 0:
        return results

    try:
        count = await locator.count()
    except PlaywrightError:
        return results

    for idx in range(count):
        if len(results) >= limit:
            break
        candidate = locator.nth(idx)
        try:
            if not await candidate.is_visible():
                continue
        except PlaywrightError:
            continue
        if require_enabled:
            try:
                if not await candidate.is_enabled():
                    continue
            except PlaywrightError:
                pass
        try:
            handle = await candidate.element_handle()
        except PlaywrightError:
            continue
        if not handle:
            continue

        info = await _serialize_handle(handle)
        try:
            await handle.dispose()
        except Exception:  # noqa: BLE001
            pass
        if not info:
            continue

        state = (info.get("metadata") or {}).get("state") or {}
        if kind != "modal" and state.get("disabled") is True:
            continue

        element = _create_ui_element(info, role_hint, prefix, kind)
        if not element or not element.selector:
            continue
        if element.selector in seen:
            continue
        seen.add(element.selector)
        results.append(element)

    return results


async def _serialize_handle(handle) -> Optional[Dict[str, Any]]:
    try:
        payload = await handle.evaluate(_SERIALIZE_SCRIPT)
    except PlaywrightError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _create_ui_element(
    info: Dict[str, Any],
    role_hint: Optional[str],
    prefix: str,
    kind: str,
) -> Optional[UIElement]:
    selector = _choose_selector(info, role_hint, kind)
    if not selector:
        return None
    selector = _apply_prefix(selector, prefix)
    if kind == "input":
        text_value = _input_label(info)
    else:
        text_value = info.get("text")
    if isinstance(text_value, str):
        text_value = text_value[:160]
    role_value = role_hint or info.get("role")
    metadata = info.get("metadata") or {}
    return UIElement(
        text=text_value,
        role=role_value,
        selector=selector,
        kind=kind,
        metadata=dict(metadata),
    )


def _choose_selector(info: Dict[str, Any], role_hint: Optional[str], kind: str) -> Optional[str]:
    metadata = info.get("metadata") or {}
    tag = info.get("tag")
    nth = info.get("nth")
    nth_selector = f"{tag} >> nth={nth}" if tag is not None and nth is not None else None

    if kind == "input":
        element_id = metadata.get("id")
        if element_id:
            return f"#{_attr_escape(element_id)}"
        name_attr = metadata.get("name")
        if name_attr:
            return f"[name=\"{_attr_escape(name_attr)}\"]"
        placeholder = metadata.get("placeholder")
        if placeholder:
            return f"[placeholder=\"{_attr_escape(placeholder)}\"]"
        aria_label = info.get("ariaLabel")
        if aria_label:
            return f"[aria-label=\"{_attr_escape(aria_label)}\"]"
        data_attr = info.get("dataAttr") or {}
        attr_name = data_attr.get("attr")
        attr_value = data_attr.get("value")
        if attr_name and attr_value:
            return f"[{attr_name}=\"{_attr_escape(attr_value)}\"]"
        if nth_selector:
            return nth_selector
        return None

    role_value = role_hint or info.get("role")
    text_candidate = _clean_selector_text(info.get("text"))
    aria_label = _clean_selector_text(info.get("ariaLabel"))
    name_candidate = text_candidate or aria_label

    if role_value and name_candidate:
        trimmed = name_candidate[:80]
        if _selector_payload_safe(trimmed):
            return f"role={role_value}[name=/{_rx_escape(trimmed)}/i]"

    if text_candidate:
        trimmed_text = text_candidate[:80]
        if _selector_payload_safe(trimmed_text):
            return f"text=/{_rx_escape(trimmed_text)}/i"

    data_attr = info.get("dataAttr") or {}
    attr_name = data_attr.get("attr")
    attr_value = data_attr.get("value")
    if attr_name and attr_value:
        return f"[{attr_name}=\"{_attr_escape(attr_value)}\"]"

    if aria_label:
        trimmed_label = aria_label[:120]
        return f"[aria-label=\"{_attr_escape(trimmed_label)}\"]"

    element_id = metadata.get("id")
    if element_id:
        return f"#{_attr_escape(element_id)}"

    if nth_selector:
        return nth_selector
    return None


def _clean_selector_text(value: Optional[str], *, limit: int = 120) -> Optional[str]:
    if not value:
        return None
    collapsed = re.sub(r"\s+", " ", value).strip()
    if not collapsed:
        return None
    return collapsed[:limit]


def _selector_payload_safe(value: str) -> bool:
    lowered = value.lower()
    if any(token in lowered for token in ("function", "=>", "var ", "const ", "let ", "return", "window.")):
        return False
    return True


# NOTE: Planner failures were traced to fabricated selectors such as
# role=textbox[name=/Project name/i]. By emitting real selectors plus a
# derived human-readable label, planners can now target actual inputs while
# reasoning about the label text shown on screen.
def _input_label(info: Dict[str, Any]) -> Optional[str]:
    metadata = info.get("metadata") or {}
    label = metadata.get("labelText")
    modal_title = metadata.get("ownerDialogTitle")
    aria_label = info.get("ariaLabel")
    placeholder = metadata.get("placeholder")
    name_attr = metadata.get("name")

    candidates = [
        label,
        aria_label,
        placeholder,
        name_attr,
        info.get("text"),
    ]

    text_value = next((candidate for candidate in candidates if candidate), None)
    if modal_title and text_value:
        return f"{modal_title.strip()} · {str(text_value).strip()}"
    if text_value:
        return str(text_value).strip()
    return None


def _detect_app(url: str, clickables: Sequence[UIElement], breadcrumbs: Sequence[UIElement]) -> Optional[str]:
    lowered = (url or "").lower()
    if "linear.app" in lowered:
        return "linear"
    if "notion.so" in lowered:
        return "notion"
    tokens: List[str] = []
    for element in list(clickables)[:10] + list(breadcrumbs)[:5]:
        if element.text:
            tokens.append(element.text.lower())
    joined = " ".join(tokens)
    if "linear" in joined:
        return "linear"
    if "notion" in joined:
        return "notion"
    if lowered and lowered not in {"about:blank", "chrome://new-tab-page"}:
        return "generic"
    return None


def _prioritize_preferred(clickables: List[UIElement]) -> None:
    ranked: List[Tuple[int, int, UIElement]] = []
    for idx, element in enumerate(clickables):
        ranked.append((_synonym_score(element.text), idx, element))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    clickables[:] = [element for _, _, element in ranked]


def _synonym_score(text: Optional[str]) -> int:
    if not text:
        return 0
    lowered = text.lower()
    for idx, label in enumerate(PREFERRED_LABELS):
        if label in lowered:
            return 100 - idx
    return 0
