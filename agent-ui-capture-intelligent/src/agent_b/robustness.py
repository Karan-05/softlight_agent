"""Robust interaction utilities shared across the executor."""

from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable, List, Optional, Sequence, TypeVar

from playwright.async_api import Error as PlaywrightError, Page, TimeoutError as PlaywrightTimeoutError

T = TypeVar("T")

DEFAULT_BACKOFFS_MS: Sequence[int] = (300, 700, 1500)

logger = logging.getLogger(__name__)


class NonFillableElementError(RuntimeError):
    """Raised when a fill action targets a non-textual element."""


async def wait_for_page_quiet(page: Page, timeout_ms: int) -> None:
    """Best-effort wait for the page to settle before planning the next action."""
    try:
        await page.wait_for_load_state("domcontentloaded", timeout=timeout_ms)
    except PlaywrightTimeoutError:
        pass
    except PlaywrightError:
        pass

    idle_script = """
        () => {
            const w = window;
            if (!w.__agentMutationIdle) {
                w.__agentMutationIdle = { last: Date.now() };
                const observer = new MutationObserver(() => {
                    w.__agentMutationIdle.last = Date.now();
                });
                observer.observe(document.documentElement, { subtree: true, childList: true, attributes: true });
            }
            return Date.now() - w.__agentMutationIdle.last > 400;
        }
    """
    try:
        await page.wait_for_function(idle_script, timeout=timeout_ms)
    except PlaywrightTimeoutError:
        pass
    except PlaywrightError:
        pass


async def _with_retries(
    async_op: Callable[[], Awaitable[T]],
    retries: int,
    backoffs_ms: Optional[Sequence[int]] = None,
) -> T:
    attempts = max(1, retries)
    delays = list(backoffs_ms or DEFAULT_BACKOFFS_MS)
    last_error: Optional[Exception] = None

    for attempt in range(attempts):
        try:
            return await async_op()
        except Exception as exc:  # noqa: BLE001 - propagate final failure
            last_error = exc
            if attempt == attempts - 1:
                break
            delay = delays[attempt] if attempt < len(delays) else delays[-1]
            await asyncio.sleep(delay / 1000.0)

    if last_error:
        raise last_error
    raise RuntimeError("async_op completed without returning a value")


async def click_robust(page: Page, selector: str, timeout_ms: int, retries: int) -> None:
    """Click an element reliably by re-querying and escalating interaction strategies."""

    async def attempt() -> None:
        locator = page.locator(selector).first
        await locator.wait_for(state="visible", timeout=timeout_ms)
        if not await locator.is_enabled():
            raise RuntimeError(f"Element {selector} is disabled")

        handle = await locator.element_handle()
        box_center = None
        if handle:
            try:
                await handle.scroll_into_view_if_needed(timeout=timeout_ms)
            except PlaywrightError:
                pass
            try:
                box = await handle.bounding_box()
                if box:
                    box_center = (box["x"] + box["width"] / 2.0, box["y"] + box["height"] / 2.0)
            except PlaywrightError:
                pass

        try:
            await locator.click(timeout=timeout_ms)
            return
        except PlaywrightTimeoutError:
            pass
        except PlaywrightError:
            pass

        if box_center:
            try:
                await page.mouse.move(box_center[0], box_center[1])
                await page.mouse.click(box_center[0], box_center[1], delay=20)
                return
            except PlaywrightError:
                pass

        await locator.click(timeout=timeout_ms, force=True)

    await _with_retries(attempt, retries=retries)


async def fill_robust(page: Page, selector: str, value: str, timeout_ms: int, retries: int) -> bool:
    """Fill an input reliably, clearing existing text and verifying the result. Returns False when the selector is not text-compatible."""

    locator = page.locator(selector).first
    await locator.wait_for(state="visible", timeout=timeout_ms)
    if not await locator.is_enabled():
        raise RuntimeError(f"Input {selector} is disabled")
    info = await _describe_element(locator)
    if not _element_supports_text_entry(info):
        tag = info.get("tag")
        input_type = info.get("type")
        role = info.get("role")
        editable = info.get("contentEditable")
        logger.warning(
            "telemetry:non_text_input selector=%s tag=%s type=%s role=%s editable=%s",
            selector,
            tag,
            input_type,
            role,
            editable,
        )
        return False

    async def attempt() -> None:
        handle = await locator.element_handle()
        if handle:
            try:
                await handle.scroll_into_view_if_needed(timeout=timeout_ms)
            except PlaywrightError:
                pass

        try:
            await locator.click(timeout=timeout_ms)
        except PlaywrightError:
            pass

        modifier = "Control"
        try:
            platform = await page.evaluate("() => navigator.platform || ''")
            if isinstance(platform, str) and platform.lower().startswith("mac"):
                modifier = "Meta"
        except PlaywrightError:
            pass

        try:
            await page.keyboard.press(f"{modifier}+A")
            await page.keyboard.press("Backspace")
        except PlaywrightError:
            pass

        try:
            await locator.fill(value, timeout=timeout_ms)
        except PlaywrightError:
            await locator.type(value, timeout=timeout_ms)

        try:
            current = await locator.input_value(timeout=timeout_ms)
        except PlaywrightError:
            current = None

        if current is not None and current.strip() != value.strip():
            raise RuntimeError("Input value did not match expected text")

    await _with_retries(attempt, retries=retries)
    return True


async def _describe_element(locator) -> dict:
    try:
        return await locator.evaluate(
            """(el) => ({
                tag: el.tagName ? el.tagName.toLowerCase() : "",
                type: el.type || "",
                role: el.getAttribute("role") || "",
                contentEditable: el.isContentEditable || false
            })"""
        )
    except Exception:  # noqa: BLE001
        return {}


def _element_supports_text_entry(info: dict) -> bool:
    tag = (info.get("tag") or "").lower()
    input_type = (info.get("type") or "").lower()
    role = (info.get("role") or "").lower()
    content_editable = bool(info.get("contentEditable"))
    if content_editable or role == "textbox":
        return True
    if tag == "textarea":
        return True
    if tag != "input":
        return False
    allowed = {"", "text", "search", "email", "url", "tel", "password", "number"}
    return input_type in allowed
