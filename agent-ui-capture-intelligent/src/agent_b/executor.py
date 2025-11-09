"""Execute planner actions inside Playwright and capture each step."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Callable, List, Optional

from playwright.async_api import Error as PlaywrightError, Page

from .capturer import capture_state
from .models import PlannerAction
from .self_heal import mark_selector_failure, mark_selector_success
from .robustness import NonFillableElementError, click_robust, fill_robust, wait_for_page_quiet

logger = logging.getLogger(__name__)


async def run_step(
    page: Page,
    action: PlannerAction,
    out_dir: Path,
    step_idx: int,
    timeout_ms: int,
    retries: int,
    log_tail_provider: Optional[Callable[[], List[dict]]] = None,
) -> bool:
    """Execute a single planner action and capture the resulting UI state."""
    capture_label = _slugify(action.capture_name or action.action)
    capture_id = f"step-{step_idx:02d}-{capture_label}"
    metadata = {"action": action.model_dump()}
    success = True

    if (action.capture_name or "").lower() == "modal-submit":
        logger.debug(
            "Executing modal submit selector=%s reason=%s",
            action.selector,
            action.reason,
        )

    try:
        await _execute_action(page, action, timeout_ms, retries)
        await wait_for_page_quiet(page, timeout_ms)
        if action.expect:
            expect_selector = _normalize_expect_selector(action.expect)
            if expect_selector:
                await page.wait_for_selector(expect_selector, timeout=timeout_ms)
            else:
                logger.debug("Skipping non-selector expect hint: %s", action.expect)
    except Exception as exc:  # noqa: BLE001 - capture unexpected failures
        success = False
        metadata["error"] = str(exc)
        logger.warning("Action %s failed: %s", action.action, exc)
    finally:
        logs: Optional[List[dict]] = None
        if log_tail_provider:
            try:
                logs = log_tail_provider()
            except Exception as exc:  # noqa: BLE001
                metadata["log_error"] = str(exc)
        await capture_state(page, out_dir, capture_id, extra={**metadata, "success": success}, log_tail=logs)

    if action.selector:
        if success:
            mark_selector_success(action.selector)
        else:
            mark_selector_failure(action.selector)

    return success


async def _execute_action(page: Page, action: PlannerAction, timeout_ms: int, retries: int) -> None:
    if action.action == "goto":
        if not action.url:
            raise ValueError("goto action requires 'url'")
        if _should_skip_login_navigation(page, action.url):
            logger.debug("Skipping navigation to %s because session appears authenticated.", action.url)
            return
        await page.goto(action.url, wait_until="domcontentloaded", timeout=max(timeout_ms, 15000))
        return

    if action.action == "click":
        if not action.selector:
            raise ValueError("click action requires 'selector'")
        await click_robust(page, action.selector, timeout_ms=timeout_ms, retries=retries)
        return

    if action.action == "fill":
        if not action.selector:
            raise ValueError("fill action requires 'selector'")
        if action.value is None:
            raise ValueError("fill action requires 'value'")
        try:
            success = await fill_robust(page, action.selector, action.value, timeout_ms=timeout_ms, retries=retries)
            if not success:
                raise NonFillableElementError(action.selector)
        except NonFillableElementError:
            raise
        except Exception as exc:
            fallback_selector = await _fallback_input_selector(page, action.selector)
            if fallback_selector and fallback_selector != action.selector:
                success = await fill_robust(page, fallback_selector, action.value, timeout_ms=timeout_ms, retries=retries)
                if not success:
                    raise NonFillableElementError(fallback_selector)
            else:
                raise exc
        return

    if action.action == "wait_for":
        if not action.selector:
            raise ValueError("wait_for action requires 'selector'")
        await page.wait_for_selector(action.selector, timeout=timeout_ms)
        return

    if action.action == "capture":
        return

    if action.action == "press":
        if not action.value:
            raise ValueError("press action requires 'value' with the key combination")
        await page.keyboard.press(action.value)
        await page.wait_for_timeout(120)
        return

    if action.action == "done":
        return

    raise ValueError(f"Unsupported action: {action.action}")


def _slugify(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()
    return text or "action"


def _should_skip_login_navigation(page: Page, target_url: str) -> bool:
    lowered = target_url.lower()
    login_tokens = ("/login", "login?", "signin", "sign-in")
    if not any(token in lowered for token in login_tokens):
        return False
    try:
        current = (page.url or "").lower()
    except Exception:  # noqa: BLE001
        current = ""
    if not current:
        return False
    if any(token in current for token in login_tokens):
        return False
    return True


def _normalize_expect_selector(expect: str) -> Optional[str]:
    candidate = expect.strip()
    if not candidate:
        return None
    lowered = candidate.lower()
    if lowered.startswith(("text=", "css=", "xpath=", "id=", "role=")):
        return candidate

    selector_hint_chars = {"#", ".", "[", "]", ":", ">", "=", "\\", '"', "'"}
    if any(ch in candidate for ch in selector_hint_chars):
        if _looks_like_natural_language(candidate):
            return None
        return candidate

    if " " not in candidate:
        return candidate

    return None


_NATURAL_LANGUAGE_PATTERN = re.compile(r"[a-zA-Z]{3,}")


def _looks_like_natural_language(candidate: str) -> bool:
    tokens = _NATURAL_LANGUAGE_PATTERN.findall(candidate)
    return len(tokens) >= 3


async def _fallback_input_selector(page: Page, original_selector: Optional[str]) -> Optional[str]:
    try:
        from .ui_describer import describe_ui
    except ImportError:
        return None

    try:
        snapshot = await describe_ui(page)
    except PlaywrightError:
        return None

    for element in snapshot.inputs:
        if not element.selector or element.selector == original_selector:
            continue
        metadata = element.metadata or {}
        if metadata.get("ownerDialog"):
            return element.selector
    for element in snapshot.inputs:
        if element.selector and element.selector != original_selector:
            return element.selector
    return None
