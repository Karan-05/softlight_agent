"""Task runner orchestrating the Perception → Decision → Action → Capture loop."""

from __future__ import annotations

import json
import logging
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import re

try:  # optional dependency
    from openai import AsyncOpenAI  # type: ignore
except ImportError:
    AsyncOpenAI = None  # type: ignore[assignment]

from playwright.async_api import BrowserContext, Error as PlaywrightError, Page, async_playwright

from .config import (
    APP_LAUNCH_URLS,
    DATASET_ROOT,
    DEFAULT_BROWSER,
    OPENAI_MODEL,
    PLAYWRIGHT_CHANNEL,
    PLAYWRIGHT_EXECUTABLE,
    USER_DATA_DIR,
    get_openai_api_key,
)
from .executor import run_step
from .experience_store import ExperienceStore
from .semantic_index import SemanticIndex
from .heuristic_planner import HeuristicPlanner
from .hierarchical_planner import HierarchicalPlanner
from .llm_planner import LLMPlanner
from .models import PlannerAction, UISnapshot
from .plan_memory import ActionPlanMemory
from .self_heal import configure_memory
from .success import evaluate_success
from .ui_describer import describe_ui
from .robustness import wait_for_page_quiet
from .intent import reset_task_intent

logger = logging.getLogger(__name__)

VIEWPORT = {"width": 1440, "height": 900}


class TelemetryWriter:
    """Append structured events to a run.jsonl file."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = path.open("a", encoding="utf-8")

    def write(self, event: Dict[str, Any]) -> None:
        payload = dict(event)
        payload.setdefault("timestamp", datetime.utcnow().isoformat() + "Z")
        self._fp.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._fp.flush()

    def close(self) -> None:
        try:
            self._fp.close()
        except Exception:  # noqa: BLE001
            pass


def _snapshot_indicates_login(snapshot: UISnapshot) -> bool:
    url = (snapshot.url or "").lower()
    if any(keyword in url for keyword in ("/login", "login?", "sign-in", "signin")):
        return True
    login_keywords = {"email", "password", "sign in", "sign-in", "log in"}
    for element in snapshot.inputs:
        content = f"{(element.selector or '').lower()} {(element.text or '').lower()}"
        if any(keyword in content for keyword in login_keywords):
            return True
    for element in snapshot.clickables[:10]:
        content = f"{(element.selector or '').lower()} {(element.text or '').lower()}"
        if any(keyword in content for keyword in login_keywords):
            return True
    return False


def _determine_app(task: str, app_hint: Optional[str]) -> Optional[str]:
    tokens = {token for token in re.findall(r"[a-zA-Z]+", task.lower()) if len(token) > 2}
    if "notion" in tokens:
        return "notion"
    if "linear" in tokens:
        return "linear"
    return app_hint


async def run_agent_task(
    task: str,
    app_hint: Optional[str],
    out_dir: str,
    headless: bool,
    max_steps: int,
    timeout_ms: int,
    max_retries: int,
    profile_dir: Optional[str],
    browser: Optional[str],
    storage_state: Optional[str],
) -> None:
    dataset_root = Path(out_dir or DATASET_ROOT)
    dataset_root.mkdir(parents=True, exist_ok=True)
    task_slug = _slugify_task(task)
    reset_task_intent(task)

    memory_root = USER_DATA_DIR / "_memory"
    configure_memory(memory_root / "selectors.json", threshold=3)

    api_key = get_openai_api_key()
    llm_client = AsyncOpenAI(api_key=api_key) if api_key and AsyncOpenAI else None
    llm_planner = LLMPlanner(client=llm_client, model=OPENAI_MODEL) if llm_client else None
    heuristic_planner = HeuristicPlanner()
    experience_store = ExperienceStore(memory_root / "experiences.jsonl")
    semantic_index = SemanticIndex(memory_root / "semantic_index.jsonl")
    planner = HierarchicalPlanner(
        llm_planner=llm_planner,
        heuristic_planner=heuristic_planner,
    )
    plan_memory = ActionPlanMemory(memory_root / "plans")

    if llm_client:
        logger.info("LLM planner enabled (%s)", OPENAI_MODEL)
    else:
        logger.warning("OPENAI_API_KEY missing; using heuristic planner only")

    history: List[Dict[str, Any]] = []
    executed_history: List[Tuple[PlannerAction, bool, str]] = []
    resolved_app = _determine_app(task, app_hint)
    dataset_dir: Optional[Path] = None
    telemetry: Optional[TelemetryWriter] = None
    def _planner_event_logger(event: Dict[str, Any]) -> None:
        if telemetry:
            telemetry.write(event)

    planner.set_event_logger(_planner_event_logger)
    memory_plan_loaded = False
    memory_plan_key_app: Optional[str] = None
    memory_plan_active = False
    memory_plan_used_this_run = False
    experience_plan_loaded = False
    experience_plan_used = False
    semantic_plan_loaded = False
    semantic_plan_used = False
    login_wait_cycles = 0
    last_reload_at: Dict[str, float] = {}

    completed_via_done = False
    latest_snapshot: Optional[UISnapshot] = None
    non_interactive_cycles = 0
    blocked_reason: Optional[str] = None

    async with async_playwright() as pw:
        context, resolved_dir = await _launch_browser(
            pw,
            browser_choice=(browser or DEFAULT_BROWSER).lower(),
            headless=headless,
            profile_dir=profile_dir,
            app_hint=resolved_app,
        )
        try:
            page = context.pages[0] if context.pages else await context.new_page()
            await _apply_default_viewport(page)
            await _inject_reduced_motion(page)
            await _apply_storage_state(context, storage_state)
            await _bootstrap_app_hint(page, task, app_hint, dataset_root, task_slug)
            await wait_for_page_quiet(page, timeout_ms)

            console_events: List[Dict[str, Any]] = []
            page_errors: List[Dict[str, Any]] = []

            def handle_console(message) -> None:
                entry = {
                    "type": message.type,
                    "text": message.text,
                    "location": message.location,
                }
                console_events.append(entry)
                if telemetry:
                    telemetry.write({"event": "console", "payload": entry})

            def handle_page_error(exc) -> None:
                entry = {"message": str(exc)}
                page_errors.append(entry)
                if telemetry:
                    telemetry.write({"event": "pageerror", "payload": entry})

            page.on("console", handle_console)
            page.on("pageerror", handle_page_error)

            step_idx = 0
            current_app_folder: Optional[str] = None
            run_completed = False

            pending_snapshot: Optional[UISnapshot] = None

            while step_idx < max_steps:
                if pending_snapshot is not None:
                    snapshot = pending_snapshot
                    pending_snapshot = None
                else:
                    snapshot = await _wait_for_interactive_state(page, timeout_ms)
                latest_snapshot = snapshot
                resolved_app = resolved_app or snapshot.detected_app or _infer_app(snapshot.url)

                if not memory_plan_loaded and not _snapshot_indicates_login(snapshot):
                    plan_app_key = resolved_app or app_hint
                    stored_plan = plan_memory.load(plan_app_key, task_slug)
                    if stored_plan and stored_plan.failures < 3:
                        validated_actions, invalid_reason = plan_memory.validate_for_snapshot(
                            task=task,
                            snapshot=snapshot,
                            plan=stored_plan,
                        )
                        if validated_actions:
                            planner.prime_actions(validated_actions, source="memory")
                            memory_plan_loaded = True
                            memory_plan_key_app = plan_app_key
                            memory_plan_active = True
                            memory_plan_used_this_run = True
                        else:
                            reason_text = invalid_reason or "cached plan invalid for current snapshot"
                            logger.info(
                                "Discarding cached plan for %s/%s: %s",
                                plan_app_key or "generic",
                                task_slug,
                                reason_text,
                            )
                            plan_memory.mark_failure(plan_app_key, task_slug)
                            memory_plan_loaded = True
                    else:
                        memory_plan_loaded = True
                elif not memory_plan_loaded:
                    memory_plan_loaded = True

                if not experience_plan_loaded and not memory_plan_active and not memory_plan_used_this_run and not _snapshot_indicates_login(snapshot):
                    plan_app_key = resolved_app or app_hint
                    retrieved = experience_store.retrieve(plan_app_key, task)
                    if retrieved:
                        planner.prime_actions(retrieved.actions, source="experience")
                        experience_plan_loaded = True
                        experience_plan_used = True
                    else:
                        experience_plan_loaded = True

                if (
                    not semantic_plan_loaded
                    and not memory_plan_active
                    and not memory_plan_used_this_run
                    and not experience_plan_used
                    and not _snapshot_indicates_login(snapshot)
                ):
                    plan_app_key = resolved_app or app_hint
                    retrieved_semantic = semantic_index.retrieve(plan_app_key, task)
                    if retrieved_semantic:
                        planner.prime_actions(retrieved_semantic.actions, source="semantic")
                        semantic_plan_used = True
                    semantic_plan_loaded = True

                if not memory_plan_loaded:
                    plan_app_key = resolved_app or app_hint
                    stored_plan = plan_memory.load(plan_app_key, task_slug)
                    if stored_plan and stored_plan.failures < 3:
                        validated_actions, invalid_reason = plan_memory.validate_for_snapshot(
                            task=task,
                            snapshot=snapshot,
                            plan=stored_plan,
                        )
                        if validated_actions:
                            planner.prime_actions(validated_actions, source="memory")
                            memory_plan_loaded = True
                            memory_plan_key_app = plan_app_key
                            memory_plan_active = True
                            memory_plan_used_this_run = True
                        else:
                            reason_text = invalid_reason or "cached plan invalid for current snapshot"
                            logger.info(
                                "Discarding cached plan for %s/%s: %s",
                                plan_app_key or "generic",
                                task_slug,
                                reason_text,
                            )
                            plan_memory.mark_failure(plan_app_key, task_slug)
                            memory_plan_loaded = True
                    else:
                        memory_plan_loaded = True

                desired_folder = (resolved_app or app_hint or "generic") or "generic"
                if current_app_folder != desired_folder or dataset_dir is None:
                    current_app_folder = desired_folder
                    dataset_dir = dataset_root / current_app_folder / task_slug
                    dataset_dir.mkdir(parents=True, exist_ok=True)
                    if telemetry:
                        telemetry.close()
                    telemetry = TelemetryWriter(dataset_dir / "run.jsonl")
                    telemetry.write(
                        {
                            "event": "run_start",
                            "task": task,
                            "app": current_app_folder,
                            "headless": headless,
                            "timeout_ms": timeout_ms,
                            "max_retries": max_retries,
                        }
                    )

                snapshot_stats = _snapshot_stats(snapshot)
                if snapshot_stats["clickables"] == 0 and snapshot_stats["inputs"] == 0:
                    if _snapshot_indicates_login(snapshot):
                        login_wait_cycles += 1
                        logger.warning(
                            "Login surface detected; waiting for manual authentication (cycle %s).",
                            login_wait_cycles,
                        )
                        await page.wait_for_timeout(min(timeout_ms, 5000))
                        if login_wait_cycles >= 6:
                            logger.error("Login not completed after multiple waits; aborting task.")
                            break
                        continue

                    non_interactive_cycles += 1
                    if non_interactive_cycles >= 5:
                        blocked_reason = "non_interactive_ui"
                        logger.error("UI remained non-interactive for %s cycles; aborting run.", non_interactive_cycles)
                        if telemetry:
                            telemetry.write({"event": "blocked_ui", "reason": blocked_reason})
                        break
                    await page.wait_for_timeout(350)
                    rescanned = await describe_ui(page)
                    resolved_app = resolved_app or rescanned.detected_app or _infer_app(rescanned.url)
                    snapshot = rescanned
                    snapshot_stats = _snapshot_stats(snapshot)
                    if snapshot_stats["clickables"] == 0 and snapshot_stats["inputs"] == 0:
                        current_url = snapshot.url or page.url or ""
                        now = time.monotonic()
                        last = last_reload_at.get(current_url, 0.0)
                        if now - last >= 20.0:
                            logger.warning("No actionable UI elements detected; reloading page.")
                            await _reload_page(page, timeout_ms)
                            last_reload_at[current_url] = now
                        else:
                            logger.debug("Reload skipped for %s due to recent attempt.", current_url)
                        continue
                else:
                    non_interactive_cycles = 0
                login_wait_cycles = 0

                hint_payload = _snapshot_hint_event(snapshot)
                if telemetry:
                    telemetry.write({"event": "snapshot_hint", "step": step_idx, **hint_payload})
                logger.debug(
                    "Snapshot hint step %s: %s",
                    step_idx,
                    [entry.get("text") for entry in hint_payload["clickables"]],
                )

                repeat_avoid = _selectors_with_high_repeats(history)
                avoid_selectors = [
                    entry.get("selector")
                    for entry in history
                    if isinstance(entry, dict) and not entry.get("success") and entry.get("selector")
                ]
                avoid_selectors.extend(repeat_avoid)

                try:
                    action, planner_source = await planner.plan_next(
                        task,
                        snapshot,
                        history,
                        avoid_selectors=avoid_selectors,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Composite planner failure: %s", exc)
                    if telemetry:
                        telemetry.write({"event": "planner_error", "error": str(exc)})
                    action = heuristic_planner.plan_next(task, snapshot, history, avoid_selectors=avoid_selectors)
                    planner_source = "heuristic"
                else:
                    invalidated = planner.consume_invalidated_sources()
                    if "memory" in invalidated and memory_plan_active and memory_plan_key_app:
                        logger.info("Cached memory plan invalidated by current snapshot; marking failure.")
                        plan_memory.mark_failure(memory_plan_key_app, task_slug)
                        memory_plan_active = False
                    if invalidated - {"memory"}:
                        logger.debug("Planner invalidated cached actions from: %s", ", ".join(sorted(invalidated - {"memory"})))

                start_console = len(console_events)
                start_errors = len(page_errors)

                def log_tail_provider() -> List[Dict[str, Any]]:
                    tail: List[Dict[str, Any]] = []
                    for entry in console_events[start_console:]:
                        tail.append({"kind": "console", **entry})
                    for entry in page_errors[start_errors:]:
                        tail.append({"kind": "pageerror", **entry})
                    return tail

                success = await run_step(
                    page=page,
                    action=action,
                    out_dir=dataset_dir,
                    step_idx=step_idx,
                    timeout_ms=timeout_ms,
                    retries=max_retries,
                    log_tail_provider=log_tail_provider,
                )

                console_tail = console_events[start_console:]
                page_error_tail = page_errors[start_errors:]

                step_event = {
                    "event": "step",
                    "step": step_idx,
                    "action": action.model_dump(),
                    "success": success,
                    "planner": planner_source,
                    "console": console_tail,
                    "page_errors": page_error_tail,
                    "affordances": snapshot_stats,
                }
                if telemetry:
                    telemetry.write(step_event)

                history.append({**action.model_dump(), "success": success, "planner": planner_source})
                executed_history.append((action, success, planner_source))
                if not success and (planner_source.startswith("llm") or planner_source in {"experience", "semantic", "memory"}):
                    planner.clear_pending()
                if planner_source == "memory" and not success and memory_plan_active and memory_plan_key_app:
                    plan_memory.mark_failure(memory_plan_key_app, task_slug)
                    memory_plan_active = False

                post_snapshot = await describe_ui(page)
                pending_snapshot = post_snapshot
                latest_snapshot = post_snapshot
                success_signal = evaluate_success(task, post_snapshot, history)
                if success_signal.satisfied:
                    reason = success_signal.reason or "Generic success condition met."
                    if telemetry:
                        telemetry.write(
                            {
                                "event": "success",
                                "step": step_idx,
                                "reason": reason,
                                "category": success_signal.category,
                            }
                        )
                    logger.info("Success detected via generic rules: %s", reason)
                    done_step = step_idx + 1
                    done_action = PlannerAction(action="done", capture_name="success-state", reason=reason)
                    done_success = await run_step(
                        page=page,
                        action=done_action,
                        out_dir=dataset_dir if dataset_dir else dataset_root,
                        step_idx=done_step,
                        timeout_ms=timeout_ms,
                        retries=max_retries,
                    )
                    done_event = {
                        "event": "step",
                        "step": done_step,
                        "action": done_action.model_dump(),
                        "success": done_success,
                        "planner": "success-monitor",
                        "console": [],
                        "page_errors": [],
                        "affordances": _snapshot_stats(post_snapshot),
                    }
                    if telemetry:
                        telemetry.write(done_event)
                    history.append({**done_action.model_dump(), "success": done_success, "planner": "success-monitor"})
                    executed_history.append((done_action, done_success, "success-monitor"))
                    run_completed = True
                    completed_via_done = done_success
                    step_idx = done_step + 1
                    break

                step_idx += 1
                if action.action == "done":
                    logger.info("Planner reported task complete at step %s", step_idx - 1)
                    run_completed = True
                    completed_via_done = success
                    break
        finally:
            if telemetry:
                telemetry.write({"event": "run_end"})
                telemetry.close()

            if memory_plan_used_this_run and not run_completed and memory_plan_key_app:
                plan_memory.mark_failure(memory_plan_key_app, task_slug)

            if run_completed and completed_via_done and executed_history:
                successful_actions: List[PlannerAction] = [
                    action
                    for action, was_success, planner_source in executed_history
                    if was_success and action.action not in {"capture", "done"}
                ]
                if successful_actions:
                    plan_key = resolved_app or app_hint
                    plan_memory.save(
                        plan_key,
                        task_slug,
                        successful_actions,
                        source="memory",
                        snapshot=latest_snapshot,
                        task=task,
                    )
                    experience_store.add(plan_key, task, successful_actions)
                    semantic_index.add(plan_key, task, successful_actions)
            await context.close()


async def _bootstrap_app_hint(
    page: Page,
    task: str,
    app_hint: Optional[str],
    dataset_root: Path,
    task_slug: str,
) -> None:
    if not app_hint:
        return
    launch_url = APP_LAUNCH_URLS.get(app_hint.lower())
    if not launch_url:
        return
    try:
        await page.goto(launch_url, wait_until="domcontentloaded")
    except PlaywrightError as exc:
        logger.warning("Initial navigation to %s failed: %s", launch_url, exc)


async def _apply_default_viewport(page: Page) -> None:
    try:
        await page.set_viewport_size(VIEWPORT)
    except PlaywrightError:
        pass


async def _inject_reduced_motion(page: Page) -> None:
    styles = """
        *, *::before, *::after {
            transition-duration: 0s !important;
            animation-duration: 0s !important;
            scroll-behavior: auto !important;
        }
    """
    try:
        await page.add_style_tag(content=styles)
    except PlaywrightError:
        pass


async def _apply_storage_state(context: BrowserContext, storage_state: Optional[str]) -> None:
    if not storage_state:
        return
    storage_path = Path(storage_state).expanduser()
    if not storage_path.exists():
        logger.warning("Storage state not found at %s; continuing without it.", storage_path)
        return
    try:
        payload = json.loads(storage_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logger.warning("Invalid storage state JSON (%s); continuing without it.", exc)
        return

    cookies = payload.get("cookies") or []
    if cookies:
        try:
            await context.add_cookies(cookies)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Unable to apply baked cookies: %s", exc)

    origins = payload.get("origins") or []
    for entry in origins:
        origin = entry.get("origin")
        local_storage = entry.get("localStorage") or []
        if not origin or not local_storage:
            continue
        page: Optional[Page] = None
        try:
            page = await context.new_page()
            await page.goto(origin, wait_until="domcontentloaded", timeout=15000)
            for item in local_storage:
                name = item.get("name")
                value = item.get("value")
                if name is None or value is None:
                    continue
                await page.evaluate(
                    "(data) => window.localStorage.setItem(data.name, data.value)",
                    {"name": name, "value": value},
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Unable to hydrate localStorage for %s: %s", origin, exc)
        finally:
            if page is not None:
                try:
                    await page.close()
                except Exception:  # noqa: BLE001
                    pass


async def _launch_browser(
    playwright,
    browser_choice: str,
    headless: bool,
    profile_dir: Optional[str],
    app_hint: Optional[str],
) -> Tuple[BrowserContext, Path]:
    if browser_choice == "chrome":
        return await _launch_chrome_browser(playwright, profile_dir, app_hint, headless)

    resolved_dir = _prepare_generic_user_data_dir(profile_dir, app_hint, browser_choice)
    browser_type = getattr(playwright, browser_choice, None)
    if browser_type is None:
        raise ValueError(f"Unsupported browser engine: {browser_choice}")
    context = await browser_type.launch_persistent_context(
        user_data_dir=str(resolved_dir),
        headless=headless,
        viewport=VIEWPORT,
        reduced_motion="reduce",
    )
    return context, resolved_dir


async def _launch_chrome_browser(pw, profile_dir: Optional[str], app_hint: Optional[str], headless: bool):
    resolved_dir, profile_directory = _prepare_chrome_user_data_dir(profile_dir, app_hint)

    launch_kwargs: Dict[str, Any] = {
        "user_data_dir": str(resolved_dir),
        "headless": headless,
        "viewport": VIEWPORT,
        "reduced_motion": "reduce",
    }
    if PLAYWRIGHT_EXECUTABLE:
        launch_kwargs["executable_path"] = PLAYWRIGHT_EXECUTABLE
    elif PLAYWRIGHT_CHANNEL:
        launch_kwargs["channel"] = PLAYWRIGHT_CHANNEL

    if profile_directory:
        launch_kwargs.setdefault("args", [])
        launch_kwargs["args"].append(f"--profile-directory={profile_directory}")

    context = await pw.chromium.launch_persistent_context(**launch_kwargs)
    return context, resolved_dir


def _prepare_generic_user_data_dir(
    profile_dir: Optional[str],
    app_hint: Optional[str],
    browser_choice: str,
) -> Path:
    if profile_dir:
        dest = Path(profile_dir).expanduser()
        dest.mkdir(parents=True, exist_ok=True)
        logger.info("Using provided %s profile directory: %s", browser_choice, dest)
        return dest

    dest = USER_DATA_DIR / browser_choice / (app_hint or "generic")
    dest.mkdir(parents=True, exist_ok=True)
    logger.info("Using agent-managed %s profile directory: %s", browser_choice, dest)
    return dest


def _prepare_chrome_user_data_dir(profile_dir: Optional[str], app_hint: Optional[str]) -> Tuple[Path, Optional[str]]:
    if not profile_dir:
        dest_dir = USER_DATA_DIR / "chrome" / (app_hint or "generic")
        dest_dir.mkdir(parents=True, exist_ok=True)
        return dest_dir, None

    source_path = Path(profile_dir).expanduser()
    if not source_path.exists():
        raise FileNotFoundError(f"Profile directory does not exist: {source_path}")

    if (source_path / "Preferences").exists() and (source_path.parent / "Local State").exists():
        source_root = source_path.parent
        profile_name = source_path.name
    elif (source_path / "Local State").exists():
        source_root = source_path
        profile_name = None
    else:
        raise ValueError(
            "Provided profile path must point to a Chrome user data root or a profile directory (e.g. 'Default')."
        )

    digest = _hash_path(source_path)
    dest_root = USER_DATA_DIR / "_imported" / digest
    _clone_chrome_profile(source_root, dest_root, profile_name)

    return dest_root, profile_name


def _clone_chrome_profile(source_root: Path, dest_root: Path, profile_name: Optional[str]) -> None:
    if dest_root.exists():
        shutil.rmtree(dest_root)
    dest_root.mkdir(parents=True, exist_ok=True)

    items_to_copy: Optional[List[str]] = None
    if profile_name:
        items_to_copy = ["Local State", profile_name]

    for entry in source_root.iterdir():
        if items_to_copy and entry.name not in items_to_copy:
            continue
        src = entry
        dst = dest_root / entry.name
        try:
            if entry.is_dir():
                shutil.copytree(
                    src,
                    dst,
                    dirs_exist_ok=True,
                    symlinks=True,
                    ignore_dangling_symlinks=True,
                )
            else:
                shutil.copy2(src, dst)
        except (FileNotFoundError, PermissionError) as exc:
            logger.warning("Skipping Chrome profile entry %s: %s", src, exc)


def _hash_path(path: Path) -> str:
    return hex(abs(hash(str(path))))[2:]


def _infer_app(url: str) -> Optional[str]:
    lowered = url.lower()
    if "linear.app" in lowered:
        return "linear"
    if "notion.so" in lowered:
        return "notion"
    if lowered and lowered not in {"about:blank", "chrome://new-tab-page"}:
        return "generic"
    return None


def _slugify_task(task: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "-" for ch in task.lower())
    cleaned = "-".join(part for part in cleaned.split("-") if part)
    return cleaned or "task"


async def _wait_for_interactive_state(page: Page, timeout_ms: int, *, max_attempts: int = 6) -> UISnapshot:
    attempts = 0
    snapshot = await describe_ui(page)
    while attempts < max_attempts:
        if snapshot.clickables or snapshot.inputs:
            return snapshot
        if await _body_has_class(page, "logged-out"):
            return snapshot
        await wait_for_page_quiet(page, timeout_ms)
        await page.wait_for_timeout(min(800, 400 + attempts * 200))
        snapshot = await describe_ui(page)
        attempts += 1
    return snapshot


async def _body_has_class(page: Page, class_name: str) -> bool:
    try:
        return bool(
            await page.evaluate(
                "(cls) => Boolean(document && document.body && document.body.classList && document.body.classList.contains(cls))",
                class_name,
            )
        )
    except PlaywrightError:
        return False


async def _reload_page(page: Page, timeout_ms: int) -> None:
    try:
        await page.reload(wait_until="domcontentloaded")
    except PlaywrightError as exc:
        logger.warning("Page reload failed: %s", exc)
    await wait_for_page_quiet(page, timeout_ms)


def _selectors_with_high_repeats(history: Sequence[Dict[str, Any]], limit: int = 3) -> List[str]:
    if limit < 2:
        limit = 2
    repeated: List[str] = []
    streak = 0
    last_key: Optional[Tuple[str, str]] = None
    for entry in reversed(history):
        if not isinstance(entry, dict):
            continue
        action = str(entry.get("action") or "").lower()
        selector = entry.get("selector")
        if not selector or action not in {"click", "fill", "press"}:
            continue
        failure = entry.get("success") is False
        if not failure:
            streak = 0
            last_key = None
            continue
        key = (action, selector)
        if key == last_key:
            streak += 1
        else:
            streak = 1
            last_key = key
        if streak >= limit:
            repeated.append(selector)
    seen: Dict[str, None] = {}
    for selector in repeated:
        seen.setdefault(selector, None)
    return list(seen.keys())


def _snapshot_stats(snapshot: UISnapshot) -> Dict[str, int]:
    return {
        "clickables": len(snapshot.clickables),
        "inputs": len(snapshot.inputs),
        "modals": len(snapshot.modals),
        "primary": len(snapshot.primary_actions),
    }


def _snapshot_hint_event(snapshot: UISnapshot) -> Dict[str, Any]:
    top_clickables: List[Dict[str, Optional[str]]] = []
    for element in snapshot.clickables[:10]:
        top_clickables.append(
            {
                "text": element.text,
                "selector": element.selector,
                "role": element.role,
            }
        )
    modal_summaries: List[Dict[str, Optional[str]]] = []
    for element in snapshot.modals[:5]:
        modal_summaries.append(
            {
                "text": element.text,
                "selector": element.selector,
                "role": element.role,
            }
        )
    return {
        "url": snapshot.url,
        "counts": _snapshot_stats(snapshot),
        "clickables": top_clickables,
        "modals": modal_summaries,
    }
