"""CLI entrypoint for Agent B."""

from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path

from agent_b.config import DATASET_ROOT, DEFAULT_BROWSER
from agent_b.runner import run_agent_task


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Agent B against a natural-language UI task.")
    parser.add_argument("--task", required=True, help="Natural-language task for the agent to complete.")
    parser.add_argument("--app", help="Optional app hint, e.g. linear or notion.")
    parser.add_argument("--outdir", default=str(DATASET_ROOT), help="Directory to store captures and telemetry.")
    parser.add_argument("--headless", action="store_true", help="Run the browser in headless mode.")
    parser.add_argument("--max-steps", type=int, default=15, help="Maximum planner iterations before stopping.")
    parser.add_argument("--timeout-ms", type=int, default=8000, help="Base timeout for element operations.")
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum retries for flaky element operations.")
    parser.add_argument(
        "--browser",
        default=DEFAULT_BROWSER,
        help="Browser engine to use (chrome, chromium, firefox, or webkit).",
    )
    parser.add_argument("--profile-dir", help="Optional user data directory to reuse between runs.")
    parser.add_argument(
        "--storage-state",
        help="Path to a Playwright storage state JSON file; used to seed authenticated sessions.",
    )
    parser.add_argument("--log-level", default="INFO", help="Python logging level.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_file = _configure_logging(args.log_level)
    logging.info("Log file: %s", log_file)
    _validate_args(args)
    _warn_auth_hint(args)

    asyncio.run(
        run_agent_task(
            task=args.task,
            app_hint=args.app,
            out_dir=args.outdir,
            headless=args.headless,
            max_steps=args.max_steps,
            timeout_ms=args.timeout_ms,
            max_retries=args.max_retries,
            profile_dir=args.profile_dir,
            browser=args.browser,
            storage_state=args.storage_state,
        )
    )


def _validate_args(args: argparse.Namespace) -> None:
    if args.storage_state:
        storage_path = Path(args.storage_state).expanduser()
        if not storage_path.exists():
            raise SystemExit(f"Storage state file not found: {storage_path}")
        if not storage_path.is_file():
            raise SystemExit(f"Storage state must be a file: {storage_path}")
    if args.profile_dir:
        profile_path = Path(args.profile_dir).expanduser()
        if profile_path.exists() and not profile_path.is_dir():
            raise SystemExit(f"Profile directory must be a directory path: {profile_path}")


def _warn_auth_hint(args: argparse.Namespace) -> None:
    if not args.storage_state and not args.profile_dir:
        logging.warning(
            "No storage state or profile directory provided. "
            "Agent B will prompt for manual login when authentication is required."
        )


def _configure_logging(log_level: str) -> Path:
    level = getattr(logging, log_level.upper(), logging.INFO)
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"agent-b-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}.log"
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logging.basicConfig(level=level, handlers=[stream_handler, file_handler])
    return log_file


if __name__ == "__main__":
    main()
