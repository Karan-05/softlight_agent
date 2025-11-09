"""Capture utilities for storing UI state."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from playwright.async_api import Error as PlaywrightError, Page


async def capture_state(
    page: Page,
    out_dir: Path,
    name: str,
    extra: Optional[Dict[str, Any]] = None,
    log_tail: Optional[List[Dict[str, Any]]] = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_extra: Dict[str, Any] = dict(extra or {})
    if log_tail:
        meta_extra["log_tail"] = log_tail

    metadata: Dict[str, Any] = {
        "name": name,
        "url": page.url if hasattr(page, "url") else None,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "extra": meta_extra,
    }

    try:
        await page.screenshot(path=str(out_dir / f"{name}.png"), full_page=True)
    except PlaywrightError as exc:
        metadata.setdefault("extra", {})["screenshot_error"] = str(exc)

    try:
        html = await page.content()
        (out_dir / f"{name}.html").write_text(html, encoding="utf-8")
    except PlaywrightError as exc:
        metadata.setdefault("extra", {})["html_error"] = str(exc)

    (out_dir / f"{name}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
