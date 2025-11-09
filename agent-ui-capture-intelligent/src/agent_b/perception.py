"""DOM perception utilities for Agent B."""

from __future__ import annotations

from typing import Any, Dict

from playwright.async_api import Error as PlaywrightError, Page


_PERCEPTION_SCRIPT = r"""
() => {
  const cssEscape = (value) => {
    if (typeof CSS !== 'undefined' && CSS.escape) return CSS.escape(value);
    return value.replace(/"/g, '\"');
  };

  const visibilityCheck = (el) => {
    const style = window.getComputedStyle(el);
    if (!style || style.visibility === 'hidden' || style.display === 'none' || Number(style.opacity) === 0) return false;
    const rect = el.getBoundingClientRect();
    return rect && rect.width > 1 && rect.height > 1;
  };

  const buildSelector = (el, fallback, index) => {
    if (!el) return fallback;
    if (el.id) return `#${cssEscape(el.id)}`;
    const dataTestId = el.getAttribute('data-test-id');
    if (dataTestId) return `[data-test-id="${cssEscape(dataTestId)}"]`;
    const dataTracking = el.getAttribute('data-tracking-id');
    if (dataTracking) return `[data-tracking-id="${cssEscape(dataTracking)}"]`;
    const ariaLabel = el.getAttribute('aria-label');
    if (ariaLabel) return `${el.tagName.toLowerCase()}[aria-label="${cssEscape(ariaLabel)}"]`;
    const nameAttr = el.getAttribute('name');
    if (nameAttr) return `${el.tagName.toLowerCase()}[name="${cssEscape(nameAttr)}"]`;
    const placeholder = el.getAttribute('placeholder');
    if (placeholder) return `${el.tagName.toLowerCase()}[placeholder="${cssEscape(placeholder)}"]`;
    const text = (el.innerText || '').trim();
    if (text && text.length <= 40) return `text="${text.replace(/"/g, '\"')}"`;
    return `${el.tagName.toLowerCase()} >> nth=${index}`;
  };

  const describeElements = (elements, kind, limit = 40) => {
    const result = [];
    elements.every((el, idx) => {
      if (!el || !visibilityCheck(el)) return true;
      const rect = el.getBoundingClientRect();
      const text = (el.innerText || '').replace(/\\s+/g, ' ').trim();
      const entry = {
        kind,
        tag: el.tagName ? el.tagName.toLowerCase() : null,
        role: el.getAttribute('role'),
        text: text ? text.slice(0, 120) : null,
        ariaLabel: el.getAttribute('aria-label'),
        ariaPressed: el.getAttribute('aria-pressed'),
        dataTestId: el.getAttribute('data-test-id'),
        dataTrackingId: el.getAttribute('data-tracking-id'),
        href: el.getAttribute('href'),
        placeholder: el.getAttribute('placeholder'),
        type: el.getAttribute('type'),
        selector: buildSelector(el, `${kind} >> nth=${idx}`, idx),
        boundingBox: { x: rect.x, y: rect.y, width: rect.width, height: rect.height },
      };
      result.push(entry);
      return result.length < limit;
    });
    return result;
  };

  const clickables = Array.from(document.querySelectorAll('button, [role=button], a[href], [data-test-id], [data-tracking-id]'));
  const inputs = Array.from(document.querySelectorAll('input, textarea, [contenteditable="true"], [role="textbox"]'));
  const modals = Array.from(document.querySelectorAll('[role=dialog], dialog, [data-layer-kind="modal"]'));
  const overlays = Array.from(document.querySelectorAll('[data-animated-popover-backdrop], [data-layer-kind="modal-backdrop"], .ReactModal__Overlay'));

  const primaryActions = clickables.filter((el) => {
    const text = (el.innerText || '').toLowerCase();
    return text.includes('create') || text.includes('new') || text.includes('add') || text.includes('start');
  });

  return {
    clickables: describeElements(clickables, 'clickable', 40),
    inputs: describeElements(inputs, 'input', 25),
    modals: describeElements(modals, 'modal', 10),
    overlays: describeElements(overlays, 'overlay', 10),
    primaryActions: describeElements(primaryActions, 'primary', 15),
  };
}
"""


async def collect_affordances(page: Page) -> Dict[str, Any]:
    try:
        data = await page.evaluate(_PERCEPTION_SCRIPT)
        return data or {}
    except PlaywrightError:
        return {}
