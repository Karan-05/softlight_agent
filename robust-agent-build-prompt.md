
Robust Intelligent UI Agent – Codex Build Prompt
================================================

You are an expert senior engineer. Rewrite/build the **Agent B** repository as a robust, production-grade, LLM-in-the-loop UI agent. It must complete natural-language tasks on unseen web apps by iterating:
Perception -> Decision -> Action -> Capture, with strong anti-flake engineering.

This brief replaces any naive or hardcoded flows. Implement all items below.

------------------------------------------------------------
HIGH-LEVEL OBJECTIVE
------------------------------------------------------------
- Input: a user task like "Create a project", "Filter a database", "Change notification settings".
- The agent opens a real browser, describes the current UI, asks an LLM to choose ONE next action from the visible affordances, executes it, captures state, and repeats until done or max steps.
- No hardcoded app flows. All choices must be derived from what is observed on the current page.

------------------------------------------------------------
REPO STRUCTURE
------------------------------------------------------------
agent-ui-capture-intelligent/
├── README.md
├── requirements.txt
├── .env.example
├── src/
│   ├── __init__.py
│   ├── main.py
│   └── agent_b/
│       ├── __init__.py
│       ├── config.py
│       ├── models.py
│       ├── ui_describer.py
│       ├── llm_planner.py
│       ├── heuristic_planner.py
│       ├── robustness.py
│       ├── executor.py
│       ├── capturer.py
│       └── runner.py
└── datasets/
    └── README.md

Write all files with real, runnable code.

------------------------------------------------------------
RUNTIME LOOP
------------------------------------------------------------
The agent must perform up to max_steps of:

1) Perception: build a compact UISnapshot from the live page:
   - url, title
   - clickables[]: {text, role, selector, visible, enabled}
   - inputs[]: {label_or_placeholder, selector, visible, enabled}
   - modals[]: {title_or_aria_label, selector}
   - frame awareness: gather from top frame and visible iframes (limit small)
   - limit counts per category (e.g., 40) to keep prompts small

2) Decision: call LLMPlanner.plan_next(task, snapshot, history) to return EXACTLY ONE action as JSON:
   - { action: "goto"|"click"|"fill"|"wait_for"|"capture"|"done",
       selector?: string, url?: string, value?: string,
       capture_name?: string, expect?: string, reason?: string }
   - If the LLM fails or returns invalid JSON, fall back to HeuristicPlanner.

3) Action: execute with robust wrappers (see "Robustness Requirements").

4) Capture: ALWAYS write full-page screenshot, DOM HTML, and metadata JSON.

5) Repeat until action == "done" or max_steps reached.

------------------------------------------------------------
ROBUSTNESS REQUIREMENTS (MANDATORY)
------------------------------------------------------------
Implement these in a new module robustness.py and use them across the executor:

A) Element Resolution & Scoring
   - Given a textual intent like "create project" and a set of candidates, compute a score per element:
     - text similarity (case-insensitive, partial, synonyms): ["create","new","+ new","add","add new","save","submit"]
     - role preference (buttons and links > generic divs)
     - visibility & enabled state
     - if a modal is open, prefer elements inside it
   - Prefer selectors that are stable: role+name, exact text, data-testid, aria-label, placeholder, label->for mapping.
   - Provide resolve_click_target(snapshot, preferred_texts: list[str]) -> str|None that returns the best selector.

B) Modal/Overlay Detection
   - Detect dialogs via [role=dialog], dialog, [data-modal], [data-overlay], and fixed overlays.
   - If a modal is present, prioritize actions inside the modal.

C) Wait Strategies
   - Implement wait_for_page_quiet(page) that waits for:
     - network idle
     - no spinners/overlays (common selectors; configurable)
     - minimal DOM churn (page.wait_for_function with MutationObserver counters)
   - For each action, wait for target visible+enabled before and a postcondition after (modal present/absent, url change, DOM size change).

D) Retry Policy (with Re-query)
   - For click/fill/wait_for: attempt up to 3 tries with exponential backoff (300ms, 700ms, 1500ms).
   - Re-query the locator each attempt.
   - On final failure, capture an error state and return control for planner "repair".

E) Click & Fill Wrappers
   - click_robust(selector): scroll into view, ensure visible/enabled, try locator.click(); then page.mouse.click(center); last resort locator.click(force=True).
   - fill_robust(selector, text): focus, Control+A then Backspace, fill; verify value after.
   - Both wrappers must capture on failure with reason.

F) Viewport, Motion, Timeouts
   - Stable viewport (1440x900).
   - Reduced motion (CSS injection + context option).
   - Default timeouts (e.g., 8000ms) with CLI overrides.

G) iFrames & Shadow DOM
   - ui_describer includes visible iframes; provide a way to prefix selectors with a frame index.
   - Note shadow DOM limitations; rely on locators where possible.

H) Verification
   - If action.expect present, assert it post-action (modal seen, url changed, element appears/disappears). Otherwise mark step "uncertain".

I) Logging & Telemetry
   - JSONL per-run with per-step action, selector, timings, retries, errors.
   - Listen to console and pageerror; include summaries in metadata.

------------------------------------------------------------
LLM PLANNER
------------------------------------------------------------
- Use OpenAI (1.x client). Model from config (default "gpt-4o-mini"). If missing key, skip LLM and use heuristic.
- Prompt must include:
  - task
  - url, title
  - up to 40 clickables (index, role, text, selector)
  - up to 20 inputs (index, label/placeholder, selector)
  - modal summary
  - recent history (last 5)
- Instructions:
  - Return EXACTLY ONE JSON object with keys: action, selector|url|value, capture_name (optional), expect (optional), reason (short).
  - Prefer interacting with visible dialogs first.
  - If task mentions create/new/add, prefer those words.
  - Use only selectors provided.
  - If uncertain, return {"action":"capture","capture_name":"ambiguous"}.
  - If complete, return {"action":"done"}.

------------------------------------------------------------
HEURISTIC FALLBACK
------------------------------------------------------------
- If modal exists: choose best inside it using resolve_click_target([...]).
- Else: choose best clickable by synonyms; otherwise capture.

------------------------------------------------------------
EXECUTION & CAPTURE
------------------------------------------------------------
- After every step: capture_state with files:
  - step-N.png (full page)
  - step-N.html (page.content())
  - step-N.json (name, url, timestamp, action summary, retries, errors, console summary)
- Name pattern: step-{idx:02d}-{action-or-custom}.

------------------------------------------------------------
CLI & CONFIG
------------------------------------------------------------
- main.py args: --task, --app, --outdir, --headless, --max-steps, --timeout-ms, --max-retries.
- config.py: DATASET_ROOT, PLAYWRIGHT_STORAGE, OPENAI_MODEL, reading OPENAI_API_KEY from env.
- Load storage_state if available for app.

------------------------------------------------------------
TESTABILITY
------------------------------------------------------------
- Add mock instructions in README to test: "New" opens a modal with "Name" input and "Create" button; a spinner overlay clears after 600ms.
- Provide example CLI commands and expected capture layout.

------------------------------------------------------------
CODING RULES
------------------------------------------------------------
- async/await for Playwright.
- Relative imports, ASCII quotes.
- Implement all functions; no placeholders.
- Graceful degradation without OpenAI (heuristic-only).
- Always capture after each step.

------------------------------------------------------------
OUTPUT
------------------------------------------------------------
Output ALL files with their contents, each separated by file path lines so they can be written to disk.
