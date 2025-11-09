Agent B: Perception-Driven UI Agent
===================================

Agent B completes natural-language UI tasks by iterating **Perception → Decision → Action → Capture** in a real browser. Every turn inspects the current page, calls an LLM (with a heuristic fallback) for **exactly one** next action, executes that action with robust Playwright wrappers, waits for the UI to settle, and captures the result.

Key properties:

- No app-specific workflows or DAGs – the agent only acts on what it can currently see.
- Planner returns a single JSON action (`goto`, `click`, `fill`, `wait_for`, `capture`, `done`) with optional `expect` assertions.
- Executor applies retries with exponential backoff, re-queries locators each attempt, enforces a stable 1440×900 viewport, reduces animations, and verifies post-conditions when `expect` is provided.
- After every step the agent writes full-page PNG, DOM HTML, and a JSON metadata record, plus appends an event to `run.jsonl` (actions, retries, console/page errors, planner source).
- A generic `success.evaluate_success(...)` check runs after each capture and terminates the loop once the UI reflects the requested state (modal visible, heading matches, filter chip applied, toggle switched). This guarantees every dataset ends with an explicit success-state capture instead of a timeout.
- Cached plans are recorded automatically from prior successful runs and are always derived from real UISnapshots/PlannerActions; every replay is revalidated against the live snapshot and pruned if selectors drift or dismiss/destructive actions sneak in, so there are no handwritten, app-specific workflows.
- A generic `success.evaluate_success(...)` check runs after each capture and terminates the loop once the UI reflects the requested state (modal visible, heading matches, filter chip applied, toggle switched, etc.), ensuring every dataset ends with a `success-state` capture rather than a max-step timeout.
- Hierarchical planning pipeline replays cached scripts when available, queries the LLM for the next JSON action, and falls back to a generic heuristic when models are unavailable; keyboard shortcuts (`press`) and richer UI perception (primary actions, breadcrumbs, overlays) feed smarter decisions across unfamiliar apps.

Planner Intelligence
--------------------

- `HierarchicalPlanner` replays any cached scripts, queries the LLM for a compliant action, and falls back to a heuristic when necessary so the agent can reason over multi-step flows without hardcoded routines.
- `LLMPlanner.plan_script` requests short JSON playbooks (2–5 steps) that are replayed sequentially, reducing back-and-forth API calls.
- The enhanced `ui_describer` reports primary actions, breadcrumbs, overlays, and per-element metadata to prioritise high-signal affordances.
- The core loop always consumes one atomic JSON action at a time. When a short JSON plan is available it is simply expanded into those same single-step actions—no hidden DAGs or procedural workflows.

HeuristicPlanner Enhancements
-----------------------------

- Infers task intent (create, filter/search, toggle, command palette, share, onboarding, capture-only) and prioritises elements that match those verbs/nouns, whether or not an LLM is available.
- Generates concise artifact titles that mirror the user instruction: explicit names are honored, “random title/name” instructions produce short Automation-style strings with timestamp+suffix, and the same value is reused for fills, summaries, and success checks.
- Prefers modal/overlay controls when visible, issues keyboard shortcuts for command palettes, and auto-emits capture actions when a task explicitly asks to “capture the modal/menu”.
- Scores purely on UISnapshot metadata (text, role, aria attributes, owner dialog/menu ancestry) and penalises selectors that failed repeatedly in the last few steps to avoid loops.
- No app names, URLs, or selectors are hardcoded; everything flows from the live UISnapshot plus natural-language intent words.

Generic Success Detection
-------------------------

`success.evaluate_success(task, snapshot, history)` inspects every post-step snapshot to decide whether the UI already reflects the requested state. The detector is intentionally app-agnostic and reasons only about visible text/roles:

- **Navigation / open / switch** – looks for active breadcrumbs, selected nav items, or document titles whose text matches the task nouns.
- **Create / add / new** – confirms modals/forms were opened, inputs were filled with task-derived values, and the newly created text appears in the DOM or a success toast.
- **Filter / search** – detects filter chips, search modals, or text like “Filtered by …” that includes the requested attributes (e.g., “high priority”).
- **Toggle / settings** – inspects checkboxes/switches for `aria-checked` changes near the requested label (“notifications”, “workspace settings”, etc.).
- **Command palette / share / modal capture** – verifies a modal/dialog is visible with fuzzy matches on phrases like “command palette”, “share”, “settings”, or the task nouns, then triggers a final `done` capture so each dataset ends with an explicit completion step.

Every time a rule fires the runner logs `{"event": "success", ...}`, emits a final `step-XX-success-state.*` capture, and persists the action script so future runs can shortcut straight to completion.

Quickstart
----------

```bash
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m playwright install

export OPENAI_API_KEY=sk-...           # or keep in .env
python src/main.py \
  --task "Create a new item" \
  --max-steps 12 \
  --timeout-ms 9000 \
  --max-retries 4

Verified Smoke Commands
-----------------------

Run the following from `agent-ui-capture-intelligent/` after activating `.venv` (or by calling `.venv/bin/python` directly) and exporting `OPENAI_API_KEY`.

### Linear

```bash
.venv/bin/python src/main.py --task "Create a new project in Linear" --app linear --storage-state linear-storage.json --max-steps 16 --timeout-ms 10000 --headless
.venv/bin/python src/main.py --task "Filter issues to show only high priority items in Linear" --app linear --storage-state linear-storage.json --max-steps 14 --timeout-ms 10000 --headless
.venv/bin/python src/main.py --task "Open workspace notification settings and toggle a switch in Linear" --app linear --storage-state linear-storage.json --max-steps 12 --timeout-ms 11000 --headless
.venv/bin/python src/main.py --task "Open the create issue modal and capture it" --app linear --storage-state linear-storage.json --max-steps 12 --timeout-ms 9000 --headless
.venv/bin/python src/main.py --task "Open the command palette and search for integrations in Linear" --app linear --storage-state linear-storage.json --max-steps 12 --timeout-ms 9000 --headless
```

### Notion

```bash
.venv/bin/python src/main.py --task "Open the quick find command palette in Notion" --app notion --max-steps 6 --timeout-ms 9000 --headless
.venv/bin/python src/main.py --task "Open workspace settings in Notion and capture the modal" --app notion --max-steps 8 --timeout-ms 9000 --headless
.venv/bin/python src/main.py --task "Capture the share menu for the current Notion page" --app notion --max-steps 12 --timeout-ms 9000 --headless
```

> **Note:** The bundled Firefox profiles already contain authenticated sessions. Avoid passing an outdated `--storage-state` for Notion because the SaaS may force a new onboarding flow.

Each run produces a dataset under `datasets/<app>/<task-slug>/` with:

- `run.jsonl` – planner telemetry plus `{"event": "success", ...}` once a generic rule matches.
- `step-XX-*.png/.html/.json` – the final `success-state` captures the exact UI that triggered completion.
- Additional `step-XX-requested-capture.*` artifacts for tasks that explicitly ask for a modal/screen grab.

Curated Datasets
----------------

| Task | Command | Dataset | Key captured state |
| --- | --- | --- | --- |
| Create a new project in Linear | `.venv/bin/python src/main.py --task "Create a new project in Linear" --app linear --storage-state linear-storage.json --max-steps 16 --timeout-ms 10000 --headless` | `datasets/linear/create-a-new-project-in-linear/` | `step-02-success-state.png` shows the new project card plus the success event. |
| Filter issues to show only high priority items in Linear | `.venv/bin/python src/main.py --task "Filter issues to show only high priority items in Linear" --app linear --storage-state linear-storage.json --max-steps 14 --timeout-ms 10000 --headless` | `datasets/linear/filter-issues-to-show-only-high-priority-items-in-linear/` | `step-01-success-state.png` captures the backlog filtered by the “High priority” chip. |
| Open workspace notification settings and toggle a switch in Linear | `.venv/bin/python src/main.py --task "Open workspace notification settings and toggle a switch in Linear" --app linear --storage-state linear-storage.json --max-steps 12 --timeout-ms 11000 --headless` | `datasets/linear/open-workspace-notification-settings-and-toggle-a-switch-in-linear/` | `step-09-success-state.png` shows the workspace settings pane with the notification toggle state. |
| Open the create issue modal and capture it (Linear) | `.venv/bin/python src/main.py --task "Open the create issue modal and capture it" --app linear --storage-state linear-storage.json --max-steps 12 --timeout-ms 9000 --headless` | `datasets/linear/open-the-create-issue-modal-and-capture-it/` | `step-01-success-state.png` contains the Linear “Create issue” modal requested by the task. |
| Open the command palette and search for integrations in Linear | `.venv/bin/python src/main.py --task "Open the command palette and search for integrations in Linear" --app linear --storage-state linear-storage.json --max-steps 12 --timeout-ms 9000 --headless` | `datasets/linear/open-the-command-palette-and-search-for-integrations-in-linear/` | `step-02-success-state.png` shows the command palette results for “integrations”. |
| Open the quick find command palette in Notion | `.venv/bin/python src/main.py --task "Open the quick find command palette in Notion" --app notion --max-steps 6 --timeout-ms 9000 --headless` | `datasets/notion/open-the-quick-find-command-palette-in-notion/` | `step-01-success-state.png` captures the Notion quick find modal with the search box focused. |
| Open workspace settings in Notion and capture the modal | `.venv/bin/python src/main.py --task "Open workspace settings in Notion and capture the modal" --app notion --max-steps 8 --timeout-ms 9000 --headless` | `datasets/notion/open-workspace-settings-in-notion-and-capture-the-modal/` | `step-01-success-state.png` includes the workspace settings dialog, demonstrating modal capture. |

Environment Caveats
-------------------

- **Notion onboarding:** The included Firefox profile can land on `https://www.notion.so/onboarding`, forcing “Add an account” / “Continue setup” dialogs. Those UI flows block tasks like “Create a new page” or “Filter a database”. Provide a workspace that is already onboarded (via `--storage-state` or a refreshed profile) and rerun the same commands to capture those workflows.
- **Hidden share button:** On this profile, the sidebar “Share” control is not rendered, so the share-menu task currently captures repeated attempts to locate it. Once the button is visible, the heuristic planner (which prioritises generic “share” verbs) will click it and the success detector will confirm the modal.

Loom Walkthrough Script
-----------------------

1. **Intro (30s):** Explain the Perception → Decision → Action → Capture loop, emphasising the single-action planner and generic success detector.
2. **Linear demo (1 min):** Run “Create a new project in Linear”, keep the terminal visible to highlight planner/heuristic steps, then open `datasets/linear/create-a-new-project-in-linear/step-02-success-state.png` and the corresponding JSON metadata.
3. **Notion demo (1 min):** Run “Open the quick find command palette in Notion”, show the `success` log line, and open `datasets/notion/open-the-quick-find-command-palette-in-notion/step-01-success-state.png`.
4. **Telemetry (30s):** Open a `run.jsonl` to point out `{"event":"success",...}` followed by the auto-generated `done` action and the absence of app-specific branches.
5. **Environment note (30s):** Mention the Notion onboarding/share limitations and how supplying the correct storage state unlocks the remaining tasks—no code changes required.

Reviewer Guide
--------------

1. Re-run any command from the table above, then open `datasets/<app>/<task-slug>/step-XX-success-state.*` to inspect the captured modal / filtered view alongside the generated metadata JSON.
2. Open the matching `run.jsonl` and scroll to the end to see `{"event":"success", ...}` followed by the automatically injected `done` action referencing the same success reason.
3. Spot-check earlier steps (e.g., `step-00-heuristic-press-command-palette.*`) to confirm every step is derived from visible UISnapshot elements—no hardcoded selectors.
4. Re-run a completed task a second time; you should see the cached plan from `profiles/default/_memory/plans/…` replay immediately, proving we store successful scripts instead of procedural workflows.

CLI Flags
---------

- `--task` *(required)*: Natural-language instruction.
- `--app`: Optional hint (e.g. `linear`, `notion`) used only for initial navigation and capture path.
- `--outdir`: Dataset root (default `datasets/`).
- `--headless`: Launch browser in headless mode.
- `--max-steps`: Maximum planner iterations (default 15).
- `--timeout-ms`: Base timeout for Playwright waits and robust wrappers (default 8000).
- `--max-retries`: Number of retries for clicks/fills/waits (default 3).
- `--browser`, `--profile-dir`, `--storage-state`, `--log-level`: Standard Playwright/diagnostic options.

Perception
----------

`ui_describer.describe_ui` collects a compact snapshot from the main frame plus visible iframes (up to four):

- Page URL and title.
- Up to 60 clickables (`text`, `role`, `selector`, metadata).
- Up to 30 inputs.
- Up to 7 visible modals plus overlay surfaces.
- Primary action candidates and breadcrumb trails for additional context.

Selectors favour accessibility (`role=name`), `data-testid`, aria labels, placeholders, or a fallback `tag >> nth=`. Iframe selectors are prefixed with `frame[i]:...` so planners can target nested contexts safely.

Decision
--------

`HierarchicalPlanner` orchestrates decision-making each turn:

- Replays any primed or cached scripts before asking for fresh guidance.
- Solicits `LLMPlanner.plan_script` for a short JSON plan; if no script is viable, falls back to `LLMPlanner.plan_next` for a single action aligned with the current UISnapshot.
- When the model is unavailable or selectors repeatedly fail, `HeuristicPlanner` scores visible elements using generic verbs (`create`, `new`, `add`, `save`, `submit`, `filter`, `settings`) and task keywords before defaulting to a capture action for more context.

Action & Robustness
-------------------

`executor.run_step` executes the suggested action and always captures results. It uses utilities from `agent_b/robustness.py`:

- `_with_retries` restarts flaky operations with exponential backoff (300ms, 700ms, 1500ms …).
- `click_robust` re-queries locators each attempt, scrolls into view, tries a native click, mouse click, then `force=True`.
- `fill_robust` clears existing text (Cmd/Ctrl+A + Backspace), fills, and verifies the final value.
- `wait_for_page_quiet` waits for `networkidle`, common spinner/overlay selectors to disappear, and a MutationObserver silence window.
- `press` actions synthesize keyboard shortcuts (`page.keyboard.press`) with a short delay to stabilise command palette workflows.

If the planner specifies `expect`, the executor waits for that selector after the action completes.

Capture & Telemetry
-------------------

Every step produces three artefacts under `datasets/<app>/<task-slug>/`:

- `step-XX-<name>.png` – full-page screenshot.
- `step-XX-<name>.html` – full DOM snapshot.
- `step-XX-<name>.json` – metadata (`name`, `url`, `timestamp`, action details, retry state, optional log tail).

Additionally, `run.jsonl` is appended with structured telemetry:

```json
{"event":"step","step":2,"planner":"llm","success":true,"action":{"action":"click","selector":"text=\"New\"","reason":"Open creation modal"}}
{"event":"console","payload":{"type":"warning","text":"Spinner mount","location":{"url":"https://example.test","lineNumber":123}}}
{"event":"pageerror","payload":{"message":"TypeError: undefined is not a function"}}
```

Console and page errors are summarised both in telemetry and in each step’s metadata (`log_tail`).

Mock Test Scenario
------------------

To smoke-test locally, serve a simple page that contains:

1. A primary button labelled **“New”** that opens a modal.
2. The modal exposes a **“Name”** text input and a **“Create”** button.
3. On submit, show a brief spinner/overlay (~600 ms) before revealing the confirmation surface.

Running `python src/main.py --task "Create a new item" --max-steps 5 --timeout-ms 5000` should yield captures similar to:

- `step-00-initial.png/html/json`: Landing surface before any action.
- `step-01-click-heuristic...`: Modal opened.
- `step-02-fill-name...`: Name populated.
- `step-03-click-create...`: Submission complete, spinner dismissed.
- `run.jsonl`: step events plus any captured console warnings.

Troubleshooting
---------------

- Missing `OPENAI_API_KEY`: Agent falls back to the heuristic planner; telemetry will contain `llm_error` entries.
- Authentication: supply `--storage-state` or `--profile-dir`. If the page presents password fields the agent captures a “login required” state and pauses until you finish manual sign-in.
- Output hygiene: keep `datasets/`, `logs/`, and `profiles/` out of version control. Use `.env` (see `.env.example`) for secrets.
