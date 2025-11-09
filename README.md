Agent B: Perception-Driven UI Agent
===================================

Agent B turns natural-language requests into reliable UI automations by looping through **Perception → Decision → Action → Capture** inside a real browser. Every iteration inspects the live DOM, picks exactly one JSON action (`goto`, `click`, `fill`, `press`, `wait_for`, `capture`, or `done`), executes it with robust Playwright helpers, then captures evidence before reasoning about the next move. There are **no hand-authored workflows** or app-specific selectors—plans are grounded entirely in the current UISnapshot plus the user’s instruction.


Quick Start
-----------

Requirements: Python 3.9+, Playwright browsers, an OpenAI API key (for the LLM planner; heuristics work without it).

```bash
cd agent-ui-capture-intelligent
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m playwright install
```

Basic CLI usage (always run from the repository root):

```bash
OPENAI_API_KEY=sk-... python src/main.py \
  --task "Create a new project named Agent Project in Linear" \
  --app linear \
  --storage-state path/to/linear.json \
  --max-steps 18 \
  --timeout-ms 12000
```

Other helpful commands:

| Scenario | Command |
| --- | --- |
| Headless smoke test | `python src/main.py --task "Smoke test navigation" --headless --max-steps 5` |
| Reuse a persistent browser profile | `python src/main.py ... --profile-dir profiles/linear` |
| Run unit tests | `pytest tests/test_task_graphs.py` |


How It Works
------------

### Perception

`ui_describer.describe_ui` gathers a condensed UISnapshot from the page (and visible iframes):

* URL, title, detected app hints.
* Up to 60 clickables, 30 inputs, modals + overlays, breadcrumbs, and “primary action” candidates.
* Rich metadata for each element (text, role, aria labels, owner dialog/menu ancestry, state flags, etc.).

This structured view is the sole source of truth for planners—no hidden selectors or app-specific shortcuts exist.

### Decision

`HierarchicalPlanner` coordinates three planning stages, always emitting a single JSON action per loop:

1. **Memory/script replay** – successful action traces from previous runs can be re-used when selectors remain reliable.
2. **LLM planning** – `LLMPlanner` generates either a short script or a single action entirely from the live UISnapshot. Responses are validated against snapshot affordances and the parsed intent before execution.
3. **Heuristic fallback** – `HeuristicPlanner` keeps the loop progressing without model access by scoring visible affordances using intent cues (verbs, nouns, modals, destructive dialogs, etc.).

The planner also enforces safety rails:

* Validation rejects selectors that are missing, conflicting (e.g., clicking “Create issue” while the task targets “project”), or illegal for the requested intent.
* When a creation modal is open and an explicit title was filled, the planner will automatically fire the modal’s `Create/Add` button instead of letting scripts re-trigger background launchers.
* Self-healing avoids selectors that recently failed and skips unreliable ones remembered across runs.

### Action & Robustness

`executor.run_step` applies the chosen action with Playwright and always captures the outcome:

* `click_robust` and `fill_robust` re-query locators, scroll into view, retry with exponential backoff, and verify input state. Non-fillable elements raise `NonFillableElementError` so the planner can re-plan safely.
* Every action gets a PNG, DOM HTML, and JSON metadata capture stored under `datasets/<app>/<task-slug>/step-XX-*`.
* Telemetry for each step (planner source, selector, success flag, console/page errors) streams into `run.jsonl`.

### Capture & Success Detection

After each step, `success.evaluate_success` examines the latest snapshot/history to decide whether the task is done. Success heuristics stay generic—looking for modal titles, toggled switches, new artifact headings, etc.—and they never accept destructive dialogs as proof for create/open intents. When success is detected, the runner logs `{"event":"success", ...}`, emits a final `done` action with a “success-state” capture, and stores the verified script for future warm starts.


Project Layout
--------------

```
agent-ui-capture-intelligent/
├── src/
│   └── agent_b/
│       ├── main.py / runner.py        # CLI entry + Perception→Decision→Action→Capture loop
│       ├── perception.py / ui_describer.py
│       ├── hierarchical_planner.py / llm_planner.py / heuristic_planner.py
│       ├── planning_validation.py     # Selector + intent safety checks
│       ├── executor.py / robustness.py
│       ├── success.py                 # Intent-aware success signals
│       └── … (experience_store, memory, config, etc.)
├── datasets/<app>/<task-slug>/        # PNG/HTML/JSON artifacts + run.jsonl telemetry
├── logs/                              # Structured planner/executor logs
└── tests/                             # Lightweight structural tests (pytest)
```


Running Tasks & Inspecting Output
---------------------------------

1. Execute `src/main.py` with your task, providing either `--storage-state` or `--profile-dir` so Linear/Notion sessions stay authenticated.
2. After the run, open `datasets/<app>/<task-slug>/run.jsonl` to review the timeline. Each `step` entry includes the planner source, selector, success flag, and capture filenames.
3. Look at `step-XX-*.png` / `.html` to debug mis-clicks or confirm success-state captures.

Tips:

* Use `--max-steps` and `--timeout-ms` to tune exploration depth vs. responsiveness.
* When iterating on planners, delete just the failing dataset folder so new captures don’t mix with old runs.


Testing & Quality Gates
-----------------------

* Unit / structural tests: `pytest tests/test_task_graphs.py`
* Manual smoke tests: run representative CLI commands (Linear create, Notion toggles, etc.) with `--headless` to catch regressions quickly.
* Linting/formatting: this repo relies on `pyproject` defaults; ensure your editor keeps files ASCII-only unless the source already contains Unicode.


Security & Data Hygiene
-----------------------

* Never commit `.env`, `profiles/`, raw `datasets/`, or secrets to version control.
* Scrub real customer data before sharing captures. Synthetic samples belong under `tests/samples/`.
* Supply `OPENAI_API_KEY` via environment variables (or skip it entirely to use the heuristic planner).


Troubleshooting
---------------

| Symptom | Likely Cause / Fix |
| --- | --- |
| Planner keeps reopening “Add project” instead of submitting | Ensure the creation modal is visible; the built-in modal submit override triggers only after the explicit title fill succeeds. |
| Agent stalls on login screens | Provide `--storage-state` or `--profile-dir`; otherwise the snapshot will keep showing email/password inputs. |
| Playwright errors about missing browsers | Re-run `python -m playwright install` in the repo’s virtualenv. |


Contributing
------------

1. Create a feature branch.
2. Keep changes modular, add inline comments only for non-obvious heuristics.
3. Run the relevant CLI smoke tests + `pytest`.
4. Update this README or docs if behavior changes (e.g., new planner safeguards).
5. Use imperative commit messages (`feat:`, `fix:`, `chore:`) and include reproduction commands in your PR description.

Agent B stays intentionally app-agnostic: any improvement should continue to rely on the live UISnapshot and natural-language intent—not on hardcoded URLs, product-specific logic, or cached scripts that were not produced by real UI runs.
