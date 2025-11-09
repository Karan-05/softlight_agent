## Control Flow Summary

1. **CLI (`src/main.py`)** parses the natural-language task and runtime flags, then calls `agent_b.runner.run_agent_task`.
2. **Runner (`src/agent_b/runner.py`)** bootstraps Playwright, applies stored credentials, and loops up to `max_steps`:
   - Calls `describe_ui` to capture a `UISnapshot`.
   - Loads cached action plans from plan memory / experience / semantic stores (see `plan_memory.py`, `experience_store.py`) and primes the hierarchical planner.
   - Invokes `HierarchicalPlanner.plan_next`, which may replay queued actions, delegate to the LLM planner, or fall back to the heuristic planner.
   - Each proposed `PlannerAction` flows through `planning_validation.validate_action_against_snapshot_and_intent`.
   - `executor.run_step` executes the action, records captures, and updates history.
   - `success.evaluate_success` inspects the new snapshot + history to decide whether the task is complete. On success, the runner emits a final `done` capture and persists the winning plan back to memory/experience.

## Perception

- `ui_describer.describe_ui` walks the DOM via Playwright locators and lightweight JS helpers, emitting `UIElement` records for clickables, inputs, modals, overlays, breadcrumbs, and “primary” buttons. Metadata includes role, label text, ancestor context, modal ownership, and `data-*` ids for reliable selectors.

## Planning Stack

1. **Hierarchical layer (`hierarchical_planner.py`)**
   - Maintains a queue of primed actions (`self._pending_actions`). When available, it revalidates and replays them before generating new steps.
   - Otherwise, it tries the **LLM planner** (`llm_planner.py`) by sending the current snapshot, task intent, and recent history. The LLM can return a multi-step script or a single action; every entry is validated against the snapshot and intent constraints.
   - If the LLM path fails or is unavailable, it defers to the **heuristic planner** (`heuristic_planner.py`), which uses intent-derived keywords, modal context, and element metadata to greedily pick the next action.

2. **LLM planner**
   - Builds a prompt with structured tables of visible clickables/inputs/modals plus inferred keywords (`_extract_keywords`), suggested names/summaries, and history traces.
   - Requires selectors to match exactly what `describe_ui` surfaced, and fill actions must reference entries from the “Visible inputs” section.

3. **Heuristic planner**
   - Parses the task via `intent.parse_intent` into verbs, target nouns, and explicit titles.
   - Scores clickables by synonym hits, intent alignment, modal scope, and history penalties.
   - Handles generic flows (create/open/filter/search/toggle/share) and destructive modals, always routing through the shared validator before returning an action.

## Validation (`planning_validation.py`)

- `validate_action_against_snapshot_and_intent` enforces:
  - Selector presence in the current snapshot collections.
  - Fill targets must be true text inputs/contenteditable fields.
  - Create-intent clicks cannot hit conflicting nouns (e.g. “issue” when target is “project”) or destructive controls.
  - Explicit titles must be typed into name/title-like inputs, and fill values must contain the literal title.

This function is called for **every** LLM action/script step, every heuristic decision, and (after the latest fixes) every cached plan step before execution.

## Execution & Robustness

- `executor.run_step` routes actions to Playwright, using helpers in `robustness.py` (`click_robust`, `fill_robust`, etc.) that retry with scrolling, keyboard shortcuts, and pointer fallbacks. Each step produces PNG/HTML/JSON captures plus console/page-error logs.

## Success Detection (`success.py`)

- Parses the task intent again and evaluates a set of intent-specific checks:
  - Modal/palette visibility, navigation activation, filter chips, toggle state, search results, and **creation** signals.
  - For create tasks, it requires a successful fill + submit sequence, the absence of destructive dialogs, and the new artifact text visible outside modals before emitting `event: "success"`.

## Memory & Replay

- `ActionPlanMemory` stores successful action traces per (app, task slug). On new runs, the runner loads these scripts, primes `HierarchicalPlanner`, and replays them *only* if the selectors are still reliable and pass validation. Failures increment a counter; after 3 strikes, the plan is pruned.
- `ExperienceStore` and `SemanticIndex` provide additional recall sources keyed by task text similarity.

All cached steps still flow through `validate_action_against_snapshot_and_intent`, ensuring outdated or destructive selectors are rejected and the planner falls back to fresh reasoning.***
