# Agent B – UI Workflow Capture Agent

*Softlight Engineering Take-Home Assignment – UI Workflow Capture Agent (Agent B)*

---

## Overview

This repository implements **Agent B**, part of a multi-agent system:

* **Agent A** receives natural language questions, e.g.:

  * “How do I create a project in Linear?”
  * “How do I filter a database in Notion?”
* **Agent B (this project)**:

  * Interprets the task
  * Drives a real browser using **Playwright** + an **LLM-driven planner**
  * Navigates the live app (Linear, Notion)
  * Captures **each important UI state** in the workflow:

    * URL states (e.g. list pages)
    * Non-URL states (modals, drawers, inline forms, confirmation banners)
  * Emits a structured dataset (`run.jsonl` + screenshots + DOM summaries) that shows how to perform the workflow end-to-end

The system is **not hardcoded** to specific flows. It uses generic perception + planning + execution so it can adapt to new tasks and different web apps at runtime.

---

## Key Features

* **Natural-Language → Actions**

  * Consumes tasks like:
    `--task "Create a new project in Linear with a random title"`
  * Infers intent (create, open, filter, toggle, search, etc.) from text.

* **UI Perception (Non-URL States Included)**

  * Captures:

    * Entry pages
    * Primary buttons and CTAs
    * Modals and drawers
    * Filled form states
    * Success / confirmation states
  * Works even when the URL doesn’t change.

* **LLM + Heuristic Planner**

  * LLM planner operates under a strict JSON schema.
  * Heuristic planner handles common patterns (“Create”, “New”, “Filter”, “Settings”) safely.
  * Both are constrained to selectors observed in the current DOM snapshot.

* **Robust Playwright Executor**

  * Validates visibility, waits for stability.
  * Retries on timeouts.
  * Operates only on validated selectors.
  * Captures screenshots and HTML when key states are reached.

* **Guardrails & Safety**

  * Prevents:

    * Pasting prompts into inputs
    * Loops of repeated actions
    * Clicking dismiss/cancel for creation flows
    * Using selectors that don’t exist in the latest snapshot

* **Dataset Generation**

  * Each run produces:

    * `run.jsonl` timeline (snapshots → plans → actions → outcomes)
    * Step-wise screenshots
    * DOM summaries / metadata
  * Organized **by task**, ready for inspection or training.

---

## Architecture

At a high level:

```text
src/
  main.py                 # CLI entrypoint
  agent_b/
    runner.py             # Orchestrates the agent loop
    ui_describer.py       # DOM → structured UI elements
    hierarchical_planner.py
    llm_planner.py
    heuristic_planner.py
    planning_validation.py
    executor.py
    success.py
    plan_memory.py
    experience_store.py
    semantic_index.py
```

### Flow

1. **Entrypoint (`src/main.py`)**

   * Parses:

     * `--task` (natural language)
     * `--app` (e.g. `linear`, `notion`)
     * `--storage-state` (Playwright auth)
     * `--max-steps`, `--timeout-ms`, `--browser`
   * Calls the runner with these settings.

2. **Runner (`agent_b/runner.py`)**

   * Starts Playwright (Firefox).
   * Loads authenticated session (`--storage-state`).
   * For each step:

     * Ask `ui_describer` for a snapshot.
     * Call planners to choose the next action.
     * Validate, execute, and log.
     * Check success / stopping conditions.

3. **Perception (`agent_b/ui_describer.py`)**

   * Builds a **UISnapshot**:

     * Clickables, inputs, modals, primary actions
     * Roles, labels, text, selectors
   * Normalizes into `UIElement` objects, which are the only things planners can target.

4. **Planning**

   * **Hierarchical Planner**

     * Coordinates:

       * Previously successful plans (from memory)
       * LLM proposals
       * Heuristic fallbacks
     * Ensures each suggestion goes through validation.
   * **LLM Planner**

     * Given:

       * Task text
       * Current snapshot
       * Recent history
     * Returns a **single JSON action** or short script:

       * `action`, `selector`, `value`, `capture_name`, `reason`, `expect`, `success_hint`
     * Must reuse selectors from the snapshot (no hallucinated CSS).
   * **Heuristic Planner**

     * Pattern-matches common UI flows:

       * “Create / New / Add / Save”
       * Filters, search, toggles
       * Modals and submit buttons
     * Used when the LLM is unnecessary or ambiguous.

5. **Validation (`agent_b/planning_validation.py`)**

   * Ensures:

     * Selector exists in latest snapshot.
     * Action fits the role (e.g. `fill` on input only).
     * Task intent and element text don’t conflict (e.g. project vs issue).
     * Creation flows avoid dismiss / cancel buttons.
   * Rejects unsafe or nonsensical actions before execution.

6. **Executor (`agent_b/executor.py`)**

   * Executes `click`, `fill`, `press`, `wait`, `capture`, etc.
   * Waits for visibility and page stability.
   * On each capture:

     * Saves screenshot + metadata.
   * Returns success/failure and updated context to the runner.

7. **Success Detection (`agent_b/success.py`)**

   * Intent-aware heuristics:

     * For “create”:

       * Form filled
       * Submit/CTA clicked
       * New entity visible
   * Triggers `done` when conditions are satisfied and ensures final capture.

8. **Memory (`plan_memory`, `experience_store`, `semantic_index`)**

   * Stores successful trajectories.
   * Uses them as hints for similar future tasks (while still validating live).

---

## Dataset Format

All runs are **organized by task** so they satisfy:

> “Captured UI states for 3–5 tasks across 1–2 apps, organized by task, with blurbs.”

Example layout (conceptual):

```text
datasets/
  linear/
    create-project/
      run.jsonl
      step-00-entry.png
      step-01-projects-page.png
      step-02-create-modal-open.png
      step-03-form-filled.png
      step-04-success-or-confirmation.png
    filter-issues-high-priority/
      run.jsonl
      step-*.png
    notifications-settings/
      run.jsonl
      step-*.png
  notion/
    create-page-with-summary/
      run.jsonl
      step-*.png
    filter-database-completed/
      run.jsonl
      step-*.png
    workspace-settings-modal/
      run.jsonl
      step-*.png
```

Each `run.jsonl` includes:

* `snapshot_hint` events (what the agent sees)
* `step` events:

  * planned action
  * execution result
  * references to captures
* Optional console/page errors for debugging

Each folder is a **self-contained workflow dataset** documenting how Agent B completed a specific task.

---

## Example Tasks Covered

By design, the same agent handles multiple workflows across Linear and Notion using natural language instructions and live DOM inspection.

### Linear (selected examples)

* Create a new project and capture the confirmation modal
* Create a new project in Linear (random title)
* Filter issues to show only high priority items
* Open workspace notification settings and toggle a switch
* Navigate to roadmap via sidebar
* Open command palette and search for integrations
* Open create issue modal and capture it
* Switch to “My Issues” and capture state
* Open projects page and capture the “Add project” UI
* Inspect notifications feed and capture state

### Notion (selected examples)

* Create a new page and add a short summary
* Filter a database to show completed tasks
* Open the quick find command palette
* Duplicate the default meeting notes template
* Create a new database view for tasks
* Open workspace settings and capture the modal
* Create a toggle list
* Add a reminder to a block
* Capture the share menu for the current page
* Create a new database
* Change database filters

These tasks:

* Are **not** hardcoded as flows in the code.
* Are passed in as plain-text `--task` strings.
* Demonstrate generalization across multiple workflows and non-URL states.

---

## Quickstart

### Prerequisites

* Python 3.11+
* Playwright installed + browsers:

  ```bash
  pip install -r requirements.txt
  python -m playwright install
  ```
* Node/npm if you prefer to use `npx playwright` helpers.
* Authenticated storage states:

  * `secrets/linear-storage.json`
  * `secrets/notion-storage.json`

### Auth Setup (Example: Linear)

```bash
npx playwright codegen https://linear.app \
  --save-storage=secrets/linear-storage.json
```

Log in once in the opened browser; the storage state is saved and reused.

Do the same for Notion (or your chosen apps), saving into `secrets/notion-storage.json`.

---

## Run a Single Task

### Linear

```bash
python src/main.py \
  --task "Create a new project in Linear with a random title and capture each step" \
  --app linear \
  --browser firefox \
  --storage-state secrets/linear-storage.json \
  --max-steps 15 \
  --timeout-ms 10000
```

### Notion

```bash
python src/main.py \
  --task "Create a database and filtered view in Notion" \
  --app notion \
  --browser firefox \
  --storage-state secrets/notion-storage.json \
  --max-steps 20 \
  --timeout-ms 10000
```

Each run:

* Navigates the app based on the task.
* Captures key UI states (including non-URL states).
* Writes `run.jsonl` + screenshots/HTML into the appropriate dataset directory.

---

## Run the Full Task Suite (Recommended for Dataset Generation)

A helper script (e.g. `scripts/run_all_tasks.sh`) is included/outlined to generate multiple workflows across Linear and Notion.

```bash
#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python}
LINEAR_STATE=${LINEAR_STATE:-secrets/linear-storage.json}
NOTION_STATE=${NOTION_STATE:-secrets/notion-storage.json}

run_task() {
  local task="$1"
  local app="$2"
  local state="$3"
  echo "=== Running: $task (app=${app:-none}) ==="
  if [[ -n "$state" ]]; then
    $PYTHON src/main.py \
      --task "$task" \
      ${app:+--app "$app"} \
      --browser firefox \
      --storage-state "$state" \
      --max-steps 15 \
      --timeout-ms 10000
  else
    $PYTHON src/main.py \
      --task "$task" \
      ${app:+--app "$app"} \
      --browser firefox \
      --max-steps 15 \
      --timeout-ms 10000
  fi
}

LINEAR_TASKS=(
  "Create a new project and capture the confirmation modal"
  "Filter issues to show only high priority items"
  "Open workspace notification settings and toggle a switch"
  "Navigate to the roadmap view using the sidebar"
  "Open the command palette and search for integrations"
  "Create a new view named Automation Board"
  "Open the create issue modal and capture it"
  "Switch to the My Issues tab and capture it"
  "Open the projects page and capture the Add project modal"
  "Inspect the notifications feed and capture its state"
  "Create a new project in Linear"
)

NOTION_TASKS=(
  "Create a new page and add a short summary"
  "Filter a database to show completed tasks"
  "Open the quick find command palette"
  "Duplicate the default meeting notes template"
  "Create a new database view for tasks"
  "Open workspace settings and capture the modal"
  "Create a toggle list on the current page"
  "Add a reminder to the first block"
  "Switch to the favorites section in the sidebar"
  "Capture the share menu for the current page"
  "Create a new Database in Notion"
  "Change the database filter in Notion"
)

for task in "${LINEAR_TASKS[@]}"; do
  run_task "$task" "linear" "$LINEAR_STATE"
done

for task in "${NOTION_TASKS[@]}"; do
  run_task "$task" "notion" "$NOTION_STATE"
done
```

### How to Use

```bash
chmod +x scripts/run_all_tasks.sh
./scripts/run_all_tasks.sh
```

This will:

* Execute a rich set of workflows across Linear + Notion.
* Produce **multiple task-organized datasets** with captured UI states.
* Clearly demonstrate:

  * 3–5+ workflows across 1–2 apps
  * Thoughtful handling of non-URL states
  * Generalization from natural language tasks

---

## Loom Demo

Add a short Loom showcasing:

* Running one or more tasks with `src/main.py` or `run_all_tasks.sh`
* The agent:

  * Navigating to the right pages
  * Opening modals
  * Filling forms
  * Clicking the correct CTAs
  * Capturing screenshots and producing `run.jsonl`
* Where the outputs live in the repo / dataset

**Placeholder:** `▶ Loom demo: <ADD_LINK_HERE>`

---

## Reliability, Limitations & Extensibility

**Reliability**

* Per-action timeouts + global `--max-steps`
* Selector + intent validation before execution
* Loop detection, especially around repeated clicks
* Defenses against echoing prompt text into UI fields

**Limitations**

* UI/selector drift from app updates may require light tuning.
* Auth states depend on session longevity.

**Extensibility**

* Add new apps by:

  * Providing `--app <name>` and a corresponding auth state.
  * Letting the existing perception/planning pipeline operate on the new DOM.
* Swap or upgrade LLMs inside `llm_planner` without changing overall contracts.
* Extend capture logic for toasts, error banners, or additional metadata.

---

## Personal Note

I’m genuinely excited about this challenge and the chance to work with the Softlight team.

This project is built to reflect how I like to engineer systems:

* clear separation of perception, planning, execution, and memory,
* strong guardrails around an LLM core,
* and a focus on **real**, end-to-end workflows instead of toy demos.

The idea of turning messy, dynamic web UIs into reliable, re-usable, agent-friendly workflows is exactly the kind of problem I want to work on—combining practical automation, solid infra, and intelligent tooling that other engineers (and agents) can trust.

Looking forward to the opportunity to take this further with you.
