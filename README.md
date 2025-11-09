# Agent B – UI Workflow Capture Agent
*Softlight Engineering Take-Home Assignment – UI Workflow Capture Agent (Agent B)*

---

## Overview
Agent B collaborates with other members of Softlight’s multi-agent system. Agent A receives natural-language questions (“How do I create a project in Linear?”) and delegates them to Agent B. Agent B interprets the instruction, drives a real browser via Playwright + an LLM planner, captures every important UI state (including modals and confirmation banners that never change the URL), and emits a structured dataset that teaches downstream agents or humans how to complete the workflow.

---

## Key Features
- **LLM Planner** – Strict JSON schema, robust selector guidance, loop-safe reasoning, and capture hints for all key states.
- **Playwright Executor** – Sync Playwright driver that validates visibility, handles retries/timeouts, and converts actions to real DOM operations.
- **Non-URL Capture** – Records entry pages, controls, modals/forms, populated inputs, success views—everything Agent A needs to replicate the workflow.
- **Dataset Generation** – Saves sequential screenshots, DOM summaries, and `run.jsonl` telemetry under `captures/<task>/`.
- **Guardrails** – Per-action timeouts, repeated-action detection, selector validation, and “never paste the prompt” policies keep the system reliable.

---

## Architecture
```
┌───────────────────────────────────────────────────────────────┐
│                          Agent B                              │
├──────────────────────┬────────────────────────┬───────────────┤
│ Context Extractor    │ Planner                │ Executor      │
│ (src/agent/context.py│ (src/agent/planner.py) │ (src/agent/   │
│ + Playwright DOM     │ • strict system prompt │ executor.py)  │
│ snapshot)            │ • JSON-only actions    │ • Playwright   │
│ • URL/title/modals   │ • capture recommendations│ ops + waits  │
│ • Clickables/inputs  │ • guardrails + retries │ • success/fail │
├──────────────────────┴────────────────────────┴───────────────┤
│ Run Logger & Dataset (src/agent/run_logger.py)                │
│ • logs/agent-<ts>.log + logs/run-<ts>.jsonl                   │
│ • captures/<slug>/<step>-<capture>.png                        │
│ • Combined trace feeds Agent A and future training pipelines  │
└───────────────────────────────────────────────────────────────┘
```

### Planner
- Interprets the task + DOM summary using a strict system prompt.
- Emits exactly one JSON object per step with fields such as `action`, `selector`, `value`, `capture_name`, `reason`, `expect`, `success_hint`.
- Never emits Markdown, never pastes its instructions into input fields, and targets robust selectors (roles, aria labels, placeholders).
- Encourages collecting ~3–6 captures per workflow (entry, control visible, modal open, form filled, success state).

### Executor
- Implements `click`, `fill`, `press`, `wait_for`, `capture`, etc., via Playwright.
- Waits for elements, retries on timeouts, surfaces structured success/failure info to the planner and logger.

### Context Extractor
- Builds lightweight page/DOM summaries each loop (URL, title, clickables, inputs, breadcrumbs, modal metadata).

### Run Logger & Dataset
- Logs each planner/executor step to `run.jsonl`.
- Stores screenshots under `captures/<slugified-task>/<step>-<capture_name>.png`.
- Produces `agent-<timestamp>.log` for debugging, plus high-level DOM summaries for downstream analysis.

---

## Quickstart

### Prerequisites
- Python 3.11
- Node/npm (for Playwright helpers)
- Playwright browsers (`chromium` at minimum)

### Installation
```bash
git clone <repo>
cd agent-ui-capture-intelligent
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m playwright install
```

### Auth Setup
Capture a saved Playwright auth state for each app (Linear example):
```bash
npx playwright codegen https://linear.app --save-storage=secrets/linear-storage.json
```
Reuse the resulting `--storage-state` file for subsequent runs. Do the same for Notion or other apps.

### Example Runs
**Linear**
```bash
python src/main.py \
  --task "Create a new project in Linear with a random title" \
  --app linear \
  --storage-state secrets/linear-storage.json \
  --max-steps 18 \
  --timeout-ms 12000
```

**Notion**
```bash
python src/main.py \
  --task "Create a database and filtered view in Notion" \
  --app notion \
  --storage-state secrets/notion-storage.json \
  --headless \
  --max-steps 20
```

---

## How It Works (Step-by-Step)
1. **Task ingestion** – `src/main.py` parses CLI flags (`--task`, `--app`, `--storage-state`, `--max-steps`, `--timeout-ms`, `--headless`).
2. **Context snapshot** – `src/agent/context.py` summarizes the live page: URL, title, clickables, inputs, modals, owner dialog info.
3. **Planner call** – `LLMPlanner` (src/agent/planner.py) receives the DOM summary + task and returns one JSON action that fits the schema. It never emits extra text.
4. **Executor** – `ActionExecutor` (src/agent/executor.py) maps the JSON to Playwright operations, waits for visibility, handles timeouts, and reports structured success/failure data.
5. **Captures** – The executor triggers screenshots whenever the planner requests `capture` or when configured key states (modal open, form populated, success banner) are reached.
6. **Logging** – `RunLogger` (src/agent/run_logger.py) writes every event to `run.jsonl`, captures step metadata, and mirrors console/page errors.
7. **Stopping** – Loop ends when the planner outputs `done`, success heuristics fire, guardrails trip, or `--max-steps` is reached.

Because all captures are driven by context, Agent B records non-URL states—modals, wizards, confirmation banners—ensuring workflows remain reproducible even when the browser location stays constant.

---

## Dataset Format
```
captures/<slugified-task>/
├── step-00-entry.png / .json
├── step-01-open-control.png / .json
├── step-02-modal-open.png / .json
├── step-03-form-filled.png / .json
├── step-04-success.png / .json
└── run.jsonl
```

### run.jsonl (excerpt)
```json
{
  "event": "step",
  "step": 4,
  "action": {
    "action": "fill",
    "selector": "role=textbox[name=/Project name/i]",
    "value": "Agent Demo",
    "capture_name": "form-filled",
    "reason": "Populate project name",
    "expect": "Form ready to submit",
    "success_hint": "New project dialog still open"
  },
  "success": true,
  "planner": "llm",
  "url": "https://linear.app/.../projects",
  "dom_summary": { "clickables": 42, "inputs": 6, "modals": 1 },
  "screenshot": "captures/create-a-project/step-04-form-filled.png"
}
```

Use the `run.jsonl` timeline to correlate each action with its screenshot and DOM summary. This dataset becomes ground truth for training other agents or for documenting internal workflows.

---

## Loom Demo
- ▶ **Loom demo: <ADD_LINK_HERE>** – Walkthrough showing the agent receiving a Linear task, opening the modal, filling the form, capturing key UI states, and producing the dataset.

---

## Testing Expectations
The repository demonstrates 3–5 real tasks across at least two apps (e.g., Linear + Notion):
1. Create a new project in Linear.
2. Filter issues by status/label in Linear.
3. Create a database and filtered view in Notion.
4. Update workspace settings.

No workflows are hardcoded—the agent always consumes `--task` at runtime. Because planning relies on textual roles/labels (not app-specific scripts), the approach generalizes to unseen tasks and apps so long as they expose accessible DOM metadata.

---

## Extensibility
- **Add new apps** – Provide an auth state and call `src/main.py --task "…" --app <new-app>`; the planner/executor automatically adapts to the DOM summary.
- **Swap LLM providers** – Replace the OpenAI client inside `src/agent/planner.py` while preserving the JSON schema and prompt contract.
- **Customize capture rules** – Adjust planner prompts or extend the executor to capture additional states (toasts, error banners, etc.).

---

## Reliability & Guardrails
- Per-action timeouts (`--timeout-ms`) and global loop limits (`--max-steps`).
- Selector validation detects conflicting nouns (“issue” vs “project”) and rejects brittle CSS.
- Repeated-action detection prevents loops (e.g., re-opening “Add project” repeatedly).
- Prompt-injection defense: planner never mirrors its instructions into UI fields.
- Executor verifies fill results and surfaces failures to trigger replanning.

---

## Limitations & Future Work
- Third-party UIs change frequently; some selectors may degrade over time.
- Auth/session management depends on upstream rate limits and cookie TTLs.
- Future enhancements: cached multi-step plans, richer DOM embeddings, human-in-the-loop annotations for rare workflows.

---

## License
MIT License © Softlight Engineering. Use responsibly; never commit secrets, storage-state files, or raw customer datasets to the repository.

Agent B is intentionally app-agnostic—every improvement should continue to rely on live UISnapshots and natural-language instructions rather than hardcoded flows. Hook it up to Agent A, point it at a new workflow, and you’ll get the screenshots, actions, and metadata needed to teach the rest of your agent stack.
