# Repository Guidelines

## Project Structure & Module Organization
Agent B's runtime code lives in `agent-ui-capture-intelligent/src/`. The `agent_b/` package implements the Perception → Decision → Action → Capture loop (config, planners, executor, capturer). Deterministic app-specific flows are under `apps/` (e.g., `linear_workflows.py`). UI artifacts land in `datasets/<app>/<task-slug>/`, while run metadata is logged to `logs/` and persisted browser profiles to `profiles/`. Keep large media files out of git; commit only curated examples or compressed captures.

## Build, Test, and Development Commands
Run all commands from `agent-ui-capture-intelligent/`:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m playwright install
OPENAI_API_KEY=sk-... python src/main.py --task "Create a project in Linear" --app linear
python src/main.py --task "Set project \"Roadmap\" priority to High" --app linear --storage-state path/to/linear.json
python src/main.py --task "Smoke test navigation" --headless --max-steps 5
pytest tests/test_task_graphs.py
```

The Linear examples demonstrate orchestrated DAG runs using stored credentials (`--storage-state`). Use `--profile-dir` when reusing browser profiles outside storage snapshots.

## Coding Style & Naming Conventions
Follow 4-space indentation, type-annotated functions, and module-level docstrings as shown in `src/agent_b/*.py`. Keep module and variable names in `snake_case`, classes in `PascalCase`, and constants in `UPPER_SNAKE_CASE`. Prefer async Playwright APIs unless a deterministic flow requires synchronous calls. Use `logging` over `print`, and keep configuration indirection inside `config.py`.

## Testing Guidelines
Quick structural checks live under `tests/` (run `pytest tests/test_task_graphs.py`). For end-to-end validation use the provided smoke commands or create Playwright scripts beside the tests folder. Keep synthetic logs/captures in `tests/samples/` for documentation and scrub real customer data before sharing.

## Commit & Pull Request Guidelines
Write imperative, scoped commit messages (`feat: add heuristic planner fallback`). Group related changes into one commit whenever possible and avoid committing generated datasets, logs, or profile directories. Pull requests should describe the scenario exercised, include reproduction commands, and link to any Linear issues. Attach before/after screenshots or capture paths when UI behaviour changes.

## Security & Configuration Tips
Never commit `.env`, `profiles/`, or raw `datasets/` captures that might contain credentials. Supply `OPENAI_API_KEY` via environment variables and document any additional secrets in the PR. Review logs for sensitive text before sharing and scrub Linear URLs that include team identifiers.
