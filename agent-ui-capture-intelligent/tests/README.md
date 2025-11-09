# Test & Scenario Guide

This folder gathers lightweight validation assets for Agent B’s perception-driven loop.

## Automated checks

The `pytest` suite covers planner fallbacks, executor selector handling, and retrieval stores without requiring Playwright:

```bash
pip install pytest
pytest tests/test_decision_flow.py tests/test_executor_utils.py tests/test_experience_store.py tests/test_semantic_index.py
```

## Manual smoke scripts

Run a mix of natural-language tasks (add `--storage-state` or `--profile-dir` for authenticated surfaces):

```bash
python src/main.py --task "Create a new project and capture the confirmation modal" --app linear --max-steps 12
python src/main.py --task "Filter issues to show only high priority items" --app linear --max-steps 10
python src/main.py --task "Open workspace notification settings and toggle a switch" --app linear --max-steps 10
python src/main.py --task "Create a new page and add a short summary" --app notion --max-steps 12
python src/main.py --task "Navigate using the sidebar to the roadmap view" --max-steps 8
```

Verify that:

- Each run emits multiple `step-XX-*` captures in `datasets/<app>/<task-slug>/` (e.g., `datasets/notion/create-a-new-page-and-add-a-short-summary/`).
- At least one step records a non-URL state (modal, overlay, spinner, success banner).
- No app-specific helper functions are invoked (the loop should remain Perception → Decision → Action → Capture).

## Sample artefacts

`tests/samples/` contains trimmed JSON logs that demonstrate how modal captures, retries, and telemetry entries are structured. Use them when documenting flows or validating log ingestion without running Playwright end-to-end.
