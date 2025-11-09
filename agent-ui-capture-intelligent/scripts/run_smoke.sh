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
