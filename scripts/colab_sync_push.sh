#!/usr/bin/env bash

set -u

REPO_ROOT="${REPO_ROOT:-/content/EgyptGPT}"
DRIVE_ROOT="${DRIVE_ROOT:-/content/drive/MyDrive/EgyptGPT_autoresearch}"
SYNC_SECONDS="${SYNC_SECONDS:-150}"
PUSH_LOG="$DRIVE_ROOT/git_push.log"

mkdir -p "$DRIVE_ROOT"

while true; do
  cp "$REPO_ROOT/results.tsv" "$DRIVE_ROOT/results.tsv" 2>/dev/null || true

  if [ -n "${GITHUB_TOKEN:-}" ] && cd "$REPO_ROOT" 2>/dev/null; then
    branch=$(git branch --show-current 2>/dev/null || true)
    if [ -n "$branch" ]; then
      sudo -u researcher env \
        HOME=/home/researcher \
        PATH="$PATH" \
        GITHUB_TOKEN="$GITHUB_TOKEN" \
        GIT_ASKPASS="$GIT_ASKPASS" \
        GIT_TERMINAL_PROMPT=0 \
        bash -lc "cd '$REPO_ROOT' && git push -u origin '$branch'" \
        >> "$PUSH_LOG" 2>&1 || true
    fi
  fi

  sleep "$SYNC_SECONDS"
done
