#!/usr/bin/env bash

set -u

REPO_ROOT="${REPO_ROOT:-/content/EgyptGPT}"
DRIVE_ROOT="${DRIVE_ROOT:-/content/drive/MyDrive/EgyptGPT_autoresearch}"
SYNC_SECONDS="${SYNC_SECONDS:-150}"

mkdir -p "$DRIVE_ROOT"

while true; do
  cp "$REPO_ROOT/results.tsv" "$DRIVE_ROOT/results.tsv" 2>/dev/null || true
  cp "$REPO_ROOT/RESEARCH_REPORT.md" "$DRIVE_ROOT/RESEARCH_REPORT.md" 2>/dev/null || true
  cp -r "$REPO_ROOT/out-glyphs-char" "$DRIVE_ROOT/out-glyphs-char" 2>/dev/null || true
  cp "$REPO_ROOT/data/egypt_char/meta_signs.pkl" "$DRIVE_ROOT/meta_signs.pkl" 2>/dev/null || true
  cp "$REPO_ROOT/data/egypt_char/meta.pkl" "$DRIVE_ROOT/meta.pkl" 2>/dev/null || true
  sleep "$SYNC_SECONDS"
done
