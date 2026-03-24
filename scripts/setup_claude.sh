#!/usr/bin/env bash
# Configure Claude Code model preference for the researcher user.
set -euo pipefail

MODEL="${1:?Usage: setup_claude.sh <model-name>}"

mkdir -p /home/researcher/.claude
printf '{"model": "%s"}\n' "$MODEL" > /home/researcher/.claude/settings.json
chown -R researcher:researcher /home/researcher/.claude

echo "Claude Code configured: $MODEL"
