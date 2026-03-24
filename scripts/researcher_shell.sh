#!/usr/bin/env bash
# Launch Claude Code as the researcher user with GPU access.
# Run from the Colab terminal (as root) inside tmux.
# Pass "shell" as $1 to drop into a plain bash shell instead.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/content/EgyptGPT}"
SCRIPTS="$(cd "$(dirname "$0")" && pwd)"

echo "Running CUDA check..."
bash "$SCRIPTS/cuda_smoke_test.sh" || true

echo ""
echo "════════════════════════════════════════"
echo "  Researcher mode — GPU ready"
echo "════════════════════════════════════════"
echo ""

if [[ "${1:-}" == "shell" ]]; then
  exec sudo -u researcher env \
    HOME=/home/researcher \
    "PATH=/usr/local/nvidia/bin:$PATH" \
    "LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}" \
    bash -l
fi

exec sudo -u researcher env \
  HOME=/home/researcher \
  "PATH=/usr/local/nvidia/bin:$PATH" \
  "LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}" \
  bash -lc "cd $REPO_ROOT && claude --dangerously-skip-permissions"
