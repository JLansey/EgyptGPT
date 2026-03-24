#!/usr/bin/env bash
# Full Colab environment setup: researcher user, branch, results, CUDA, Claude.
# Called from the notebook with env vars: GITHUB_TOKEN, BRANCH, CLAUDE_MODEL,
# REPO_ROOT, DRIVE_ROOT.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/content/EgyptGPT}"
DRIVE_ROOT="${DRIVE_ROOT:-/content/drive/MyDrive/EgyptGPT_autoresearch}"
BRANCH="${BRANCH:?BRANCH is required}"
CLAUDE_MODEL="${CLAUDE_MODEL:?CLAUDE_MODEL is required}"

SCRIPTS="$REPO_ROOT/scripts"
ASKPASS="$SCRIPTS/colab_git_askpass.sh"

run_as_researcher() {
  sudo -u researcher env \
    HOME=/home/researcher \
    "PATH=$PATH" \
    "GITHUB_TOKEN=$GITHUB_TOKEN" \
    "GIT_ASKPASS=$ASKPASS" \
    GIT_TERMINAL_PROMPT=0 \
    "$@"
}

# ── 1. Researcher user, permissions, git config, token validation ──
bash "$SCRIPTS/setup_researcher.sh"

# ── 2. Branch: fetch from remote if it exists, otherwise create ──
run_as_researcher bash -lc "cd $REPO_ROOT && git fetch origin $BRANCH" 2>/dev/null || true
if run_as_researcher bash -lc "cd $REPO_ROOT && git checkout $BRANCH" 2>/dev/null; then
  echo "Checked out existing branch: $BRANCH"
else
  sudo -u researcher bash -lc "cd $REPO_ROOT && git checkout -b $BRANCH"
  echo "Created new branch: $BRANCH"
fi

# ── 3. results.tsv: local → Drive → blank ──
REPO_RESULTS="$REPO_ROOT/results.tsv"
DRIVE_RESULTS="$DRIVE_ROOT/results.tsv"

if [[ -s "$REPO_RESULTS" ]]; then
  echo "results.tsv exists locally ($(wc -c < "$REPO_RESULTS" | tr -d ' ') bytes)"
elif [[ -f "$DRIVE_RESULTS" ]]; then
  cp "$DRIVE_RESULTS" "$REPO_RESULTS"
  chown researcher:researcher "$REPO_RESULTS"
  echo "Restored results.tsv from Drive ($(wc -c < "$DRIVE_RESULTS" | tr -d ' ') bytes)"
else
  touch "$REPO_RESULTS"
  chown researcher:researcher "$REPO_RESULTS"
  echo "Created blank results.tsv"
fi

# ── 4. CUDA smoke test (non-fatal) ──
bash "$SCRIPTS/cuda_smoke_test.sh" || true

# ── 5. Configure Claude model ──
bash "$SCRIPTS/setup_claude.sh" "$CLAUDE_MODEL"

echo ""
echo "Branch: $BRANCH"
echo "Model:  $CLAUDE_MODEL"
echo "Drive:  $DRIVE_ROOT"
