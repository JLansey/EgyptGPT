#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/content/EgyptGPT}"
DRIVE_ROOT="${DRIVE_ROOT:-/content/drive/MyDrive/EgyptGPT_autoresearch}"
ASKPASS="$REPO_ROOT/scripts/colab_git_askpass.sh"

if [[ -z "${GITHUB_TOKEN:-}" ]]; then
  echo "ERROR: GITHUB_TOKEN is not set." >&2
  exit 1
fi

# ── Create non-root user (Claude Code blocks --dangerously-skip-permissions under root) ──
useradd -m researcher 2>/dev/null || true
passwd -d researcher

# ── Ownership & device permissions ──
chown -R researcher:researcher "$REPO_ROOT"
chmod a+rw /dev/nvidia* /dev/dri/* 2>/dev/null || true

# ── Drive access (non-recursive — just the mount points the researcher needs) ──
chmod a+rx /content/drive 2>/dev/null || true
chmod a+rx /content/drive/MyDrive 2>/dev/null || true
mkdir -p "$DRIVE_ROOT"
chmod a+rwx "$DRIVE_ROOT"

# ── Git config ──
sudo -u researcher git config --global --add safe.directory "$REPO_ROOT"
sudo -u researcher git config --global user.email "autoresearch@colab"
sudo -u researcher git config --global user.name "autoresearch"

# ── Ensure helper scripts are executable ──
chmod +x "$ASKPASS" "$REPO_ROOT/scripts/colab_sync_push.sh"

# ── Validate GitHub token ──
output=$(sudo -u researcher env \
  HOME=/home/researcher \
  "PATH=$PATH" \
  "GITHUB_TOKEN=$GITHUB_TOKEN" \
  "GIT_ASKPASS=$ASKPASS" \
  GIT_TERMINAL_PROMPT=0 \
  bash -lc "cd $REPO_ROOT && git ls-remote --exit-code origin HEAD" 2>&1) || {
    echo "ERROR: GITHUB_TOKEN could not authenticate to origin." >&2
    echo "Check that the token is valid and has repo push access." >&2
    echo "$output" >&2
    exit 1
  }

# ── Persist token so researcher shell can push ──
TOKEN_FILE="/home/researcher/.github_env"
cat > "$TOKEN_FILE" <<EOF
export GITHUB_TOKEN="$GITHUB_TOKEN"
export GIT_ASKPASS="$ASKPASS"
export GIT_TERMINAL_PROMPT=0
EOF
chmod 600 "$TOKEN_FILE"
chown researcher:researcher "$TOKEN_FILE"

BASHRC="/home/researcher/.bashrc"
SOURCE_LINE="[ -f $TOKEN_FILE ] && source $TOKEN_FILE"
grep -qF "$SOURCE_LINE" "$BASHRC" 2>/dev/null || echo -e "\n$SOURCE_LINE" >> "$BASHRC"

echo "Researcher user ready. Git configured. Token validated."
