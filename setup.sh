#!/usr/bin/env bash
#
# One-shot setup for fin-media-poc.
#
# Automates what can be done safely:
#   - creates/uses the .venv virtualenv
#   - installs Python dependencies
#   - pulls the Ollama embedding model (if Ollama is running)
#
# System-level tools (ffmpeg, Ollama) can't be installed without your
# involvement, so those are checked and reported with exact install commands
# rather than installed silently.
#
# Usage:  ./setup.sh   (or:  bash setup.sh)

set -u

# Run from the project root regardless of where the script is called from.
cd "$(dirname "$0")"

ok()    { printf "  \033[32m✓\033[0m %s\n" "$1"; }
warn()  { printf "  \033[33m!\033[0m %s\n" "$1"; }
fail()  { printf "  \033[31m✗\033[0m %s\n" "$1"; }
step()  { printf "\n\033[1m%s\033[0m\n" "$1"; }

READY=1

# --- 1. Python virtualenv --------------------------------------------------
step "1. Python virtualenv"
if [ -d ".venv" ]; then
    ok ".venv already exists"
elif command -v python3.11 >/dev/null 2>&1; then
    python3.11 -m venv .venv && ok "created .venv with python3.11"
else
    fail "python3.11 not found — install Python 3.11, then re-run this script"
    READY=0
fi

PY=".venv/bin/python"
PIP=".venv/bin/pip"

# --- 2. Python dependencies ------------------------------------------------
step "2. Python dependencies"
if [ -x "$PIP" ]; then
    if "$PIP" install -q --disable-pip-version-check -r requirements.txt; then
        ok "installed from requirements.txt"
    else
        fail "pip install failed — see output above"
        READY=0
    fi
else
    fail "no virtualenv pip — fix step 1 first"
    READY=0
fi

# --- 3. ffmpeg (system tool) ----------------------------------------------
step "3. ffmpeg"
if command -v ffmpeg >/dev/null 2>&1; then
    ok "ffmpeg found ($(ffmpeg -version | head -1 | cut -d' ' -f1-3))"
else
    fail "ffmpeg not found — install it with:  brew install ffmpeg"
    READY=0
fi

# --- 4. Ollama (system tool) + embedding model ----------------------------
step "4. Ollama + embedding model"
EMBED_MODEL="mxbai-embed-large"
if ! command -v ollama >/dev/null 2>&1; then
    fail "ollama not found — install it from https://ollama.com/download"
    READY=0
elif ! curl -s -o /dev/null localhost:11434/api/tags; then
    warn "ollama installed but not running — start it with:  ollama serve"
    warn "then pull the model with:  ollama pull $EMBED_MODEL"
    READY=0
else
    ok "ollama is running"
    if curl -s localhost:11434/api/tags | grep -q "$EMBED_MODEL"; then
        ok "$EMBED_MODEL already pulled"
    else
        echo "  pulling $EMBED_MODEL ..."
        if ollama pull "$EMBED_MODEL"; then
            ok "$EMBED_MODEL pulled"
        else
            fail "failed to pull $EMBED_MODEL"
            READY=0
        fi
    fi
fi

# --- Summary ---------------------------------------------------------------
step "Summary"
if [ "$READY" -eq 1 ]; then
    ok "All set. Start the app with:  .venv/bin/python app.py"
else
    warn "Some steps need attention (see above). Re-run ./setup.sh after fixing them."
fi
