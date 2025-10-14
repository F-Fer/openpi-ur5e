#!/bin/bash

set -euo pipefail

service ssh start

# Attempt to log into Weights & Biases if the API key is available.
if [ -n "${WANDB_API_KEY:-}" ]; then
    if ! wandb login --relogin "$WANDB_API_KEY" >/dev/null 2>&1; then
        echo "[startup] Warning: wandb login failed." >&2
    fi
else
    echo "[startup] WANDB_API_KEY not set; skipping wandb login." >&2
fi

# Attempt to log into Hugging Face using either HUGGINGFACE_HUB_TOKEN or HUGGINGFACE_TOKEN.
HF_TOKEN="${HUGGINGFACE_HUB_TOKEN:-${HUGGINGFACE_TOKEN:-}}"
if [ -n "$HF_TOKEN" ]; then
    if ! huggingface-cli whoami >/dev/null 2>&1; then
        if ! huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential >/dev/null 2>&1; then
            echo "[startup] Warning: huggingface-cli login failed." >&2
        fi
    fi
else
    echo "[startup] HUGGINGFACE_HUB_TOKEN not set; skipping huggingface-cli login." >&2
fi

cd /workspace

uv run jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root \
    --ServerApp.allow_origin='*' \
    --ServerApp.allow_remote_access=True \
    --ServerApp.disable_check_xsrf=True \
    --ServerApp.root_dir=/workspace \
    --NotebookApp.token="" \
    --NotebookApp.password="" &

sleep infinity
