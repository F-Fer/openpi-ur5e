#!/bin/bash

set -euo pipefail

service ssh start

# Attempt to log into Weights & Biases if the API key is available.
if [ -n "${WANDB_API_KEY:-}" ]; then
    if ! uv run wandb login --relogin "$WANDB_API_KEY" >/dev/null 2>&1; then
        echo "[startup] Warning: wandb login failed." >&2
    fi
else
    echo "[startup] WANDB_API_KEY not set; skipping wandb login." >&2
fi



# Attempt to log into Hugging Face using either HUGGINGFACE_HUB_TOKEN or HUGGINGFACE_TOKEN.
HF_TOKEN="${HUGGINGFACE_HUB_TOKEN:-${HUGGINGFACE_TOKEN:-}}"
if [ -n "$HF_TOKEN" ]; then
    uv run huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential
fi

# Generate SSL certificate
echo "[startup] Generating SSL certificates in /root/..."
openssl req -new -newkey rsa:4096 -days 365 -nodes -x509 \
    -subj "/C=DE/ST=Bavaria/L=Nuremberg/O=Siemens/CN=localhost" \
    -keyout /root/key.pem \
    -out /root/cert.pem

# Berechtigungen sicher setzen (optional, aber gut für debugging)
chmod 600 /root/key.pem
chmod 644 /root/cert.pem


cd /workspace

# Build Jupyter auth flags from JUPYTER_PASSWORD env var.
if [ -n "${JUPYTER_PASSWORD:-}" ]; then
    HASHED_PW=$(uv run python -c "from jupyter_server.auth import passwd; print(passwd('${JUPYTER_PASSWORD}'))")
    JUPYTER_AUTH_ARGS="--ServerApp.token='' --ServerApp.password='${HASHED_PW}'"
    echo "[startup] Jupyter password protection enabled."
else
    JUPYTER_AUTH_ARGS="--ServerApp.token='' --ServerApp.password=''"
    echo "[startup] WARNING: No JUPYTER_PASSWORD set — Jupyter is running without authentication!" >&2
fi

uv run jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root \
    --ServerApp.allow_origin='*' \
    --ServerApp.allow_remote_access=True \
    --ServerApp.disable_check_xsrf=True \
    --ServerApp.root_dir=/workspace \
    $JUPYTER_AUTH_ARGS &

sleep infinity
