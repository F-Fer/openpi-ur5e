#!/bin/bash

set -euo pipefail

ECHO_PREFIX="[OpenPI Setup]"

info() {
  echo -e "${ECHO_PREFIX} $1"
}

ensure_uv() {
  if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found. Please ensure this script runs inside the Docker container." >&2
    exit 1
  fi
}

verify_setup() {
  info "Environment details"
  python --version
  uv --version
  info "Checking GPU availability"
  python - <<'PY'
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device name: {torch.cuda.get_device_name(0)}")
PY
}

compute_norm_stats() {
  ensure_uv
  info "Computing normalization stats (if not using pre-trained stats)"
  uv run scripts/compute_norm_stats.py pi0_ur5e_finetune_lora
}

run_training() {
  ensure_uv
  local framework=$1
  local exp_name=${2:-ur5e_run_$(date +%Y%m%d_%H%M%S)}
  info "Starting ${framework} training (exp_name=${exp_name})"
  if [[ ${framework} == "jax" ]]; then
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
      uv run scripts/train.py pi0_ur5e_finetune_lora --exp_name "${exp_name}"
  else
    uv run scripts/train_pytorch.py pi0_ur5e_finetune_lora --exp_name "${exp_name}"
  fi
}

run_inference() {
  ensure_uv
  local checkpoint_dir=${1:-./checkpoints/pi0_ur5e_finetune_lora/ur5e_run_*/}
  info "Starting inference server for checkpoint ${checkpoint_dir}"
  uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi0_ur5e \
    --policy.dir="${checkpoint_dir}"
}

print_menu() {
  cat <<'EOF'
Choose an option:
  1) Verify environment setup
  2) Compute normalization stats (if needed)
  3) Start JAX training
  4) Start PyTorch training
  5) Start inference server
  6) Exit
EOF
}

main() {
  while true; do
    print_menu
    read -rp "Enter choice [1-6]: " choice
    case ${choice} in
      1) verify_setup ;;
      2) compute_norm_stats ;;
      3)
        read -rp "Experiment name [default auto]: " exp
        run_training jax "${exp}"
        ;;
      4)
        read -rp "Experiment name [default auto]: " exp
        run_training pytorch "${exp}"
        ;;
      5)
        read -rp "Checkpoint directory [default latest]: " ckpt
        run_inference "${ckpt:-}"
        ;;
      6)
        info "Exiting."
        exit 0
        ;;
      *)
        echo "Invalid choice, try again."
        ;;
    esac
  done
}

main "$@"
