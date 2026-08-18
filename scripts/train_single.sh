#!/bin/bash
# Train a single experiment configuration
# Usage: bash scripts/train_single.sh <config_id> [hf_token]
# Example: bash scripts/train_single.sh C01

set -euo pipefail

CONFIG_ID="${1:?Usage: train_single.sh <config_id> [hf_token]}"
HF_TOKEN="${2:-}"

CONFIG_FILE="configs/experiment_${CONFIG_ID}.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

echo "=== Training ${CONFIG_ID} ==="
echo "Config: ${CONFIG_FILE}"
echo "GPUs: $(nvidia-smi -L 2>/dev/null | wc -l) visible"

EXTRA_ARGS=""
if [ -n "$HF_TOKEN" ]; then
    EXTRA_ARGS="--hf-token ${HF_TOKEN}"
fi

deepspeed --num_gpus=8 \
    src/run_experiment.py \
    --config "$CONFIG_FILE" \
    --mode train \
    --deepspeed configs/deepspeed_zero2_8gpu.json \
    $EXTRA_ARGS

echo "=== Training ${CONFIG_ID} complete ==="
