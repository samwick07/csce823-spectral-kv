#!/bin/bash
# Run evaluation for a single configuration and seed
# Usage: bash scripts/eval_single.sh <config_id> <seed> [hf_token]
# Example: bash scripts/eval_single.sh C01 0

set -euo pipefail

CONFIG_ID="${1:?Usage: eval_single.sh <config_id> <seed> [hf_token]}"
SEED="${2:?Usage: eval_single.sh <config_id> <seed> [hf_token]}"
HF_TOKEN="${3:-}"

CONFIG_FILE="configs/experiment_${CONFIG_ID}.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

echo "=== Evaluating ${CONFIG_ID} seed=${SEED} ==="

EXTRA_ARGS=""
if [ -n "$HF_TOKEN" ]; then
    EXTRA_ARGS="--hf-token ${HF_TOKEN}"
fi

CUDA_VISIBLE_DEVICES=0 python \
    src/run_experiment.py \
    --config "$CONFIG_FILE" \
    --mode eval \
    --seed "$SEED" \
    $EXTRA_ARGS

echo "=== Evaluation ${CONFIG_ID} seed=${SEED} complete ==="
