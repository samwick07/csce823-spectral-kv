#!/bin/bash
# Run 30-seed evaluation sweep for a single configuration
# Usage: bash scripts/run_seed_sweep.sh <config_id> [hf_token] [gpu_id]
# Example: bash scripts/run_seed_sweep.sh C01 "" 0
#
# This script runs all 30 seeds sequentially on a single GPU.
# For parallel execution, run multiple instances on different GPUs.

set -euo pipefail

CONFIG_ID="${1:?Usage: run_seed_sweep.sh <config_id> [hf_token] [gpu_id]}"
HF_TOKEN="${2:-}"
GPU_ID="${3:-0}"

echo "=== Seed sweep: ${CONFIG_ID} on GPU ${GPU_ID} ==="

for SEED in $(seq 0 29); do
    echo "--- Seed ${SEED}/29 ---"
    CUDA_VISIBLE_DEVICES=${GPU_ID} bash scripts/eval_single.sh "$CONFIG_ID" "$SEED" "$HF_TOKEN"
done

echo "=== Seed sweep ${CONFIG_ID} complete ==="
