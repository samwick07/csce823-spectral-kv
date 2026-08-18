#!/bin/bash
# Run the full experiment matrix: train all 14 configs, eval all seeds
# Usage: bash scripts/run_all.sh [hf_token]
#
# This script trains all 14 configurations sequentially and then
# runs the 30-seed evaluation sweep for each.
# For parallel execution, see docs/parallel_execution.md

set -euo pipefail

HF_TOKEN="${1:-}"

CONFIGS=("C00" "C01" "C02" "C03" "C04" "C05" "C06" "C07" "C08" "C09" "C10" "C11" "C12")

echo "=== Full experiment matrix: ${#CONFIGS[@]} configs ==="
echo "Start: $(date)"

# Phase 1: Train all configs
for CONFIG_ID in "${CONFIGS[@]}"; do
    echo ""
    echo "============================================"
    echo "  Training ${CONFIG_ID}"
    echo "============================================"
    bash scripts/train_single.sh "$CONFIG_ID" "$HF_TOKEN"
done

# Phase 2: Evaluate all configs (30 seeds each)
for CONFIG_ID in "${CONFIGS[@]}"; do
    echo ""
    echo "============================================"
    echo "  Evaluating ${CONFIG_ID} (30 seeds)"
    echo "============================================"
    bash scripts/run_seed_sweep.sh "$CONFIG_ID" "$HF_TOKEN" 0
done

# Phase 3: Aggregate and analyze
echo ""
echo "============================================"
echo "  Aggregating and analyzing results"
echo "============================================"
python -m src.stats.aggregate
python -m src.stats.analyze

echo ""
echo "=== Full experiment matrix complete ==="
echo "End: $(date)"
