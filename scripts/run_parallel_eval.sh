#!/bin/bash
# Parallel evaluation across 8 GPUs
# Usage: bash scripts/run_parallel_eval.sh [hf_token]
#
# Distributes the 14 configs x 30 seeds = 420 eval runs across 8 GPUs.
# Each GPU processes configs sequentially, seeds sequentially within a config.
#
# Estimated time: ~300-510 GPU-hours / 8 GPUs = ~38-64 hours

set -euo pipefail

HF_TOKEN="${1:-}"

# Split configs across GPUs (2 configs per GPU, with C00 baseline first on GPU 0)
GPU_0_CONFIGS=("C00" "C01")
GPU_1_CONFIGS=("C02" "C03")
GPU_2_CONFIGS=("C04" "C05")
GPU_3_CONFIGS=("C06" "C07")
GPU_4_CONFIGS=("C08" "C09")
GPU_5_CONFIGS=("C10" "C11")
GPU_6_CONFIGS=("C12")
GPU_7_CONFIGS=()  # Reserved for efficiency / overflow

echo "=== Parallel evaluation across 8 GPUs ==="
echo "Start: $(date)"

# Launch background jobs for each GPU
for GPU_ID in 0 1 2 3 4 5 6; do
    CONFIGS_VAR="GPU_${GPU_ID}_CONFIGS[@]"
    CONFIGS=("${!CONFIGS_VAR}")

    if [ ${#CONFIGS[@]} -eq 0 ]; then
        continue
    fi

    (
        for CONFIG_ID in "${CONFIGS[@]}"; do
            echo "GPU ${GPU_ID}: Starting ${CONFIG_ID}"
            bash scripts/run_seed_sweep.sh "$CONFIG_ID" "$HF_TOKEN" "$GPU_ID"
            echo "GPU ${GPU_ID}: Finished ${CONFIG_ID}"
        done
    ) &
    echo "Launched GPU ${GPU_ID} (PID $!)"
done

echo "All GPU jobs launched. Waiting for completion..."
wait

echo "=== Parallel evaluation complete ==="
echo "End: $(date)"
echo ""
echo "Run aggregation and analysis:"
echo "  python -m src.stats.aggregate"
echo "  python -m src.stats.analyze"
