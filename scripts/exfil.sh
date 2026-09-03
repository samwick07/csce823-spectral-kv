#!/usr/bin/env bash
# =============================================================================
# exfil.sh — Package experiment results and upload to HuggingFace Hub.
#
# This script:
#   1. Generates the experiment manifest (provenance)
#   2. Exports filter mask arrays from learnable configs
#   3. Packages all results, LoRA adapters, manifest, and stats
#   4. Uploads to HuggingFace Hub as a dataset repo
#
# The upload is resumable — if interrupted, re-running continues
# from where it left off (HF Hub uses content-addressed storage).
#
# Usage:
#   bash scripts/exfil.sh                    # full exfil + upload
#   bash scripts/exfil.sh --package-only     # package but don't upload
#   bash scripts/exfil.sh --upload-only      # upload existing package
#
# Environment variables:
#   HF_TOKEN       — HuggingFace write-scoped token (REQUIRED for upload)
#   HF_REPO_ID     — Dataset repo ID (default: samwick07/spectral-kv-results)
#
# Called automatically by the orchestrator after analysis completes.
# Can also be run manually at any time.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

VENV_DIR="$PROJECT_ROOT/.venv"
PYTHON="${VENV_DIR}/bin/python"
if [[ ! -f "$PYTHON" ]]; then
    PYTHON="$(command -v python3 || command -v python)"
fi

RESULTS_DIR="$PROJECT_ROOT/results"
CHECKPOINT_DIR="$PROJECT_ROOT/checkpoints"
EXFIL_DIR="$PROJECT_ROOT/results/exfil"
MANIFEST="$RESULTS_DIR/manifest.json"

HF_REPO_ID="${HF_REPO_ID:-samwick07/spectral-kv-results}"
PACKAGE_ONLY=false
UPLOAD_ONLY=false

# Parse args
for arg in "$@"; do
    case "$arg" in
        --package-only) PACKAGE_ONLY=true ;;
        --upload-only)  UPLOAD_ONLY=true ;;
    esac
done

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

header() {
    echo -e "\n${BLUE}============================================================${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}============================================================${NC}"
}

# -----------------------------------------------------------------------------
# Step 1: Generate manifest
# -----------------------------------------------------------------------------
if [[ "$UPLOAD_ONLY" == "false" ]]; then
    header "STEP 1: GENERATING EXPERIMENT MANIFEST"
    "$PYTHON" -m src.utils.manifest
    echo -e "${GREEN}Manifest: $MANIFEST${NC}"
fi

# -----------------------------------------------------------------------------
# Step 2: Export filter mask arrays (for paper figures)
# -----------------------------------------------------------------------------
if [[ "$UPLOAD_ONLY" == "false" ]]; then
    header "STEP 2: EXPORTING FILTER MASK ARRAYS"
    MASK_DIR="$RESULTS_DIR/filter_masks"
    mkdir -p "$MASK_DIR"

    "$PYTHON" -c "
import json
import sys
from pathlib import Path

# Export mask arrays from learnable config checkpoints
# These are the raw numpy arrays needed for publication-quality figures
checkpoint_dir = Path('$CHECKPOINT_DIR')
mask_dir = Path('$MASK_DIR')

learnable_configs = ['C04', 'C05', 'C06', 'C10', 'C11', 'C12']

for config_id in learnable_configs:
    ckpt_path = checkpoint_dir / config_id / 'phase2_longalpaca' / 'final'
    if not ckpt_path.exists():
        ckpt_path = checkpoint_dir / config_id / 'phase1_redpajama' / 'final'
    if not ckpt_path.exists():
        print(f'  {config_id}: no checkpoint found, skipping')
        continue

    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM
        from src.utils.constants import DEFAULT_MODEL_NAME
        import os
        import numpy as np

        token = os.environ.get('HF_TOKEN')
        base = AutoModelForCausalLM.from_pretrained(
            DEFAULT_MODEL_NAME, token=token, torch_dtype='auto',
            low_cpu_mem_usage=True, device_map='cpu',
            attn_implementation='eager',
        )
        model = PeftModel.from_pretrained(base, str(ckpt_path))
        model.eval()

        # Extract filter masks from spectral layers
        masks_exported = 0
        config_mask_dir = mask_dir / config_id
        config_mask_dir.mkdir(parents=True, exist_ok=True)

        for name, module in model.named_modules():
            if hasattr(module, 'get_mask') and callable(getattr(module, 'get_mask')):
                mask = module.get_mask()
                if hasattr(mask, 'cpu'):
                    mask = mask.cpu().numpy()
                layer_idx = name.split('.')[-2] if '.' in name else 'unknown'
                np.save(config_mask_dir / f'layer_{layer_idx}_mask.npy', mask)
                masks_exported += 1

                # Also save retained fraction per head
                if hasattr(module, 'get_retained_fraction'):
                    retained = module.get_retained_fraction()
                    if hasattr(retained, 'cpu'):
                        retained = retained.cpu().numpy()
                    np.save(config_mask_dir / f'layer_{layer_idx}_retained.npy', retained)

        print(f'  {config_id}: exported {masks_exported} mask arrays')

        # Clean up model to free memory
        del model
        del base
        import gc; gc.collect()

    except Exception as e:
        print(f'  {config_id}: FAILED - {e}')
        # Continue — masks are nice-to-have, not blocking
" 2>&1 || {
        echo -e "${YELLOW}Filter mask export failed. Continuing — this is not blocking.${NC}"
    }
fi

# -----------------------------------------------------------------------------
# Step 3: Package everything into exfil directory
# -----------------------------------------------------------------------------
if [[ "$UPLOAD_ONLY" == "false" ]]; then
    header "STEP 3: PACKAGING RESULTS"

    mkdir -p "$EXFIL_DIR"

    # Copy results (aggregated CSVs, stats JSON, raw eval results)
    if [[ -d "$RESULTS_DIR/aggregated" ]]; then
        cp -r "$RESULTS_DIR/aggregated" "$EXFIL_DIR/"
        echo -e "${GREEN}  Copied: aggregated results${NC}"
    fi
    if [[ -f "$RESULTS_DIR/statistical_analysis.json" ]]; then
        cp "$RESULTS_DIR/statistical_analysis.json" "$EXFIL_DIR/"
        echo -e "${GREEN}  Copied: statistical analysis${NC}"
    fi
    if [[ -f "$MANIFEST" ]]; then
        cp "$MANIFEST" "$EXFIL_DIR/"
        echo -e "${GREEN}  Copied: experiment manifest${NC}"
    fi
    if [[ -d "$RESULTS_DIR/filter_masks" ]]; then
        cp -r "$RESULTS_DIR/filter_masks" "$EXFIL_DIR/"
        echo -e "${GREEN}  Copied: filter mask arrays${NC}"
    fi
    if [[ -f "$RESULTS_DIR/orchestrator_state.json" ]]; then
        cp "$RESULTS_DIR/orchestrator_state.json" "$EXFIL_DIR/"
        echo -e "${GREEN}  Copied: orchestrator state${NC}"
    fi

    # Copy raw eval results (per-config per-seed JSONs)
    if [[ -d "$RESULTS_DIR/raw" ]]; then
        cp -r "$RESULTS_DIR/raw" "$EXFIL_DIR/"
        echo -e "${GREEN}  Copied: raw eval results${NC}"
    fi

    # Copy LoRA adapter weights (small: ~21 MB per config)
    ADAPTER_DIR="$EXFIL_DIR/lora_adapters"
    mkdir -p "$ADAPTER_DIR"
    if [[ -d "$CHECKPOINT_DIR" ]]; then
        for config_dir in "$CHECKPOINT_DIR"/C*/; do
            config_name=$(basename "$config_dir")
            # Copy the final phase2 checkpoint (or phase1 if phase2 missing)
            for phase in "phase2_longalpaca/final" "phase1_redpajama/final"; do
                src="$config_dir/$phase"
                if [[ -d "$src" ]]; then
                    dest="$ADAPTER_DIR/${config_name}"
                    mkdir -p "$dest"
                    # Copy adapter_config.json and .safetensors (skip optimizer states)
                    cp "$src"/adapter_config.json "$dest/" 2>/dev/null || true
                    cp "$src"/*.safetensors "$dest/" 2>/dev/null || true
                    cp "$src"/*.bin "$dest/" 2>/dev/null || true
                    echo -e "${GREEN}  Copied: LoRA adapter ${config_name} (${phase%%/*})${NC}"
                    break
                fi
            done
        done
    fi

    # Copy configs (experiment YAMLs + DeepSpeed configs)
    mkdir -p "$EXFIL_DIR/configs"
    cp "$PROJECT_ROOT"/configs/experiment_C*.yaml "$EXFIL_DIR/configs/" 2>/dev/null || true
    cp "$PROJECT_ROOT"/configs/deepspeed_zero2_*.json "$EXFIL_DIR/configs/" 2>/dev/null || true
    echo -e "${GREEN}  Copied: experiment configs${NC}"

    # Create README for the dataset repo
    cat > "$EXFIL_DIR/README.md" << 'READMEEOF'
---
license: apache-2.0
language:
- en
tags:
- spectral-compression
- kv-cache
- llama
- fft
- dct
- ablation-study
size_categories:
- n<1K
---

# Spectral KV-Cache Compression — Ablation Study Results

This dataset contains the full experimental artifacts for the paper:

**"Spectral KV-Cache Compression via Complex FFT with Learnable
Frequency Selection: An Ablation Study of Phase Preservation and
Adaptive Filtering"**

## Contents

| Path | Description |
|------|-------------|
| `manifest.json` | Experiment provenance (git hash, GPU model, package versions) |
| `orchestrator_state.json` | Run statistics (start time, crash count, completion) |
| `aggregated/per_seed.csv` | Per-config per-seed per-benchmark raw metrics |
| `aggregated/summary.csv` | Per-config per-benchmark summary statistics |
| `statistical_analysis.json` | 7-step statistical analysis (Wilcoxon, Friedman, ART ANOVA, etc.) |
| `raw/` | Per-seed evaluation JSONs (perplexity, LongBench scores, efficiency) |
| `filter_masks/` | Raw numpy arrays of learned frequency masks (learnable configs only) |
| `lora_adapters/` | Trained LoRA adapter weights (safetensors format) |
| `configs/` | Experiment YAML configs + DeepSpeed ZeRO-2 configs |

## Reproducibility

See `manifest.json` for the exact git commit, Python/torch/CUDA versions,
and GPU model used. The orchestrator state file records crash count and
total runtime.
READMEEOF

    echo -e "${GREEN}  Created: dataset README${NC}"

    # Calculate package size
    PKG_SIZE=$(du -sh "$EXFIL_DIR" | cut -f1)
    echo -e "${GREEN}\nPackage size: $PKG_SIZE${NC}"
    echo -e "Package location: $EXFIL_DIR"
fi

# -----------------------------------------------------------------------------
# Step 4: Upload to HuggingFace Hub
# -----------------------------------------------------------------------------
if [[ "$PACKAGE_ONLY" == "false" ]]; then
    header "STEP 4: UPLOADING TO HUGGINGFACE HUB"

    if [[ -z "${HF_TOKEN:-}" ]]; then
        echo -e "${RED}HF_TOKEN not set. Cannot upload.${NC}"
        echo -e "${YELLOW}Package is ready at: $EXFIL_DIR${NC}"
        echo -e "${YELLOW}Set HF_TOKEN and re-run: bash scripts/exfil.sh --upload-only${NC}"
        exit 1
    fi

    echo "  Repo: $HF_REPO_ID"

    "$PYTHON" -c "
import os
import sys
from pathlib import Path
from huggingface_hub import HfApi, create_repo

repo_id = os.environ.get('HF_REPO_ID', 'samwick07/spectral-kv-results')
token = os.environ.get('HF_TOKEN')
exfil_dir = Path('$EXFIL_DIR')

if not exfil_dir.exists():
    print(f'ERROR: Package directory not found: {exfil_dir}', file=sys.stderr)
    sys.exit(1)

api = HfApi(token=token)

# Create the dataset repo if it doesn't exist
try:
    create_repo(repo_id, repo_type='dataset', token=token, exist_ok=True)
    print(f'  Repo ready: {repo_id}')
except Exception as e:
    print(f'  Repo creation: {e}')
    # Try uploading anyway — it might already exist

# Upload the entire exfil directory
print(f'  Uploading {exfil_dir} ...')
api.upload_folder(
    folder_path=str(exfil_dir),
    repo_id=repo_id,
    repo_type='dataset',
    token=token,
    commit_message='Experiment results exfil',
)

print(f'  Upload complete: https://huggingface.co/datasets/{repo_id}')
" 2>&1

    if [[ $? -eq 0 ]]; then
        echo -e "${GREEN}Upload complete!${NC}"
        echo -e "  https://huggingface.co/datasets/${HF_REPO_ID}"
    else
        echo -e "${RED}Upload failed. Package is at: $EXFIL_DIR${NC}"
        echo -e "${YELLOW}Re-run when HF_TOKEN is set: bash scripts/exfil.sh --upload-only${NC}"
        exit 1
    fi
fi

# -----------------------------------------------------------------------------
# Done
# -----------------------------------------------------------------------------
header "EXFIL COMPLETE"
echo -e "  Package:  $EXFIL_DIR"
echo -e "  HF Hub:   https://huggingface.co/datasets/${HF_REPO_ID}"
echo ""
echo "Next steps (on your workstation):"
echo "  1. Clone the dataset: git clone https://huggingface.co/datasets/${HF_REPO_ID}"
echo "  2. Review statistical_analysis.json"
echo "  3. Generate figures from filter_masks/ and aggregated/per_seed.csv"
echo "  4. Write the paper"
