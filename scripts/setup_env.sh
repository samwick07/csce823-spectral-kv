#!/usr/bin/env bash
# =============================================================================
# setup_env.sh — One-time environment setup for the Coder workspace.
#
# Creates a Python venv, installs all dependencies, downloads the model
# and datasets, and runs a quick smoke test.
#
# Usage:
#   bash scripts/setup_env.sh
#
# Environment variables:
#   HF_TOKEN — HuggingFace token (required for Llama-3.1-8B-Instruct)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

VENV_DIR="$PROJECT_ROOT/.venv"

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

# --- Check prerequisites ---
header "PREREQUISITE CHECKS"

# Python
if ! command -v python3 &>/dev/null; then
    echo -e "${RED}python3 not found. Install Python 3.11+ first.${NC}"
    exit 1
fi
PY_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo -e "${GREEN}Python: $PY_VERSION${NC}"

# GPU
if ! command -v nvidia-smi &>/dev/null; then
    echo -e "${RED}nvidia-smi not found. GPU drivers required.${NC}"
    exit 1
fi
GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l)
echo -e "${GREEN}GPUs: $GPU_COUNT${NC}"

# HF token
if [[ -z "${HF_TOKEN:-}" ]]; then
    echo -e "${YELLOW}WARNING: HF_TOKEN not set.${NC}"
    echo -e "${YELLOW}Llama-3.1-8B-Instruct is gated. Set: export HF_TOKEN=hf_your_token${NC}"
    echo -e "${YELLOW}Continuing anyway (download will fail if token is missing)...${NC}"
else
    echo -e "${GREEN}HF_TOKEN: set${NC}"
fi

# Disk space check (need ~200 GB for model + datasets + checkpoints)
AVAILABLE_GB=$(df -BG --output=avail "$PROJECT_ROOT" | tail -1 | tr -dc '0-9')
if [[ "$AVAILABLE_GB" -lt 200 ]]; then
    echo -e "${RED}INSUFFICIENT DISK SPACE: ${AVAILABLE_GB}GB available${NC}"
    echo -e "${RED}Need at least 200GB for model (16GB) + datasets (50GB) + checkpoints (50GB) + cache${NC}"
    exit 1
fi
echo -e "${GREEN}Disk space: ${AVAILABLE_GB}GB available${NC}"

# --- Create venv ---
header "CREATING VIRTUAL ENVIRONMENT"

if [[ -d "$VENV_DIR" ]]; then
    echo -e "${YELLOW}venv already exists at $VENV_DIR${NC}"
    # Non-interactive: recreate if --force is passed, otherwise keep
    if [[ "${1:-}" == "--force" ]]; then
        echo -e "${YELLOW}--force detected, recreating venv...${NC}"
        rm -rf "$VENV_DIR"
    fi
fi

if [[ ! -d "$VENV_DIR" ]]; then
    python3 -m venv "$VENV_DIR"
    echo -e "${GREEN}Created venv at $VENV_DIR${NC}"
fi

# Activate
source "$VENV_DIR/bin/activate"

# Upgrade pip
pip install --upgrade pip setuptools wheel

# --- Install dependencies ---
header "INSTALLING DEPENDENCIES"

pip install -r requirements.txt
echo -e "${GREEN}Dependencies installed.${NC}"

# --- Verify torch + GPU ---
header "VERIFYING PYTORCH + GPU"

python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)} '
              f'({torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB)')
"

# --- Download model ---
header "DOWNLOADING MODEL (Llama-3.1-8B-Instruct)"

python -c "
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = 'meta-llama/Llama-3.1-8B-Instruct'
token = os.environ.get('HF_TOKEN')

print(f'Downloading tokenizer: {model_name}')
tok = AutoTokenizer.from_pretrained(model_name, token=token)
print(f'Tokenizer vocab size: {tok.vocab_size}')

print(f'Downloading model: {model_name}')
model = AutoModelForCausalLM.from_pretrained(
    model_name, token=token, torch_dtype='auto', low_cpu_mem_usage=True
)
print(f'Model params: {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B')
print('Model cached at ~/.cache/huggingface/')
"

# --- Download datasets ---
header "DOWNLOADING DATASETS"

python -c "
from datasets import load_dataset

datasets = [
    ('togethercomputer/RedPajama-Data-1T-Sample', 'train', {}),
    ('Yukang/LongAlpaca-16k', 'train', {}),
    ('deepmind/pg19', 'test', {}),
    ('EleutherAI/proof-pile', 'test', {'trust_remote_code': True}),
]

for name, split, kwargs in datasets:
    print(f'Downloading: {name} [{split}]')
    ds = load_dataset(name, split=split, **kwargs)
    print(f'  Rows: {len(ds)}')
"

# LongBench (separate due to different loading)
echo "Downloading: LongBench V1..."
python -c "
from datasets import load_dataset
tasks = ['narrativeqa', 'qasper', 'multifieldqa_en', 'hotpotqa', '2wikimqa', 'musique', 'gov_report', 'qmsum', 'multi_news', 'trec', 'triviaqa', 'samsum', 'passage_count', 'passage_retrieval_en']
for task in tasks:
    try:
        ds = load_dataset('THUIAR/LongBench', task, split='test', trust_remote_code=True)
        print(f'  {task}: {len(ds)} rows')
    except Exception as e:
        print(f'  {task}: FAILED - {e}')
"

echo -e "${GREEN}Datasets cached at ~/.cache/huggingface/${NC}"

# --- Smoke test ---
header "RUNNING SMOKE TEST"

echo "Running unit tests (no GPU required)..."
python -m pytest tests/test_spectral_transforms.py -v --tb=short
echo -e "${GREEN}Unit tests passed.${NC}"

echo ""
echo "Running integration smoke test (requires GPU)..."
if [[ $GPU_COUNT -gt 0 ]]; then
    bash scripts/smoke_test.sh --skip-eval || {
        echo -e "${YELLOW}Smoke test failed. Check logs above.${NC}"
        echo -e "${YELLOW}The experiment can still be started; issues will surface${NC}"
        echo -e "${YELLOW}in the first training run.${NC}"
    }
else
    echo -e "${YELLOW}No GPU detected, skipping integration smoke test.${NC}"
fi

# --- Done ---
header "SETUP COMPLETE"

echo -e "${GREEN}Environment is ready.${NC}"
echo ""
echo "Next steps:"
echo "  1. bash scripts/run.sh --pilot          # run pilot (3 configs x 5 seeds)"
echo "  2. bash scripts/run.sh --status         # check progress"
echo "  3. bash scripts/run.sh --attach         # attach to running session"
echo "  4. bash scripts/monitor.sh              # live monitoring"
echo ""
echo "For the full experiment:"
echo "  bash scripts/run.sh                     # 14 configs x 30 seeds"
echo ""
echo "The venv is at: $VENV_DIR"
echo "Logs are in:   $PROJECT_ROOT/logs/"
echo "Results in:    $PROJECT_ROOT/results/"
