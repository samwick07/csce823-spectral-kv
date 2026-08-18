# CSCE 823: Spectral KV-Cache Compression via Complex FFT with Learnable Frequency Selection

An ablation study of phase preservation and adaptive filtering for KV-cache compression
in transformer-based large language models.

## Overview

This project investigates two modifications to spectral KV-cache compression:

1. **Complex FFT** replacing the DCT, preserving phase information
2. **Learnable spectral filter** replacing fixed low-pass filtering, adaptively selecting
   frequency bands per layer and per head

The study uses a 2×2 factorial design on Llama-3-8B-Instruct with LoRA rank 8,
evaluated across three compression ratios on PG-19, Proof-pile, and LongBench V1.

## Project Structure

```
csce823-spectral-kv/
├── src/
│   ├── spectral/          # DCT/FFT transforms, filters, compressed attention
│   ├── training/          # RedPajama + LongAlpaca fine-tuning pipelines
│   ├── eval/              # PG-19, Proof-pile, LongBench evaluation
│   ├── stats/             # Statistical analysis (Wilcoxon, Friedman, ART ANOVA)
│   └── utils/             # Config, checkpointing, results storage
├── configs/               # DeepSpeed, LoRA, and experiment configs
├── scripts/               # Shell scripts for training and evaluation runs
├── notebooks/             Analysis and visualization notebooks
├── results/               # Checkpoints, raw results, aggregated data, figures
└── docs/                  # Proposal and documentation
```

## Setup

```bash
# Create virtual environment
python3 -m venv ~/csce823-venv
source ~/csce823-venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Login to HuggingFace (requires Llama-3 access approval)
huggingface-cli login
```

## Usage

```bash
# Train a single configuration
python -m src.run_experiment --config configs/experiment_C01.yaml --mode train

# Evaluate a trained configuration
python -m src.run_experiment --config configs/experiment_C01.yaml --mode eval --seed 0

# Run 30-seed evaluation sweep
bash scripts/run_seed_sweep.sh C01

# Aggregate results and run statistical analysis
python -m src.stats.aggregate
python -m src.stats.analyze
```

## Experiment Matrix

| Config | Transform | Filter      | gamma | Compression |
|--------|-----------|-------------|-------|-------------|
| C00    | N/A       | N/A         | 1.0   | 1× (baseline) |
| C01-03 | DCT       | Fixed LP    | 0.50/0.22/0.01 | 2×/4.5×/100× |
| C04-06 | DCT       | Learnable   | 0.50/0.22/0.01 | 2×/4.5×/100× |
| C07-09 | Complex FFT | Fixed LP  | 0.50/0.22/0.01 | 2×/4.5×/100× |
| C10-12 | Complex FFT | Learnable | 0.50/0.22/0.01 | 2×/4.5×/100× |

## Hardware

- 8× NVIDIA H200 (141 GB HBM3e, 4.8 TB/s bandwidth)
- AFIT Center for Cyberspace Research (CCR) AI Cluster
- Coder workspace at coder.afitcdn.org

## Author

Samuel Chadwick — Air Force Institute of Technology

## License

MIT
