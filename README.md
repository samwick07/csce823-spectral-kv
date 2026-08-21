# Spectral KV-Cache Compression via Complex FFT with Learnable Frequency Selection

An ablation study of phase preservation and adaptive filtering for KV-cache
compression in transformer-based large language models. This project serves
as preliminary work for a forthcoming journal article and dissertation
research on spectral methods for LLM memory efficiency.

## Overview

Standard KV-cache compression with DCT and fixed low-pass filtering
(FreqKV) discards phase information and cannot adapt to frequency content
that varies across layers, heads, and domains. This project investigates
two modifications:

1. **Complex FFT** replacing the DCT, preserving phase information in the
   spectral representation of cached keys and values
2. **Learnable spectral filter** replacing fixed low-pass filtering,
   adaptively selecting frequency bands per layer and per head via a
   sigmoid soft mask trained with the model

The study uses a 2x2 factorial design on Llama-3.1-8B-Instruct with LoRA
rank 8, evaluated across three compression ratios (2x, 4.5x, 100x) on
PG-19, Proof-pile, and LongBench V1.

## Experiment Matrix

| Config | Transform   | Filter    | gamma | Compression  |
|--------|-------------|-----------|-------|--------------|
| C00    | N/A         | N/A       | 1.0   | 1x (baseline) |
| C01-03 | DCT         | Fixed LP  | 0.50 / 0.22 / 0.01 | 2x / 4.5x / 100x |
| C04-06 | DCT         | Learnable | 0.50 / 0.22 / 0.01 | 2x / 4.5x / 100x |
| C07-09 | Complex FFT | Fixed LP  | 0.50 / 0.22 / 0.01 | 2x / 4.5x / 100x |
| C10-12 | Complex FFT | Learnable | 0.50 / 0.22 / 0.01 | 2x / 4.5x / 100x |

Statistical analysis uses a 7-step pipeline: Wilcoxon signed-rank,
Friedman test, Nemenyi post-hoc, Kolmogorov-Smirnov, Anderson-Darling,
ART ANOVA (Wobbrock et al., 2011), and Holm-Bonferroni correction.

## Project Structure

```
csce823-spectral-kv/
├── src/
│   ├── spectral/          # DCT/FFT transforms, filters, compressed attention
│   ├── training/          # RedPajama + LongAlpaca fine-tuning pipelines
│   ├── eval/              # PG-19, Proof-pile, LongBench, efficiency metrics
│   ├── stats/             # Aggregation + 7-step statistical analysis
│   ├── tests/             # Unit tests + integration smoke tests
│   ├── utils/             # Config, constants, W&B, manifest, checkpointing
│   ├── orchestrator.py    # Resilient 4-phase experiment driver
│   └── run_experiment.py  # Single-config train/eval entry point
├── configs/               # 13 experiment YAMLs + DeepSpeed ZeRO-2 configs
├── scripts/               # run.sh, setup_env.sh, exfil.sh, monitor.sh, smoke_test.sh
├── tests/                 # Spectral transform unit tests (28 tests)
├── results/               # Generated: raw evals, aggregated CSVs, stats, manifest
├── checkpoints/           # Generated: DeepSpeed checkpoints + LoRA adapters
└── docs/                  # Compute resources, dataset verification, reexamination
```

## Hardware

- 4x NVIDIA H200 (141 GB HBM3e, 4.8 TB/s bandwidth)
- 1x AMD EPYC 9004 (96 cores, 192 threads)
- AFIT Center for Cyberspace Research (CCR) AI Cluster
- Coder workspace at coder.afitcdn.org
- Estimated: ~1,278 GPU-hours, ~14 days wall clock

## Quick Start (Cluster Deployment)

### Prerequisites

- HuggingFace token with read + write access (set as `HF_TOKEN` env var)
- Weights & Biases API key (set as `WANDB_API_KEY` env var)
- Llama-3.1 model license approved on HuggingFace
- 4x H200 GPUs visible via `nvidia-smi`

### One-Time Setup

```bash
git clone https://github.com/samwick07/csce823-spectral-kv.git
cd csce823-spectral-kv

# Set environment variables
export HF_TOKEN=hf_your_token
export WANDB_API_KEY=your_wandb_key

# Create venv, install dependencies, download model + datasets, run smoke tests
bash scripts/setup_env.sh
```

### Running the Experiment

```bash
# Pilot run first (3 configs x 5 seeds, ~22 hours)
bash scripts/run.sh --pilot

# Full experiment (14 configs x 30 seeds, ~14 days)
bash scripts/run.sh
```

That's it. `run.sh` launches the orchestrator inside a tmux session that
survives SSH disconnects and VPN drops. The orchestrator runs 4 phases
automatically:

| Phase | Description | Hardware | Duration |
|-------|-------------|----------|----------|
| 1. Train | 14 configs via DeepSpeed ZeRO-2 | 4x H200 | ~6.5 days |
| 2. Eval | 420 runs, 4-way parallel (1 GPU each) | 4x H200 | ~7.5 days |
| 3. Analyze | 7-step statistical pipeline (ART ANOVA) | CPU | seconds |
| 4. Exfil | Package + upload to HuggingFace Hub | CPU | minutes |

### Monitoring

```bash
bash scripts/monitor.sh --watch   # live dashboard (GPU usage, progress, logs)
bash scripts/run.sh --status      # quick one-shot progress summary
bash scripts/run.sh --attach      # attach to tmux session
tail -f logs/run_*.log            # raw log stream
```

### Crash Recovery

If the workspace loses power or the node reboots:

```bash
bash scripts/run.sh    # just re-run the same command
```

The orchestrator will:
- Load `results/orchestrator_state.json`
- Skip all completed training configs and eval seeds
- Resume training from the last DeepSpeed checkpoint (every 500 steps)
- Resume eval from the next incomplete seed
- Increment `crash_count` in the state file

No manual intervention needed. No data loss. No duplicated work.

### Stopping

```bash
bash scripts/run.sh --stop    # graceful (finishes current subprocess)
# In tmux: Ctrl+C once for graceful, twice to force-kill
```

## Resilience Architecture

The system has 4 layers of crash protection:

1. **DeepSpeed checkpoint resume** — Training saves every 500 steps
   (Phase 1) or every epoch (Phase 2). On restart, the trainer
   auto-detects the latest checkpoint and resumes from there.

2. **Marker files + result validation** — Training writes a
   `.training_complete` marker. Each eval writes a self-contained
   `all_results.json`. Partial writes from crashes are detected and
   re-run from scratch.

3. **Atomic state file** — `results/orchestrator_state.json` tracks
   all completed work. Written atomically (tmp + rename) so it cannot
   be corrupted by a power outage mid-write.

4. **W&B deterministic run IDs** — Each phase uses a deterministic
   run ID (`{config_id}_phase1_redpajama`, etc.) with `resume="allow"`.
   Crashes resume the same W&B run and append logs — no fragmentation.

## Exfiltration

After analysis completes, the orchestrator automatically runs
`scripts/exfil.sh`, which:

1. Generates `manifest.json` (git commit, Python/torch/CUDA versions,
   GPU model, experiment config, orchestrator state)
2. Exports filter mask arrays as `.npy` files (for paper figures)
3. Packages all results, LoRA adapters, configs, and manifest
4. Uploads to `huggingface.co/datasets/samwick07/spectral-kv-results`

The upload is resumable — re-run `bash scripts/exfil.sh --upload-only`
if interrupted.

## Post-Experiment (Workstation)

After the experiment completes and results are on HF Hub:

```bash
git clone https://huggingface.co/datasets/samwick07/spectral-kv-results
```

From the downloaded data:
- `statistical_analysis.json` — interpret 7-step stats results
- `aggregated/per_seed.csv` — generate publication figures
- `filter_masks/` — visualize learned frequency responses
- `manifest.json` — write the reproducibility appendix
- `lora_adapters/` — reproduce or extend experiments

## Testing

```bash
# Unit tests (no GPU required, 28 tests)
pytest tests/test_spectral_transforms.py -v

# Integration smoke tests (requires GPU + network)
bash scripts/smoke_test.sh
```

## Key Design Decisions

- **Eager attention**: FlashAttention-2 is incompatible with the spectral
  forward pass (which overrides `LlamaAttention.forward` with manual SDPA).
  Removed `flash-attn` from requirements; using `attn_implementation="eager"`.

- **4-GPU DeepSpeed configs**: `deepspeed_zero2_4gpu.json` and
  `deepspeed_zero2_4gpu_longctx.json` with `train_batch_size=32`
  (8 micro-batch per GPU x 4 GPUs).

- **Centralized model identifiers**: All model name references go through
  `src/utils/constants.py` (`DEFAULT_MODEL_NAME`,
  `SMOKE_TEST_MODEL_NAME`), both pointing to
  `meta-llama/Llama-3.1-8B-Instruct`.

- **ART ANOVA**: The Aligned Rank Transform (Wobbrock et al., 2011) is
  used instead of per-factor Kruskal-Wallis because it correctly detects
  interaction effects — the core research question of whether the benefit
  of learnable filtering depends on transform type.

## Author

Samuel Chadwick — Air Force Institute of Technology

## License

Apache-2.0
