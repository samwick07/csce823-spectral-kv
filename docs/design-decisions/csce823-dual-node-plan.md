# Dual-Node Execution Plan: Spectral KV-Cache Compression Experiment

**Project:** Spectral KV-Cache Compression via Complex FFT with Learnable Frequency Selection  
**Course:** CSCE 823 Final Project  
**Author:** Samuel Chadwick  
**Date:** 21 August 2026  
**Deadline:** 31 August 2026 (results, paper, presentation)  

---

## 1. Situation

The experiment requires ~1,210 GPU-hours across 14 configurations (13 compressed + 1 baseline), each evaluated over 30 seeds (420 total eval runs). The original plan allocated 4x H200 for 14 days. The deadline of 31 August leaves ~10 days, which is insufficient for a single 4-GPU node.

A second workspace with 4x H100 (80GB) is available. InfiniBand (400G) exists physically but is not configured by admins, so the two 4-GPU nodes will operate as **independent DeepSpeed groups** with no cross-node gradient sync.

This plan splits the experiment across both nodes, consolidates results, and runs the full 7-step statistical analysis on the merged dataset.

---

## 2. Hardware

| Node  | GPUs                  | Memory  | Bandwidth | Role           |
|-------|-----------------------|---------|-----------|----------------|
| H200  | 4x NVIDIA H200 (141GB)| 4.8 TB/s| 400G IB (unconfigured) | Primary, faster |
| H100  | 4x NVIDIA H100 (80GB) | 3.35 TB/s| 400G IB (unconfigured) | Secondary, ~15% slower training, ~30% slower eval |

**No InfiniBand required.** Each node runs an independent 4-GPU DeepSpeed ZeRO-2 group. Results are consolidated after both nodes complete.

---

## 3. Configuration Split

### 3.1 Experiment Matrix

| Config | Transform | Filter    | Gamma | Compression  | 2x2 Cell   |
|--------|-----------|-----------|-------|--------------|------------|
| C00    | N/A       | N/A       | 1.0   | 1x (baseline)| Baseline   |
| C01    | DCT       | Fixed LP  | 0.50  | 2x           | A: DCT+Fix |
| C02    | DCT       | Fixed LP  | 0.22  | 4.5x         | A: DCT+Fix |
| C03    | DCT       | Fixed LP  | 0.01  | 100x         | A: DCT+Fix |
| C04    | DCT       | Learnable | 0.50  | 2x           | B: DCT+Lrn |
| C05    | DCT       | Learnable | 0.22  | 4.5x         | B: DCT+Lrn |
| C06    | DCT       | Learnable | 0.01  | 100x         | B: DCT+Lrn |
| C07    | FFT       | Fixed LP  | 0.50  | 2x           | C: FFT+Fix |
| C08    | FFT       | Fixed LP  | 0.22  | 4.5x         | C: FFT+Fix |
| C09    | FFT       | Fixed LP  | 0.01  | 100x         | C: FFT+Fix |
| C10    | FFT       | Learnable | 0.50  | 2x           | D: FFT+Lrn |
| C11    | FFT       | Learnable | 0.22  | 4.5x         | D: FFT+Lrn |
| C12    | FFT       | Learnable | 0.01  | 100x         | D: FFT+Lrn |

### 3.2 Assignment

C00 (baseline) requires no training — only evaluation. It is assigned to H200 for eval (faster inference).

12 training configs split 6/6, balanced so every 2x2 cell has representation on both nodes:

| Node | Configs (Training)          | A | B | C | D | DCT | FFT | Fixed | Learn |
|------|-----------------------------|---|---|---|---|-----|-----|-------|-------|
| H200 | C01, C02, C04, C07, C08, C12| 2 | 1 | 2 | 1 | 3   | 3   | 4     | 2     |
| H100 | C03, C05, C06, C09, C10, C11| 1 | 2 | 1 | 2 | 3   | 3   | 2     | 4     |

**Properties:**
- Every 2x2 cell has configs on BOTH nodes (no cell is node-exclusive)
- Transform types balanced (3 DCT, 3 FFT per node)
- Consolidated dataset: all 14 configs, all 420 eval runs
- The filter imbalance (4/2 vs 2/4) is NOT a statistical concern — GPU type is not a factor in the ART ANOVA design, and all configs land in the same aggregated dataset regardless of which node trained them

### 3.3 Statistical Validity

The ART ANOVA (Wobbrock et al., 2011) tests:
- Main effect of transform (DCT vs FFT)
- Main effect of filter (Fixed vs Learnable)
- Interaction effect (does learnable filtering benefit depend on transform?)

These tests operate on the **consolidated** per-seed results. The node that trained a config is irrelevant to the statistical model. The only risk is systematic GPU-architecture effects on LoRA adapter quality, but:
1. H100 and H200 use the same CUDA cores and cuDNN kernels (both Hopper architecture)
2. Floating-point reduction order differences are at the 1e-5 level, far below the 30-seed noise floor
3. The split ensures any node effect is orthogonal to all treatment factors

---

## 4. Timeline

### 4.1 Phase Breakdown

| Phase | H200 Node | H100 Node | Wall Clock |
|-------|-----------|-----------|------------|
| 0. Setup + Pilot | 4h | 4h | 4h (parallel) |
| 1. Training | 60h (6 configs x 10h) | 69h (6 configs x 11.5h) | 69h |
| 2. Eval (pipelined) | ~12h after training | ~16h after training | overlaps |
| 3. Consolidation | — | — | 1h |
| 4. Statistical Analysis | — | — | < 5 min |
| 5. Exfil + Paper Prep | — | — | 2h |

### 4.2 Detailed Timeline (with pipelining)

With config-level pipelining, evaluation of completed configs begins while remaining configs are still training.

```
H200 TIMELINE:
  t=0h:    Start training C01
  t=10h:   C01 done → start eval C01 (4-way parallel) + train C02
  t=20h:   C02 done, C01 eval done → eval C02 + train C04
  t=30h:   C04 done, C02 eval done → eval C04 + train C07
  t=40h:   C07 done, C04 eval done → eval C07 + train C08
  t=50h:   C08 done, C07 eval done → eval C08 + train C12
  t=60h:   C12 done, C08 eval done → eval C12
  t=60h:   Also start C00 eval (30 seeds, 4-way parallel)
  t=72h:   C12 eval done. C00 eval done (fits in slack).
  H200 COMPLETE: ~72h (3.0 days)

H100 TIMELINE:
  t=0h:    Start training C03
  t=11.5h: C03 done → start eval C03 + train C05
  t=23h:   C05 done, C03 eval done → eval C05 + train C06
  t=34.5h: C06 done, C05 eval done → eval C06 + train C09
  t=46h:   C09 done, C06 eval done → eval C09 + train C10
  t=57.5h: C10 done, C09 eval done → eval C10 + train C11
  t=69h:   C11 done, C10 eval done → eval C11
  t=85h:   C11 eval done.
  H100 COMPLETE: ~85h (3.5 days)

CONSOLIDATION + ANALYSIS:
  t=85h:   Rsync results from H100 → H200 (or vice versa)
  t=86h:   Run aggregation (src/stats/aggregate.py)
  t=86h:   Run 7-step statistical analysis (src/stats/analyze.py)
  t=86h:   Run exfil (scripts/exfil.sh)
  t=87h:   Results on HuggingFace Hub, ready for paper

GRAND TOTAL: ~87h (3.6 days)
```

### 4.3 Calendar Mapping (Aug 21 → Aug 31)

| Date       | Day | Activity                                          |
|------------|-----|---------------------------------------------------|
| Aug 21 (Fri)| 1  | Workspace provisioning, env setup, pilot run      |
| Aug 22 (Sat)| 2  | Training begins (both nodes), pilot results review|
| Aug 23 (Sun)| 3  | Training continues, first evals begin (pipelined) |
| Aug 24 (Mon)| 4  | Training continues, pipelined evals               |
| Aug 25 (Tue)| 5  | H200 training complete, H200 evals in progress    |
| Aug 26 (Wed)| 6  | H100 training complete, H100 evals in progress    |
| Aug 27 (Thu)| 7  | All evals complete, consolidation + analysis      |
| Aug 28 (Fri)| 8  | Paper writing, figure generation                  |
| Aug 29 (Sat)| 9  | Paper revision, presentation prep                 |
| Aug 30 (Sun)| 10 | Final review, buffer                              |
| Aug 31 (Mon)| 11 | SUBMIT                                             |

**Margin: ~3 days of buffer** for crashes, VPN drops, or slow configs.

---

## 5. Execution Plan

### 5.1 Phase 0: Setup (Aug 21, ~4h per node)

Both nodes run these steps in parallel:

```bash
# On EACH node (H200 and H100):

# 1. Clone repo
git clone https://github.com/samwick07/csce823-spectral-kv.git
cd csce823-spectral-kv

# 2. Set environment
export HF_TOKEN=hf_your_token
export WANDB_API_KEY=your_wandb_key

# 3. Setup environment (venv, deps, model download, datasets)
bash scripts/setup_env.sh

# 4. Run smoke test (verify spectral transforms, model loading, 10-step train)
bash scripts/smoke_test.sh
```

### 5.2 Phase 0.5: Pilot Run (Aug 21, ~4h)

Run the pilot on the H200 node only (3 configs x 5 seeds):

```bash
# H200 node only
python -m src.orchestrator --pilot
```

This validates:
- DeepSpeed ZeRO-2 launches correctly on 4 GPUs
- Spectral transforms (DCT/FFT) work in the training loop
- LoRA adapter saves and loads correctly
- Eval pipeline produces valid all_results.json
- W&B logging works
- Checkpoint resume works

**Gate:** Do not proceed to full run until pilot produces valid results for C00, C07, and C10.

### 5.3 Phase 1+2: Training + Pipelined Eval (Aug 22, both nodes)

#### H200 Node:

```bash
# H200: Train + eval 6 assigned configs + C00 baseline eval
# The orchestrator already supports --config for single configs.
# Run them sequentially; the orchestrator auto-skips completed work.

for config in C01 C02 C04 C07 C08 C12; do
    python -m src.orchestrator --config $config --phase full
done

# After all training is done, eval C00 baseline
python -m src.orchestrator --config C00 --phase eval
```

#### H100 Node:

```bash
# H100: Train + eval 6 assigned configs
for config in C03 C05 C06 C09 C10 C11; do
    python -m src.orchestrator --config $config --phase full
done
```

#### Important: Use tmux on both nodes

```bash
# On each node:
tmux new-session -s experiment
# ... run the commands above ...
# Ctrl+B, D to detach
# tmux attach -t experiment to reattach
```

The orchestrator's existing crash recovery handles:
- DeepSpeed checkpoint resume (every 500 steps)
- Atomic orchestrator_state.json (survives power loss)
- W&B deterministic run IDs (resume same run)
- Marker files (.training_complete, valid all_results.json)

### 5.4 Required Code Changes

Three changes are needed before the run. All are small and backward-compatible.

#### Change 1: True Parallel Eval (CRITICAL)

The current orchestrator runs eval seeds sequentially despite assigning GPU IDs. This must be fixed to achieve 4-way parallelism.

**File:** `src/orchestrator.py`, function `eval_config_seed` and the eval loop in `run_full_sweep`

**Current behavior:**
```python
# Runs one seed at a time, despite gpu_id assignment
for gpu_id, seed in enumerate(batch):
    success = eval_config_seed(config_id, seed, hf_token, gpu_id)
```

**Required behavior:**
```python
# Launch all 4 seeds as parallel subprocesses, wait for all
import concurrent.futures

def eval_batch_parallel(config_id, seeds, hf_token, num_gpus):
    """Run up to num_gpus eval seeds in true parallel."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_gpus) as executor:
        futures = {}
        for gpu_id, seed in enumerate(seeds):
            futures[executor.submit(eval_config_seed, config_id, seed, hf_token, gpu_id)] = seed
        results = {}
        for future in concurrent.futures.as_completed(futures):
            seed = futures[future]
            results[seed] = future.result()
        return results
```

**Impact:** Reduces eval time by ~4x (from sequential to 4-way parallel). Without this fix, the timeline is 6.6 days instead of 3.5 days.

#### Change 2: Config-List Argument (convenience)

Add `--configs` argument to accept a comma-separated list, so each node can be launched with a single command instead of a shell loop.

**File:** `src/orchestrator.py`, CLI section

```python
parser.add_argument(
    "--configs", type=str, default=None,
    help="Comma-separated config IDs (e.g., C01,C02,C04). Overrides --pilot.",
)
```

Then in main():
```python
if args.configs:
    config_ids = args.configs.split(",")
```

#### Change 3: Consolidation Script (new file)

A new script that merges results from both nodes into a single results directory before running aggregation.

**File:** `scripts/consolidate.sh` (new)

```bash
#!/bin/bash
# Consolidate results from two nodes into a single results directory.
# Usage: bash scripts/consolidate.sh <remote_results_dir> <local_results_dir>
# Example: bash scripts/consolidate.sh user@h100-node:/path/to/csce823/results ./results

set -euo pipefail

REMOTE="${1:?Usage: consolidate.sh <remote_results> <local_results>}"
LOCAL="${2:?Usage: consolidate.sh <remote_results> <local_results>}"

echo "=== Consolidating results ==="
echo "  Remote: $REMOTE"
echo "  Local:  $LOCAL"

# Sync remote results into local (merge, don't overwrite existing)
rsync -av --ignore-existing "$REMOTE/" "$LOCAL/"

# Verify all expected configs are present
echo "=== Verifying config coverage ==="
for config in C00 C01 C02 C03 C04 C05 C06 C07 C08 C09 C10 C11 C12; do
    count=$(find "$LOCAL/raw/$config" -name "all_results.json" 2>/dev/null | wc -l)
    if [ "$count" -eq 30 ]; then
        echo "  $config: $count/30 seeds ✓"
    elif [ "$count" -gt 0 ]; then
        echo "  $config: $count/30 seeds ⚠ INCOMPLETE"
    else
        echo "  $config: 0/30 seeds ✗ MISSING"
    fi
done

echo ""
echo "=== Consolidation complete ==="
echo "Run aggregation: python -m src.stats.aggregate"
echo "Run analysis:    python -m src.stats.analyze"
```

### 5.5 Phase 3: Consolidation + Analysis (Aug 27, ~1h)

Once both nodes are complete:

```bash
# On the H200 node (primary):

# 1. Pull results from H100 node
bash scripts/consolidate.sh \
    samuel@h100-node:/path/to/csce823-spectral-kv/results \
    ./results

# 2. Verify all 14 configs x 30 seeds are present
# (consolidate.sh does this automatically)

# 3. Aggregate all results into CSVs
python -m src.stats.aggregate

# 4. Run 7-step statistical analysis
python -m src.stats.analyze

# 5. Package and upload to HuggingFace Hub
bash scripts/exfil.sh
```

### 5.6 Statistical Analysis Pipeline

The existing 7-step pipeline runs on the consolidated dataset:

| Step | Test                  | Purpose                                      |
|------|-----------------------|----------------------------------------------|
| 1    | Wilcoxon signed-rank  | Pairwise: each config vs baseline (C00)      |
| 2    | Friedman test         | Multi-group comparison across all configs    |
| 3    | Nemenyi post-hoc      | Identify which configs differ significantly   |
| 4    | Kolmogorov-Smirnov    | Distribution comparison (per config)          |
| 5    | Anderson-Darling      | Normality check (validates non-parametric choice) |
| 6    | ART ANOVA             | 2x2 factorial: transform x filter main effects + interaction |
| 7    | Holm-Bonferroni       | Multiple comparison correction across all tests |

**Correctness checks before analysis:**
1. All 14 configs present in aggregated CSV (13 compressed + 1 baseline)
2. All 30 seeds present per config (420 rows total)
3. No NaN or inf values in metric columns
4. C00 baseline results are present (reference for Wilcoxon)
5. Each config's results come from the correct LoRA checkpoint (verified via manifest)

---

## 6. Risk Mitigation

| Risk                  | Likelihood | Impact | Mitigation                                    |
|-----------------------|------------|--------|-----------------------------------------------|
| Node crash / reboot   | Medium     | High   | Orchestrator auto-resumes from last checkpoint. State file is atomic. Re-run same command. |
| VPN drop              | High       | Low    | WANDB_MODE=offline fallback. W&B syncs later. Training continues. |
| H100 OOM (80GB)       | Low        | High   | 8B + LoRA r=8 + ZeRO-2 fits in 80GB. Verified: ~50GB peak. If OOM, reduce micro-batch from 8 to 4. |
| Config fails to train | Low        | Medium | Orchestrator skips to next config. Failed config can be re-run later. With 6 configs per node, losing 1 still gives 5/6 data points for that cell. |
| Results corruption    | Low        | High   | Each eval writes self-contained all_results.json. Partial writes are detected (JSON parse fails → re-run). State file is atomic. |
| Deadline pressure     | Medium     | High   | 3-day buffer (Aug 28-30) for paper writing. If H100 is slow, can reduce seeds from 30 to 20 (loses statistical power but still valid for Wilcoxon n≥20). |
| InfiniBand gets configured mid-run | Low | None | No change needed. Nodes run independently. IB would only matter for a unified 8-GPU ZeRO-2 group, which we're not using. |

### 6.1 Fallback: Reduced Seeds

If wall clock is behind schedule after Aug 25 (both nodes still training):

| Seeds per config | Total eval runs | H200 eval time | H100 eval time | Wall clock saved | Statistical impact |
|------------------|-----------------|----------------|----------------|------------------|--------------------|
| 30 (full)        | 420             | 72h            | 85h            | —                | Full power         |
| 20               | 280             | 48h            | 57h            | ~28h             | Still valid (n≥20 for Wilcoxon) |
| 10 (minimum)     | 140             | 24h            | 29h            | ~56h             | Reduced power, Friedman still works |

Decision gate: If either node has not completed training by Aug 26 (end of day), reduce remaining eval seeds to 20.

### 6.2 Fallback: Single-Node Recovery

If the H100 node becomes completely unavailable (hardware failure, access revoked):

- H200 continues with its 6 configs + C00
- H200 evals its 6 configs: 72h (3 days)
- The 6 H100 configs are lost
- The 2x2 design is degraded: cells A and C have 2 configs, cells B and D have 1 config
- ART ANOVA still works with unbalanced cell sizes (it's rank-based)
- Paper would note the reduced design as a limitation

---

## 7. Deliverables Checklist

- [ ] Phase 0: Both nodes provisioned, smoke tests pass
- [ ] Phase 0.5: Pilot run (C00, C07, C10 x 5 seeds) produces valid results
- [ ] Change 1: True parallel eval implemented and tested
- [ ] Change 2: --configs argument added
- [ ] Change 3: consolidate.sh script written
- [ ] Phase 1: All 12 configs trained (6 per node)
- [ ] Phase 2: All 420 evals complete (14 configs x 30 seeds)
- [ ] Phase 3: Results consolidated, all 14 configs verified present
- [ ] Phase 4: 7-step statistical analysis runs without errors
- [ ] Phase 5: Results exfiltrated to HuggingFace Hub
- [ ] Paper: Figures generated from aggregated CSVs
- [ ] Paper: Statistical results table from analysis output
- [ ] Paper: Filter mask visualizations from learned configs
- [ ] Presentation: Slides from paper figures
- [ ] Submit by Aug 31

---

## 8. Quick Reference Commands

### H200 Node (primary)
```bash
# Full run (training + eval for assigned configs)
tmux new-session -s h200
cd csce823-spectral-kv
export HF_TOKEN=hf_xxx
export WANDB_API_KEY=xxx

python -m src.orchestrator --configs C01,C02,C04,C07,C08,C12
python -m src.orchestrator --config C00 --phase eval
```

### H100 Node (secondary)
```bash
tmux new-session -s h100
cd csce823-spectral-kv
export HF_TOKEN=hf_xxx
export WANDB_API_KEY=xxx

python -m src.orchestrator --configs C03,C05,C06,C09,C10,C11
```

### Consolidation (on H200, after both complete)
```bash
bash scripts/consolidate.sh samuel@h100-node:~/csce823-spectral-kv/results ./results
python -m src.stats.aggregate
python -m src.stats.analyze
bash scripts/exfil.sh
```

### Status Check (on either node)
```bash
python -m src.orchestrator --status
```

---

## 9. DeepSpeed Configuration Notes

Both nodes use the existing `configs/deepspeed_zero2_4gpu.json`. No changes needed:

- `train_micro_batch_size_per_gpu`: 8 (fits in both 80GB H100 and 141GB H200)
- `gradient_accumulation_steps`: 1 (train_batch_size = 8 x 4 = 32)
- ZeRO Stage 2: optimizer state + gradient partitioning
- No ZeRO Stage 3 (not needed for 8B + LoRA, and avoids unnecessary communication)

The H100 node's 80GB is sufficient because:
- Llama-3.1-8B BF16 weights: ~16GB
- ZeRO-2 partitioned optimizer (LoRA only, r=8): ~2GB per GPU
- Activations (micro-batch 8, seq_len 4096): ~25GB
- Spectral transform overhead: ~4GB (complex tensors)
- Total peak: ~47GB (59% of 80GB)

---

## 10. W&B Project Organization

| W&B Project     | Runs                              | Node  |
|-----------------|-----------------------------------|-------|
| spectral-kv     | C01_phase1_redpajama              | H200  |
|                 | C01_phase2_longalpaca             | H200  |
|                 | C02_phase1_redpajama              | H200  |
|                 | ... (6 configs x 2 phases)       | H200  |
|                 | C03_phase1_redpajama              | H100  |
|                 | ... (6 configs x 2 phases)       | H100  |

All runs land in the same W&B project. Run IDs are deterministic (`{config}_phase{1|2}_{dataset}`), so crashes resume the same run. The `node` tag (H200/H100) should be added to each run for filtering.

Add to `scripts/run.sh` or the orchestrator:
```bash
# Detect GPU type for W&B tagging
GPU_TYPE=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1 | tr ' ' '_')
export WANDB_TAGS="node:${GPU_TYPE}"
```
