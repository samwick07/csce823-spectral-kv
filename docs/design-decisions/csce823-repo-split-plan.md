# Repo Split Plan: N=1 Class Project + N=30 Publication Archive

**Date:** 21 August 2026  
**Author:** Samuel Chadwick  
**Status:** Awaiting approval  

---

## 1. Situation

The experiment serves two roles with different requirements:

| Requirement       | Class Project (N=1)           | Publication (N=30)                  |
|-------------------|-------------------------------|-------------------------------------|
| Deadline          | Aug 31, 2026                  | No deadline (future run)            |
| Seeds per config  | 1                             | 30                                  |
| Eval runs         | 14                            | 420                                 |
| Statistics        | Point estimates + deltas      | 7-step pipeline (Wilcoxon, ART ANOVA, etc.) |
| Training          | Same 13 configs               | Same 13 configs                     |
| GPU allocation    | 4x H200                       | 4x H200 (14-day wall clock)         |

Training is identical in both versions (562 GPU-hours, 13 configs). The difference is entirely in evaluation scope and statistical analysis.

---

## 2. Wall Clock (N=1 on 4x H200, no code changes required)

| Phase    | Time    | Notes                                              |
|----------|---------|----------------------------------------------------|
| Setup    | 4h      | Clone, venv, deps, model download, smoke test      |
| Pilot    | 4h      | 3 configs (C00, C07, C10) x 1 seed — validate pipeline |
| Training | 140.5h  | 13 configs x 10.8h each (562 GPU-h / 4 GPUs)       |
| Eval     | 21.6h   | 14 runs x 1.54h (sequential, current code is fine) |
| Analysis | <1h     | Point estimate table, no 7-step pipeline           |
| Exfil    | <1h     | HF Hub upload (~10 MB for N=1)                     |
| **Total**| **~172h** | **7.2 days**                                     |

With workspace provisioned Aug 25: compute done Sep 1 → MISSES deadline.
With workspace provisioned Aug 22: compute done Aug 29 → 2 days for paper. Tight.

The parallel eval fix is no longer needed — 14 sequential runs at 1.54h each = 22h, negligible against 140h of training.

---

## 3. Training Reduction Option

If the workspace cannot be provisioned before Aug 25, training can be reduced to fit:

| Option                     | GPU-h  | Train  | Total   | Days  | Start Aug 25 → Done | Paper Buffer |
|----------------------------|--------|--------|---------|-------|---------------------|--------------|
| Full training (unchanged)  | 562    | 140.5h | 172h    | 7.2d  | Sep 1               | -1d (MISS)   |
| 50% steps (fewer epochs)   | 281    | 70.2h  | 101h    | 4.2d  | Aug 29              | 2d           |
| 25% steps (demo level)     | 140    | 35.0h  | 66h     | 2.7d  | Aug 27              | 4d           |

This is controlled by the `max_steps` or `num_train_epochs` parameter in each config YAML. No code changes needed — just config edits. The N=30 publication repo retains the full training schedule.

---

## 4. Repo Strategy

### 4.1 New Repo: N=30 Publication Archive

Create a new GitHub repo that preserves the complete N=30 experiment exactly as it exists today. This is the publication-grade archive — no modifications.

```
Repo: samwick07/csce823-spectral-kv-full
Branch: main (frozen snapshot of current codebase)
Contains:
  - All 14 config YAMLs (C00-C12) with N=30 seed range
  - 7-step statistical analysis pipeline (src/stats/)
  - Full training schedule (full steps/epochs)
  - Orchestrator with N=30 defaults
  - All unit tests (28 spectral transform tests)
  - DeepSpeed ZeRO-2 configs for 4 and 8 GPU
  - docs/compute_resources.md (N=30 budget)
  - README documenting the full publication experiment
```

This repo is ready to run when compute time is available. No deadline pressure.

### 4.2 Existing Repo: N=1 Class Project

Modify the existing repo (`samwick07/csce823-spectral-kv-compression`) to support N=1 point estimates.

#### Changes Required:

**A. Orchestrator constants (src/orchestrator.py)**

```python
# CURRENT:
ALL_SEEDS = list(range(30))
PILOT_SEEDS = list(range(5))

# N=1:
ALL_SEEDS = [0]              # single seed
PILOT_SEEDS = [0]            # same seed for pilot
```

**B. Config YAMLs (configs/experiment_C*.yaml)**

Each config needs a `max_steps` or reduced `num_train_epochs` override for the class version. Two approaches:

Option 1 (recommended): Add a `training_profile` field to each config:
```yaml
# Full (publication):
training_profile: full        # uses full step count
max_steps: null               # uses num_train_epochs

# Class:
training_profile: class       # uses reduced step count  
max_steps: 500                # or num_train_epochs: 1
```

Option 2: Keep configs unchanged, add `--max-steps` CLI argument to the orchestrator that overrides all configs.

**C. Statistical analysis (src/stats/)**

The 7-step pipeline requires N>=5 for most tests. For N=1, replace with a simple point-estimate module:

New file: `src/stats/point_estimates.py`
```python
"""Point estimate analysis for N=1 class project.

Produces a table of per-config metrics with percentage delta
from baseline (C00). No statistical tests — just raw numbers.
"""
```

The existing `src/stats/analyze.py` (7-step pipeline) is preserved in the N=30 repo. In the N=1 repo, the orchestrator calls `point_estimates.py` instead.

**D. README**

Update to reflect N=1 scope, class project deadline, and reference to the full N=30 repo for publication.

**E. Exfil**

The exfil script works unchanged. N=1 results are ~10 MB (14 runs vs 420).

#### Files Changed in N=1 Repo:

| File                    | Change                                    |
|-------------------------|-------------------------------------------|
| src/orchestrator.py     | ALL_SEEDS = [0], PILOT_SEEDS = [0]        |
| configs/experiment_*.yaml | Add training_profile or max_steps field  |
| src/stats/point_estimates.py | NEW — replaces 7-step pipeline       |
| src/orchestrator.py     | Call point_estimates instead of analyze   |
| README.md               | Update scope, reference N=30 repo         |
| docs/compute_resources.md | Update with N=1 budget                  |

#### Files NOT Changed (identical in both repos):

| File                    | Why unchanged                             |
|-------------------------|-------------------------------------------|
| src/spectral/transform.py | Core DCT/FFT transforms — research core |
| src/spectral/filter.py  | Fixed LP and learnable filters            |
| src/spectral/cache.py   | Compressed attention mechanism            |
| src/training/train_redpajama.py | Training pipeline (logic unchanged) |
| src/training/train_longalpaca.py | Training pipeline (logic unchanged) |
| src/training/lora_config.py | LoRA configuration                       |
| src/eval/pg19.py        | PG-19 evaluation                          |
| src/eval/proof_pile.py  | Proof-pile evaluation                     |
| src/eval/longbench.py   | LongBench V1 evaluation                   |
| src/utils/constants.py  | Model identifiers                         |
| src/utils/config.py     | Config loading                            |
| tests/test_spectral_transforms.py | 28 unit tests                      |

The spectral engine, training pipelines, and evaluation suite are identical in both repos. Only seed count, training step count, and analysis output differ.

---

## 5. Execution Plan (N=1, 4x H200)

### Step 1: Create N=30 archive repo
```bash
# Clone current repo
git clone https://github.com/samwick07/csce823-spectral-kv-compression.git
cd csce823-spectral-kv-compression

# Create the publication archive repo on GitHub
gh repo create samwick07/csce823-spectral-kv-full --private \
  --description "Publication-grade N=30 spectral KV-cache experiment (full statistical analysis)"

# Push current code as-is
git remote add archive https://github.com/samwick07/csce823-spectral-kv-full.git
git push archive main
```

### Step 2: Modify existing repo for N=1
```bash
# In the existing repo, create a branch
git checkout -b feat/n1-class-project

# Apply changes:
# 1. orchestrator.py: ALL_SEEDS = [0]
# 2. configs: add max_steps for class training profile
# 3. New file: src/stats/point_estimates.py
# 4. orchestrator.py: call point_estimates instead of analyze
# 5. Update README
# 6. Update compute_resources.md

git add -A
git commit -m "N=1 point estimate mode for class project

Reduces eval from 420 runs (14x30) to 14 runs (14x1).
Replaces 7-step statistical pipeline with point estimate table.
Training step count reduced via config max_steps.
Full N=30 experiment preserved in csce823-spectral-kv-full repo."

git push origin feat/n1-class-project
# Create PR, merge to main
```

### Step 3: Deploy and run on cluster
```bash
# On the Coder workspace (Aug 25 or earlier if possible)
git clone https://github.com/samwick07/csce823-spectral-kv-compression.git
cd csce823-spectral-kv-compression
bash scripts/setup_env.sh
bash scripts/smoke_test.sh

# Pilot run (3 configs x 1 seed)
python -m src.orchestrator --pilot

# Full N=1 run
python -m src.orchestrator
```

### Step 4: Future N=30 run
```bash
# When compute time is available (no deadline)
git clone https://github.com/samwick07/csce823-spectral-kv-full.git
cd csce823-spectral-kv-full
bash scripts/setup_env.sh
python -m src.orchestrator  # 14-day run, full 7-step statistics
```

---

## 6. Timeline (N=1, starting Aug 25)

| Date       | Day | Activity                                          |
|------------|-----|---------------------------------------------------|
| Aug 21-24  | —   | Create N=30 archive repo, modify existing repo for N=1, PR, merge |
| Aug 25     | 1   | Provision Coder workspace, clone, setup_env, smoke test |
| Aug 25     | 1   | Pilot run (3 configs x 1 seed, ~4h)               |
| Aug 25-26  | 1-2 | Start full training (13 configs)                  |
| Aug 26-31  | 2-7 | Training continues (5.9 days on 4x H200)          |
| Aug 31     | 7   | Training done, eval (14 runs, ~22h)               |
| Sep 1      | 8   | Eval done, point estimate analysis, exfil         |
| Sep 1-2    | 8-9 | Paper writing, figures, presentation              |
| Sep 2      | 10  | Buffer                                            |

This is TIGHT if starting Aug 25. If workspace can be provisioned Aug 22-23, there's comfortable buffer.

If training is reduced to 50% steps (max_steps override in configs), the timeline becomes:

| Date       | Day | Activity                                          |
|------------|-----|---------------------------------------------------|
| Aug 25     | 1   | Setup, smoke test, pilot                          |
| Aug 25-28  | 1-4 | Training (3 days at 50% steps)                    |
| Aug 28     | 4   | Eval (14 runs, ~22h)                              |
| Aug 29     | 5   | Analysis, exfil, paper starts                     |
| Aug 29-31  | 5-7 | Paper, figures, presentation                      |
| Aug 31     | 7   | SUBMIT                                            |

This fits comfortably from an Aug 25 start.

---

## 7. Decision Points

1. **Training scope for N=1**: Full training (7.2d, tight) vs 50% steps (4.2d, comfortable) vs 25% steps (2.7d, ample buffer). Recommend 50% — shows real convergence trends while leaving paper buffer.

2. **Workspace provisioning**: Can you get the Coder workspace before Aug 25? If yes, full training fits. If strictly Aug 25, use reduced steps.

3. **N=30 archive repo name**: `csce823-spectral-kv-full` or other preference? (Note: `csce823-spectral-kv` is tombstoned and cannot be reused.)

4. **N=1 repo**: Keep using `csce823-spectral-kv-compression` (current name) or rename?
