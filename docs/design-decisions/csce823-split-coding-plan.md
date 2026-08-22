# Coding Changes: Repo Split + Experiment Setup (REVISED after remote comparison)

**Date:** 21 Aug 2026
**Repo:** /opt/csce823-spectral-kv (container hermes-d08a3748, remote samwick07/csce823-spectral-kv)
**Status:** Awaiting approval to implement

---

## 1. CRITICAL: Local container was 5 commits behind origin/main

Initial bug audit ran against stale local HEAD (85083cb). origin/main (1975b15)
already contains a 706-line resilient orchestrator (src/orchestrator.py) plus
run.sh / setup_env.sh / monitor.sh / exfil.sh, 4-GPU DeepSpeed configs,
W&B deterministic run IDs, checkpoint auto-resume, and crash-recovery state.
Local tree is clean (no uncommitted changes) → fast-forward `git pull` is safe.

### Bug audit re-graded against origin/main

| # | Bug | Origin status | Notes |
|---|-----|---------------|-------|
| B1 | eval_single.sh hardcodes CUDA_VISIBLE_DEVICES=0 | **Fixed (new path)** | Orchestrator sets CUDA_VISIBLE_DEVICES per subprocess (env dict per Popen). Old script still has the bug but is no longer called by any entry point. |
| B2 | run_parallel_eval.sh hardcoded 8 GPUs | **Obsoleted** | Orchestrator does per-config seed batches across auto-detected GPUs. Old script unused. |
| B3 | train_single.sh --num_gpus=8 hardcoded | **Fixed (new path)** | Orchestrator auto-detects GPU count (nvidia-smi) and passes --num_gpus=N. Old script unused. |
| B4 | YAMLs say num_gpus: 8 / 8-GPU DS config | **Fixed** | All 13 YAMLs → num_gpus: 4, deepspeed_zero2_4gpu.json. |
| B5 | Global batch mismatch on 4 GPUs | **Addressed — different choice** | Origin's 4-GPU config uses train_batch_size: 32 (4×8×1, accumulation=1) → global batch halves 64→32. My earlier recommendation was accumulation=2 to hold 64. **Protocol decision needed (see §6).** |
| B6 | device_map="auto" under DeepSpeed launcher | **NOT fixed** | Still line 74 in both trainers on origin/main. |
| B7 | Dead no-op block in run_experiment.py | **Fixed** | Removed in commit 9c09dc7. |
| B8 | "14 configs / 420 runs" vs actual 13/390 | **NOT fixed — now functional** | README, docs/compute_resources.md, run.sh comment, run_all.sh, setup_env.sh, experiment_matrix.py docstring all say 14/420. **monitor.sh hardcodes 14 and 420 as progress denominators** — dashboard will show 13/14 configs and ≤390/420 evals forever, so progress tracking is wrong, not just docs. |

### New on origin (not in my original outline)

- src/orchestrator.py — single entry point: idempotent, crash-safe, marker-file
  completion tracking (.training_complete), atomic orchestrator_state.json,
  auto-resume from latest DeepSpeed checkpoint, --pilot (C00/C07/C10 × 5 seeds),
  --phase train|eval|analyze|full, --status, SIGINT/SIGTERM graceful shutdown.
- scripts/run.sh — tmux (nohup fallback) launcher, WANDB_MODE passthrough,
  --attach/--stop/--status modes.
- scripts/setup_env.sh — venv, deps, model+dataset download, GPU/HF checks.
- scripts/monitor.sh — live dashboard (GPU usage, progress, logs).
- scripts/exfil.sh + src/utils/manifest.py — resumable HF Hub upload of results.
- W&B: per-phase deterministic run IDs (C01_phase1_redpajama),
  trainer.train(resume_from_checkpoint=latest), W&B init moved into each trainer.
- requirements.txt: +huggingface_hub>=0.25, +pytest>=8.0.
- deepspeed_zero2_4gpu.json + _longctx.json.

---

## 2. Post-Split Topology (unchanged)

```
samwick07/spectral-kv            NEW  N=30 publication archive (frozen protocol)
samwick07/csce823-spectral-kv    EXISTING, keeps name — N=1 class variant, runs tonight
```

Implementation order:
1. `git pull` in container (fast-forward to 1975b15).
2. Fix remaining gaps (Phase A, §3) on a branch → test → merge to main → push.
3. Push main to new spectral-kv remote, tag n30-v1.0, README edit.
4. In csce823-spectral-kv: branch feat/n1-class → N=1 changes (§4) → merge → tag n1-v1.0.

---

## 3. Phase A — Remaining Gaps (both repos, small now)

### A1. `git pull` in container
```bash
cd /opt/csce823-spectral-kv && git pull --ff-only origin main
# verify: git rev-parse HEAD == 1975b15
```

### A2. device_map fix (B6)
`src/training/train_redpajama.py:74` and `train_longalpaca.py:74`:
```python
device_map="auto",   →   device_map=None,   # DeepSpeed launcher owns placement
```
8B bf16 ≈ 16 GB fits on one H200; ZeRO-2 shards optimizer states/gradients.

### A3. Config-count correction (B8)
- monitor.sh: replace hardcoded 14/420 denominators — derive from
  len(EXPERIMENT_MATRIX) and len(ALL_CONFIG_IDS)×len(ALL_SEEDS) (or hardcode 13/390
  with a comment pointing at experiment_matrix.py).
- README.md (lines 97, 107–108), docs/compute_resources.md (106, 118, 159, 207, 212),
  scripts/run.sh comment (13), scripts/run_all.sh comments (2, 5),
  scripts/setup_env.sh (204), src/stats/experiment_matrix.py docstring (1):
  "14" → "13", "420" → "390".

### A4. Dead old scripts
run.sh + orchestrator fully replace: run_all.sh, run_parallel_eval.sh,
run_seed_sweep.sh, train_single.sh, eval_single.sh, analyze.sh.
Recommendation: **delete all six** — they contain the stale bugs (B1–B3) and two
entry points for the same pipeline is how the container got stale in the first place.
Keep: run.sh, setup_env.sh, monitor.sh, exfil.sh, smoke_test.sh.
(Counter-option: fix+keep for manual single-config work. Orchestrator already covers
that via `--config C01 --seed 0`, so delete.)

### A5. Optional: eval throughput tweak (N=30 benefit)
Orchestrator eval loop batches per config (30 seeds / 4 GPUs = 8 batches per config;
last batch of 2 uses 2 GPUs). A global cross-config seed queue would balance better
over the long run (420 → 390 runs, tail effect ~2–3h on the full run).
Low priority: implement only if the N=30 archive is expected to run within months.

### A6. Verification
1. `pytest` in the container after pull + A2/A3.
2. On the 4× H200 workspace: `bash scripts/setup_env.sh` → `bash scripts/smoke_test.sh`
   → **`bash scripts/run.sh --pilot`** (built-in: C00/C07/C10 × 5 seeds; closes P1-3
   dataset verification at the same gate) → `bash scripts/run.sh --status`.
3. Pilot results roll into the final N=1 dataset (seed 0 of each config).

> Pilot now costs 15 evals (5 seeds) instead of 1; extra 4 seeds/config are free
> insurance for the class repo and useful sanity data. No change to the N=30 archive.

---

## 4. Phase B — spectral-kv (N=30 archive)

```bash
gh repo create samwick07/spectral-kv --private --description \
  "Spectral KV-Cache Compression: N=30 publication archive"
git remote add archive https://github.com/samwick07/spectral-kv.git
git push archive main && git tag -a n30-v1.0 -m "N=30: 13 configs x 30 seeds, 7-step stats"
git push archive n30-v1.0
```
README edit: "Publication archive. Full N=30 factorial, 7-step statistics
(Wilcoxon/Friedman/Nemenyi/KS/ART-ANOVA/Holm). No deadline:
bash scripts/setup_env.sh && bash scripts/run.sh". No protocol changes.

## 5. Phase C — N=1 (csce823-spectral-kv)

- C1 NEW `src/stats/point_estimates.py`: reads `results/raw/*/seed_0/*.json`,
  emits `results/point/point_table.{csv,md}` (per-benchmark + delta vs C00,
  "N=1 point estimate — no significance tests; see spectral-kv" header).
- C2 All 13 YAMLs: `num_seeds: 30 → 1`.
- C3 **Orchestrator N=1 support** (supersedes old run_all.sh plan):
  `ALL_SEEDS = list(range(30))` → module constant read from config/env, or add
  `--seeds N` CLI flag (default 30) to orchestrator main() →
  `python -m src.orchestrator` unchanged for N=30, `--seeds 1` for N=1.
  (Cleaner than forking orchestrator logic; flag lives in both repos.)
- C4 `python -m src.orchestrator --phase analyze` in N=1 should call
  point_estimates instead of aggregate+analyze — implement via a small branch:
  if effective seed count == 1 → point_estimates, else aggregate+analyze.
  (Keep aggregate.py/analyze.py in tree, inert at N=1.)
- C5 README: class-project scope + `bash scripts/run.sh --seeds 1` usage.
- C6 Tag n1-v1.0.

## 6. DECISIONS NEEDED

1. **Global batch (B5)**: origin's 4-GPU config halves the global batch to 32
   (accumulation=1). My recommendation: `gradient_accumulation_steps: 2`
   → `train_batch_size: 64`, keeping training dynamics identical to the 8-GPU
   protocol and comparable to any future 64-GPU run. One-line config change.
2. **Dead scripts (A4)**: delete the six superseded scripts (recommended) or fix+keep?
3. **Matrix size**: code has 13 configs (C00–C12); docs say 14. Confirm 13 is the
   intended matrix (or that one config was lost in the repo consolidation).
4. **A5 eval queue**: implement the global seed queue now (helps N=30 run) or defer?
5. Go-ahead to `git pull` + create samwick07/spectral-kv now?

## 7. Run Setup (tonight, 4× H200, after merge)

```bash
bash scripts/setup_env.sh          # venv, deps, model + datasets
bash scripts/smoke_test.sh         # gate 1
bash scripts/run.sh --pilot        # gate 2: C00/C07/C10 x 5 seeds (also closes P1-3)
bash scripts/run.sh --seeds 1      # full N=1 matrix, ~143h train + ~20h eval ≈ 6.8 days
bash scripts/run.sh --status       # anytime
bash scripts/monitor.sh --watch    # live dashboard
```
Start 21–22 Aug ⇒ finish ~28–29 Aug.
