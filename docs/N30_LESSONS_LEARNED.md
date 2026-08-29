# N=30 Experiment: Lessons Learned & Hardening Plan

**Source:** N=1 pilot run (C00, C01, C07) — August 22-24, 2026 (training); August 28, 2026 (evaluation)
**Goal:** A fully autonomous N=30 experiment with zero runtime code changes, patches, or mid-run script adjustments.

---

# PART I: TRAINING

---

## 1. Training Pre-Launch Checklist

### 1.1 Credential Validation
- [ ] **Verify HF_TOKEN is full-length (37+ chars), not truncated.** The N=1 run had `hf_qFR...LjlE` (13 chars with literal dots) saved in `.env.spectral`, causing 38 PEFT 401 warnings at every checkpoint save.
- [ ] **Verify HF_TOKEN has gated repo access.** Run `huggingface_hub.HfApi().whoami()` and attempt `hf_hub_download("meta-llama/Llama-3.1-8B-Instruct", "config.json")` before launching.
- [ ] **Verify WANDB_API_KEY is valid.** Run `wandb.Api().whoami()` before launching.
- [ ] **Add a pre-flight check script** (`scripts/preflight.sh`) that validates all credentials and fails fast with a clear message if any are missing/invalid.

### 1.2 Environment Hardening
- [ ] **Set WANDB_PROJECT in .env.spectral.** The N=1 run had WANDB_PROJECT missing, causing the HF Trainer's WandbCallback to default to `project="huggingface"`, sending 3 runs to the wrong project.
- [ ] **Set HF_HUB_OFFLINE=1 after model cache is populated.** PEFT's checkpoint save tries to fetch `config.json` from HuggingFace Hub even when the model is fully cached. Setting `HF_HUB_OFFLINE=1` eliminates these 401 warnings entirely once the model is in local cache.
- [ ] **Validate DeepSpeed config matches YAML.** The N=1 run had `warmup_max_lr=2e-05` in DeepSpeed config vs `learning_rate=1e-05` in YAML, causing all 13 configs to crash on Aug 22.

### 1.3 Training Data Verification
- [ ] **Pre-download and verify all training datasets** (RedPajama-Data-1T streaming, LongAlpaca-12k).
- [ ] **Verify LongAlpaca column names** (instruction/input/output/file). A column mismatch was fixed in commit 7a7ecf3.
- [ ] **Run dataset verification** per `docs/dataset_verification.md`.

---

## 2. WandB Integration Hardening (Training)

### 2.1 Deleted Run ID Problem (FIXED in code)
**Issue:** The orchestrator generates deterministic W&B run IDs via `new_run_id()` per `(config, phase)`. When the user deleted failed runs on the W&B site, the server permanently forbade reusing those IDs. `init_wandb(id=<deleted-id>)` raised `RuntimeError`, which was silently swallowed, and the HF callback fell back to a random ID in the wrong project.

**Fix applied (commit 9586a28):** `init_wandb` now catches the "previously created and deleted" error and retries with a suffixed ID (e.g., `C01_phase1_redpajama` → `C01_phase1_redpajama_r`).

**N=30 action:** Never delete W&B runs mid-experiment. If a run is corrupted, mark it with a tag and create a new one. Add a pre-flight check that verifies all deterministic run IDs are available (not in deleted state) before launch.

### 2.2 Project Routing (FIXED in code)
**Issue:** The HF Trainer's `WandbCallback` uses `project=os.getenv("WANDB_PROJECT", "huggingface")`. If `WANDB_PROJECT` is unset, all metrics land in a default `huggingface` project.

**Fix applied (commit 9586a28):** `scripts/relaunch.sh` now exports `WANDB_PROJECT=csce823-spectral-kv`, and the training scripts pass it through to the callback.

**N=30 action:** Verify `WANDB_PROJECT` is set in the environment before launch. Add to pre-flight check.

### 2.3 Metric Key Consistency (FIXED in code)
**Issue:** The C00 Phase 1 recovery script (`fix_wandb_phase1.py`) logged metrics with bare keys (`loss`, `grad_norm`), while the HF Trainer's WandbCallback logs with `train/` prefix (`train/loss`, `train/grad_norm`). This split C00 P1 into a separate "Charts" section on W&B.

**Fix applied (commit a8e900e):** Recovery scripts now use `train/` prefix for all metrics.

**N=30 action:** No action needed — all live runs use the HF callback which produces consistent `train/` keys. Recovery scripts (if needed) must match.

### 2.4 Phase Merging (FIXED in code)
**Issue:** When `init_wandb` failed (deleted ID), the HF callback opened its own run. Because the run was never explicitly closed between Phase 1 and Phase 2, Phase 2 data was appended to the same run, corrupting the step axis (Phase 2 steps restart from 10).

**Fix applied (commit 9586a28):** `init_wandb` now calls `wandb.finish()` on any stale active run before initializing a new one.

**N=30 action:** No action needed — the fix ensures clean phase boundaries.

### 2.5 Run Naming
**Issue:** Run names are inconsistent across the project (some have "(clean re-log)" suffix, some have `_r` suffix, C07 uses a different pattern).

**N=30 action:** After the N=1 experiment, rename all W&B runs to a consistent scheme (e.g., `C00 Phase 1 RedPajama`, `C00 Phase 2 LongAlpaca`). For N=30, set the display name explicitly in `wandb.init(name=...)` to avoid needing post-hoc renames.

---

## 3. Training Stability

### 3.1 DeepSpeed Configuration
- **Gradient accumulation mismatch:** Accelerate's `GradientAccumulationPlugin` defaults to 1, while DeepSpeed config specifies 2 (Phase 1) or 16 (Phase 2). DeepSpeed wins, but the warning is noisy. Consider setting `gradient_accumulation_steps` explicitly in `TrainingArguments` to match.
- **CUDA_HOME missing:** DeepSpeed's optional JIT ops (cutlass, fp_quantizer, etc.) fail to compile. Harmless for ZeRO-2 training. Consider suppressing with `DS_BUILD_AIO=0` or documenting as expected.
- **LR scheduler warning:** "Attempting to get learning rate from scheduler before it has started" appears at startup. Harmless.

### 3.2 Checkpoint Resume
- **Lexicographic sort bug (FIXED):** Checkpoint directories were sorted lexicographically (`checkpoint-1000` before `checkpoint-500`), causing wrong resume points. Fixed with numeric sort.
- **Kill+resume step collision:** When training is killed mid-phase and resumed from a checkpoint, W&B's `resume="allow"` mode appends new steps to the same run, creating a corrupted step axis. The `init_wandb` fix (closing stale runs) prevents this, but if a run is killed and resumed, the W&B data may still show duplicate steps. Consider using `resume="must"` with proper checkpoint-to-step mapping, or `resume="never"` for clean re-logs.

### 3.3 Training Dataset Issues
- **RedPajama-Data-1T-Sample removed from HuggingFace.** The training code falls back to streaming the full `RedPajama-Data-1T` dataset. This works but requires network access during Phase 1.
- **LongAlpaca column mismatch (FIXED in 7a7ecf3).** The dataset uses `instruction/input/output/file` columns; the training code was updated to match.

---

## 4. Loss Computation Verification

### 4.1 The 100x Loss Gap (OBSERVED in N=1)

**Issue:** During the N=1 run, compressed configs (C01, C07, C10) converged to a Phase 1 training loss of ~0.02, while the baseline (C00) maintained a flat loss of ~2.09. In Phase 2, compressed configs reached ~0.003-0.007 while the baseline sat at ~0.52. This 100x gap is abnormally large and could indicate either a genuine effect of spectral compression on the optimization landscape or a discrepancy in how loss is computed for the custom spectral attention forward pass.

**Root cause unknown -- must be verified before N=30.** The spectral attention forward pass overrides `LlamaAttention.forward` with a manual SDPA implementation. If the loss computation path differs from standard causal LM loss (e.g., different masking, different normalization, or the compressed/reconstructed K-V tensors affect the logits in a way that inflates or deflates the loss), then training loss comparisons between compressed and baseline configs are not apples-to-apples.

**N=30 actions:**
- [ ] **Audit the loss computation path in the spectral attention forward pass.** Trace from `SpectralKVAttention.forward()` through the model's `forward()` to the loss function. Verify that the loss function receives the same type of logits, applies the same label masking, and uses the same reduction as the baseline path.
- [ ] **Verify loss is computed on reconstructed (full) logits, not compressed intermediates.** The K/V tensors are compressed and reconstructed, but the output logits fed to the loss function must be full-dimensional. If loss is accidentally computed on compressed representations, it would appear artificially low.
- [ ] **Add a unit test that compares loss output for identical inputs** between `LlamaAttention` (baseline) and `SpectralKVAttention` (gamma=1.0, no compression). With gamma=1.0, the spectral transform should be a no-op and the losses should be numerically identical (within floating point tolerance). If they differ, the forward pass has a bug.
- [ ] **Log the loss computation path explicitly.** Add a one-time log line at training start that prints whether the model is using standard or spectral attention and which loss function is active. This makes it unambiguous in the logs which path produced the loss values.
- [ ] **Do not cite the 100x loss gap as a result until verified.** If the loss computation is confirmed correct and the gap persists, it is a genuine finding. If it is a bug, the corrected loss values may change the relative ranking of configs.

---

## 5. Orchestrator Resilience (Training)

### 5.1 Current State
- Atomic `orchestrator_state.json` persists progress across restarts.
- DeepSpeed checkpoint resume every 500 steps (Phase 1) / 100 steps (Phase 2).
- `WANDB_MODE=offline` fallback for VPN drops.
- `scripts/relaunch.sh` is idempotent and auto-restores HF cache.

### 5.2 N=30 Hardening Needed
- [ ] **Health check daemon:** A background process that monitors training progress, GPU utilization, and W&B sync status. Alerts (not stops) if something goes wrong.
- [ ] **Automatic W&B sync recovery:** If `WANDB_MODE` falls back to offline, a watcher should `wandb sync` local runs when connectivity returns.
- [ ] **Config validation before launch:** A script that validates all 13 YAML configs, DeepSpeed configs, and dataset availability before the orchestrator starts.
- [ ] **Per-config isolated W&B runs:** Ensure each `(config, phase, seed)` combination gets a unique, non-colliding W&B run ID. The current deterministic ID scheme works for N=1 but may collide with seeds in N=30.
- [ ] **Run name convention for N=30:** Define and implement a naming scheme like `C{XX} Phase {N} {dataset} (seed {S})` in the code, not post-hoc.

---

## 6. Pod Eviction from Disk Pressure (FIXED Aug 26, 2026)

### 6.1 Incident Summary

**Date:** August 25-26, 2026
**Run affected:** C11_phase2_longalpaca (crash #20 in orchestrator state)
**Impact:** Training killed at step 638/940 (68% complete, epoch 3.35). Pod was recreated by Kubernetes, losing all ephemeral state. ~22 minutes of training lost (steps 601-638, recovered from checkpoint-600).

**Root cause:** Kubernetes evicted the pod due to disk pressure on the node. The HuggingFace model cache (~55 GB for Llama-3.1-8B-Instruct + dataset caches) was stored on the ephemeral container overlay filesystem (`$HOME/.cache/huggingface/`), which is wiped on every pod stop/start. The overlay reached 94.3% capacity (208.3 GB / 233 GB), triggering kubelet eviction.

**Evidence:**
- Pod hostname changed from `...-rpkjm` (during crash) to `...-tqr5l` (after restart), confirming pod recreation.
- Orchestrator log cut off mid-step at step 638 with no Python traceback, no SIGTERM handler invoked, and no error message — the process was SIGKILLed externally.
- WandB system metrics showed disk at 94.3% throughout the run, climbing from 93.6% at start.
- Root overlay dropped from 94.3% to 66% after pod recreation (the 55 GB HF cache was gone).
- No dmesg/journalctl access inside the container (Kubernetes hides host kernel logs).

### 6.2 Why It Did Not Gracefully Resume

Two independent failures prevented automatic recovery:

**Failure 1: No autostart mechanism.** The orchestrator was launched with `setsid nohup`, which survives SSH disconnects and workspace agent restarts but NOT pod recreation. When Kubernetes killed the pod, PID 1 (`./coder agent`) and all child processes died. The new pod had no crontab, no systemd service, and no Coder startup script to relaunch the orchestrator. The `.orchestrator_pid` file pointed to a dead PID (245570).

**Failure 2: HF cache wiped.** The live HF model cache lived on the ephemeral overlay (`$HOME/.cache/huggingface/`). When the pod was recreated, the overlay was reset to a clean state. A backup existed on the PVC at `/workspaces/hf-cache-backup/` (created by `run.sh`'s cache backup logic), but the `run.sh` restore logic only checked for the model marker in `$HOME/.cache/huggingface/` — it would have restored on next launch, but there was no launch happening because no autostart existed.

### 6.3 Fixes Applied

**Fix 1: HF cache moved to persistent PVC.**
- Copied the 55 GB model cache from `/workspaces/hf-cache-backup/` to `/workspaces/.cache/huggingface/` (on the 1 TB Longhorn PVC).
- Added `export HF_HOME="/workspaces/.cache/huggingface"` and `export TRANSFORMERS_CACHE="/workspaces/.cache/huggingface/hub"` to `/workspaces/.env.spectral`.
- Patched `scripts/run.sh` to use `HF_CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}"` instead of the hardcoded `$HOME/.cache/huggingface`. This ensures `run.sh` checks the PVC-backed location for the model marker and only restores from backup if truly missing.
- Root overlay now stays at 66% instead of climbing to 94%+ during training.

**Fix 2: Watchdog auto-restart mechanism.**
- Created `scripts/watchdog.sh`: a lightweight polling daemon that checks every 5 minutes whether the orchestrator PID is alive. If dead and work remains (training incomplete, evals pending, or analysis/exfil not done), it sources `/workspaces/.env.spectral` and calls `scripts/relaunch.sh`.
- Added a line to `~/.bashrc` that starts the watchdog via `nohup` on any new shell session (which happens after a pod restart when someone SSHs in or VS Code connects).
- The watchdog checks `results/orchestrator_state.json` to avoid restarting when all work is complete.
- Note: `crontab` and `systemd` are not available in this container image. The `.bashrc` approach is a pragmatic fallback — it requires at least one shell session to start after a pod restart. A more robust solution would be a Coder workspace startup script or a Kubernetes init container.

### 6.4 Verification

After applying fixes and relaunching:
- Orchestrator started as PID 302416, detected 4 GPUs, loaded state (crash_count: 20).
- Skipped 7 completed configs (C00, C01, C02, C04, C07, C08, C10) automatically.
- C11 Phase 2 resumed from `checkpoints/C11/phase2_longalpaca/checkpoint-600` (confirmed in log: "Resuming Phase 2 from checkpoints/C11/phase2_longalpaca/checkpoint-600").
- WandB confirmed run resume: "Resuming run C11_phase2_longalpaca".
- All 4x H200 GPUs at 100% utilization, training at step 602+ with ~33s/step.
- Root disk stable at 66% (no HF cache on overlay).

### 6.5 Remaining Hardening for N=30

- [ ] **Set `HF_HOME` globally in the Coder workspace template** (not just `.env.spectral`). This ensures all processes — including eval scripts, analysis scripts, and ad-hoc Python — use the PVC-backed cache.
- [ ] **Add a disk pressure pre-flight check.** Before launching training, verify root overlay usage is below 80% and PVC has at least 100 GB free. Fail fast if disk is critical.
- [ ] **Consider a Kubernetes liveness/readiness probe** that checks orchestrator PID health. If the orchestrator dies, Kubernetes can restart the pod (which would trigger the watchdog via `.bashrc`). This is more reliable than relying on a shell session.
- [ ] **Set `HF_HUB_OFFLINE=1` after model load.** This was already noted in Section 1.2 but is especially relevant here: if PEFT doesn't try to fetch `config.json` from HuggingFace at every checkpoint save, there's no network dependency during training and no risk of a network error killing a checkpoint save.
- [ ] **Monitor PVC disk usage.** With the HF cache now on PVC (55 GB) plus checkpoints (~8 GB) plus venv (~7 GB), PVC usage is at 13% (128 GB / 1 TB). This is healthy, but N=30 will add ~78 GB of checkpoints. Add PVC usage to the health check dashboard.
- [ ] **Document the `save_total_limit` interaction with crash recovery.** Phase 2 uses `save_total_limit=5`, meaning only the 5 most recent checkpoints are kept. If a crash happens after step 600 and training resumes from checkpoint-600, the older checkpoints (200-500) are deleted by the trainer to enforce the limit. This is correct behavior but means there's no way to roll back further than the 5th-newest checkpoint.

---

## 7. Recovery Scripts (Training)

The following recovery scripts exist in `scripts/` and were tested during the N=1 run:
- `fix_wandb_phase1.py` — Re-logs C00 Phase 1 from `c00_phase1_clean_metrics.txt`
- `fix_wandb_phase2.py` — Re-logs C00 Phase 2 from `output.log` + orchestrator log
- `fix_wandb_C01.py` — Re-logs C01 Phase 1+2 from orchestrator log

**N=30 principle:** Recovery scripts should not be needed. If they are, the experiment design has failed. The goal is to prevent the conditions that require recovery, not to improve recovery tooling.

---

# PART II: EVALUATION

---

## 8. Evaluation Pre-Launch Checklist

### 8.1 Eval Dependencies
- [ ] **Install `zstandard` in the venv and add to workspace template.** Required by `EleutherAI/proof-pile-2` for `.jsonl.zst` decompression. The venv install is ephemeral — if the pod is recreated, the package will be missing. Add to `setup_env.sh` or the workspace Dockerfile.
- [ ] **Verify LongBench org name** is THUDM, not THUIAR.

### 8.2 Eval Dataset Pre-Caching
- [ ] **Pre-download ALL eval datasets before launching eval.** PG-19, Proof-pile, and LongBench V1 (14 tasks) must be in the HF cache. For N=30 (390 eval runs), parallel downloads will cause 429 storms. Use `scripts/preload_datasets.py`.
- [ ] **Add a pre-eval dataset check to the orchestrator.** Before launching eval subprocesses, verify that all benchmark datasets are cached locally. Fail fast with a clear message if any are missing, rather than crashing mid-eval after wasting GPU time on PG-19.

### 8.3 GPU Hygiene Before Eval
- [ ] **Check for stale GPU processes before launching eval.** Run `nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader` and kill any orphaned processes from previous runs. Stale processes holding GPU memory cause `device_map="auto"` to fail silently, splitting the model across CPU/GPU (see Section 11).

---

## 9. Proof-pile Dataset Crash (FIXED Aug 28, 2026)

**Issue:** All three Proof-pile dataset sources failed in sequence, killing every eval run:
- `hoskinson-center/proof-pile` — HTTP 429 Too Many Requests. Four parallel eval jobs (one per GPU) simultaneously attempted to download the 523 MB `proofpile_train_5.jsonl.gz` file, triggering HuggingFace Hub rate limiting.
- `EleutherAI/proof-pile-2` — `No module named 'zstandard'`. The `zstandard` Python package was not installed in the venv, but `proof-pile-2` stores data in `.jsonl.zst` format which requires it for decompression.
- `EleutherAI/proof-pile` — Dataset no longer exists on the Hub (deprecated in favor of proof-pile-2).

After all three sources failed, the code raised `RuntimeError("Could not load Proof-pile")`, killing the eval process for that config. Since every config hit the same error, all 13 eval runs failed.

**Fix applied:**
- Installed `zstandard` (v0.25.0) in the venv and added it to `requirements.txt`.
- Pre-downloaded `hoskinson-center/proof-pile` (test split, 46,251 rows) to the PVC-backed HF cache via `scripts/preload_datasets.py`, eliminating runtime downloads and 429 collisions.
- Added `trust_remote_code=True` to the `EleutherAI/proof-pile-2` load call (required by `datasets` 3.x for datasets with custom loading scripts).
- Added retry logic with exponential backoff (30s, 60s) for 429 errors in `src/eval/proof_pile.py`, so transient rate limits don't kill an eval run.

**Verification:** Smoke test confirmed `"Loaded Proof-pile from hoskinson-center/proof-pile: 46251 rows"` and `"Proof-pile results: mean=6.14, median=5.46"`.

---

## 10. WandB Import Failure in Eval Mode (FIXED Aug 28, 2026)

**Issue:** The orchestrator launches eval subprocesses as `python src/run_experiment.py` (script mode), not `python -m src.run_experiment` (module mode). In script mode, Python does not set `__package__`, so the relative import `from .utils.wandb_utils` inside `run_evaluation()` fails with `ImportError: attempted relative import with no known parent package`.

The top of `run_experiment.py` already had a try/except fallback for this (using absolute imports via `sys.path.insert`), but the W&B import inside `run_evaluation()` was a separate relative import with no fallback. When it failed, the code caught the exception and logged `"W&B init failed: ... Continuing without W&B."` — meaning PG-19 perplexity results were computed but silently dropped on the floor.

**Fix applied:** Added a try/except ImportError fallback on the W&B import inside `run_evaluation()` in `src/run_experiment.py`, mirroring the existing pattern at the top of the file:
```python
try:
    from .utils.wandb_utils import init_wandb, log_eval_results, finish_wandb
except ImportError:
    from src.utils.wandb_utils import init_wandb, log_eval_results, finish_wandb
```

**N=30 action:**
- [ ] **Audit all relative imports in `src/`.** The W&B import was the second instance of this pattern (the first was fixed at the top of `run_experiment.py`). Any remaining `from .module import ...` statements inside functions will fail in script mode. Run a grep audit and add fallbacks proactively.

---

## 11. past_key_value Naming Bug (FIXED Aug 28, 2026)

**Issue:** The code passed the `SpectralDynamicCache` to `model.generate()` using the kwarg name `past_key_value` (singular), but transformers 4.57.6 only recognizes `past_key_values` (plural). The singular form is not in transformers' `ALL_CACHE_NAMES` list, so `_validate_model_kwargs()` rejected it with:

```
ValueError: The following `model_kwargs` are not used by the model: ['past_key_value']
```

This only affected compressed configs (C01-C12) because C00 (baseline) has `spectral_cache=None` and never passes a cache to `generate()`. In the first eval run, C00 completed all benchmarks successfully while all 12 compressed configs failed at LongBench on every sample (98 total failures logged).

The bug existed in three eval modules that call `model.generate()`:
- `src/eval/longbench.py` — `gen_kwargs["past_key_value"]` (7 occurrences)
- `src/eval/metrics.py` — `gen_kwargs["past_key_value"]` (7 occurrences)
- `src/eval/efficiency.py` — passed through to `metrics.py` (4 occurrences)

And in the spectral attention forward wrapper:
- `src/spectral/attention.py` — the `wrapped_forward` and `_spectral_forward` function signatures used `past_key_value=None` (11 occurrences)

**Fix applied:** Regex replacement of all standalone `past_key_value` (not followed by `s`) with `past_key_values` across 5 files (31 total replacements). Also fixed the Llama attention forward signature in `wrapped_forward` to match transformers 4.57.6's `LlamaAttention.forward(self, hidden_states, position_embeddings, attention_mask, past_key_values, cache_position, **kwargs)`.

**Verification (v3 run):** C01 (DCT + fixed, gamma=0.5) — the first compressed config — completed PG-19 (mean=1.12, logged to W&B), Proof-pile (mean=1.15, logged to W&B), and entered LongBench with zero `model_kwargs` errors. All 4 GPUs at 93-97% utilization, generating tokens successfully.

**N=30 action:**
- [ ] **Pin the transformers version in `requirements.txt`.** The `past_key_value` vs `past_key_values` naming has changed across transformers versions. Pinning prevents a future `pip install` from silently breaking the eval.
- [ ] **Add a unit test that calls `model.generate()` with a SpectralDynamicCache.** This would have caught the naming mismatch immediately instead of discovering it at eval time.

---

## 12. Device Mismatch from Stale GPU Processes (FIXED Aug 28, 2026)

**Issue:** After killing the first eval run (which failed on `model_kwargs`) and relaunching, compressed configs immediately crashed on PG-19 with:

```
RuntimeError: Expected all tensors to be on the same device, but got index is on cuda:0,
different from other tensors on cpu (when checking argument in method wrapper_CUDA__index_select)
```

**Root cause:** Stale processes from previous eval runs (zombie C00 smoke tests and the killed orchestrator's subprocesses) were still holding 140 GB on GPU 0, leaving only 2.7 GB free. When the new eval run loaded C01 with `device_map="auto"`, accelerate couldn't fit the 16 GB model on GPU 0 and silently split it across CPU and GPU. The `embed_tokens` layer ended up on CPU while the input `input_ids` was on `cuda:0`.

**Fix applied (two parts):**

1. **Killed all stale processes:** `kill -9` on all orphaned PIDs from previous eval runs. After cleanup, all 4 GPUs showed 143 GB free each.

2. **Added explicit device movement after PeftModel loading** in `src/run_experiment.py`:
```python
# Ensure all submodules (including spectral caches added after
# from_pretrained) are on the correct device. device_map="auto"
# places the base model, but newly registered spectral_cache submodules
# start on CPU and need explicit movement.
target_device = next(model.parameters()).device
model = model.to(target_device)
```

This ensures that spectral cache submodules (registered via `attn.add_module("spectral_cache", ...)`) are moved to the correct device even if `device_map="auto"` has already placed the base model.

**N=30 action:**
- [ ] **Add stale-process cleanup to the orchestrator's eval phase.** Before launching eval subprocesses, kill any processes still holding GPU memory. A simple `nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs kill` would suffice.
- [ ] **Stagger parallel eval launches by 30s.** Even with clean GPUs, 4 simultaneous model loads can cause transient GPU memory spikes. A small stagger between GPU launches would reduce OOM risk without meaningfully increasing wall-clock time.

---

## 13. No GitHub Results Push (FIXED Aug 28, 2026)

**Issue:** The `.gitignore` excludes `results/`, `checkpoints/`, and `logs/` to keep the repo lean during development. The existing `scripts/exfil.sh` uploads to HuggingFace Hub only. There was no mechanism to push eval result JSONs to GitHub for version control and collaboration.

**Fix applied:**
- Created `scripts/push_results_github.sh`: force-adds eval result JSONs (bypassing `.gitignore`), commits with a descriptive message including timestamp, and pushes to origin.
- Added a "Phase 4b: GitHub Push" stage to `src/orchestrator.py` that runs after the HF Hub exfil completes, calling the push script as a subprocess.

---

## 14. Harmless Generation Warnings (NOT A BUG)

During LongBench evaluation, the logs fill with repeated warnings:

```
The attention mask and the pad token id were not set. As a consequence, you may observe
unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.
```

**These are harmless.** Two things are happening:

1. **"attention mask not set"** — The LongBench code calls `model.generate(input_ids, ...)` without explicitly passing `attention_mask`. Since each sample is a single sequence (batch_size=1, no padding), transformers auto-generates an all-ones mask, which is correct. For single-sequence generation there is no practical impact — the mask would be all 1s anyway.

2. **"Setting pad_token_id to eos_token_id:128001"** — Llama-3.1's tokenizer has no dedicated pad token by design. When `generate()` needs one (for knowing when to stop in batched mode), it defaults to the EOS token (128001). Since LongBench generates one sample at a time with no padding, this has zero effect on output quality.

**N=30 action:** None required. These warnings are cosmetic noise from HuggingFace's generate() pipeline. If cleaner logs are desired, pass `attention_mask=torch.ones_like(input_ids)` and `pad_token_id=tokenizer.eos_token_id` explicitly to `generate()`, but this will not change any results.

---

## 15. Eval Files Changed (Aug 28, 2026)

| File | Change |
|------|--------|
| `src/eval/proof_pile.py` | Added `trust_remote_code=True` for proof-pile-2, retry logic with backoff for 429 errors |
| `src/eval/longbench.py` | Fixed `past_key_value` → `past_key_values` (7 replacements) |
| `src/eval/metrics.py` | Fixed `past_key_value` → `past_key_values` (7 replacements) |
| `src/eval/efficiency.py` | Fixed `past_key_value` → `past_key_values` (4 replacements) |
| `src/spectral/attention.py` | Fixed `past_key_value` → `past_key_values` in forward wrapper (11 replacements) |
| `src/run_experiment.py` | Added try/except ImportError fallback for W&B import; added `model.to(target_device)` after PeftModel loading |
| `src/orchestrator.py` | Added "Phase 4b: GitHub Push" stage after HF exfil |
| `requirements.txt` | Added `zstandard` |
| `scripts/push_results_github.sh` | New script: force-add results JSONs, commit, push to GitHub |
| `scripts/preload_datasets.py` | New script: pre-download eval datasets to HF cache |

---

## 16. Remaining Eval Hardening for N=30

- [ ] **Add `zstandard` to the Coder workspace template.** The venv install is ephemeral — if the pod is recreated, the package will be missing again. Either add it to `setup_env.sh` or the workspace Dockerfile.
- [ ] **Pre-download ALL eval datasets before launching eval.** PG-19 and Proof-pile are now cached, but LongBench V1 (14 tasks) is still downloaded at eval time. For N=30 (390 eval runs), parallel downloads will cause 429 storms. Add LongBench datasets to `scripts/preload_datasets.py`.
- [ ] **Add a pre-eval dataset check to the orchestrator.** Before launching eval subprocesses, verify that all benchmark datasets are cached locally. Fail fast with a clear message if any are missing, rather than crashing mid-eval after wasting GPU time on PG-19.
- [ ] **Audit all relative imports in `src/`.** The W&B import was the second instance of this pattern (the first was fixed at the top of `run_experiment.py`). Any remaining `from .module import ...` statements inside functions will fail in script mode. Run a grep audit and add fallbacks proactively.
- [ ] **Stagger parallel eval launches by 30s.** Even with cached datasets, 4 simultaneous model loads can cause transient GPU memory spikes. A small stagger between GPU launches would reduce OOM risk without meaningfully increasing wall-clock time.
- [ ] **Add eval result verification to the orchestrator.** After each eval subprocess completes, verify that `all_results.json` exists, is valid JSON, and contains all 4 benchmark keys (`pg19`, `proof_pile`, `longbench`, `efficiency`). This prevents silent partial results from being marked as complete.
- [ ] **Pin the transformers version in `requirements.txt`.** The `past_key_value` vs `past_key_values` naming has changed across versions. Pinning prevents a future `pip install` from silently breaking the eval.
- [ ] **Add a unit test that calls `model.generate()` with a SpectralDynamicCache.** This would have caught the naming mismatch immediately instead of discovering it at eval time.
- [ ] **Add stale-process cleanup to the orchestrator's eval phase.** Before launching eval subprocesses, kill any processes still holding GPU memory.

---

# PART III: CROSS-CUTTING CONCERNS

---

## 17. HF Token Security
- [ ] **Never redact tokens in env files.** The N=1 run had the HF_TOKEN literally replaced with `hf_qFR...LjlE` (the redacted form became the actual value). Add a validation check that the token is at least 35 characters.
- [ ] **Store tokens only in `/workspaces/.env.spectral`** (PVC, outside git). Never in code, scripts, or git-tracked files.
- [ ] **Pre-flight token validation:** Verify `whoami()` succeeds and the account has access to `meta-llama/Llama-3.1-8B-Instruct` before launching.

---

## 18. N=30 Scale Considerations

### 18.1 Compute Budget
- 13 configs x 30 seeds = 390 training runs (780 phases)
- Estimated 1,216-1,340 GPU-hours on 8x H200 (~7 days wall-clock)
- At 4x H200 (current allocation): ~14 days

### 18.2 W&B Free Tier Limits
- Free tier: 100 runs per project. With 780 phases, you'll need multiple projects or a paid tier.
- **Alternative:** Use `WANDB_MODE=offline` for all runs, then sync in batches after the experiment. This eliminates sync issues during training entirely.
- **Alternative:** Split into per-config or per-seed W&B projects (e.g., `csce823-spectral-kv-C00`).

### 18.3 Checkpoint Storage
- Each Phase 1 checkpoint: ~50MB (LoRA adapters only)
- Each Phase 2 checkpoint: ~50MB (LoRA adapters only)
- 780 phases x 2 checkpoints avg = ~78GB total
- Verify PVC has sufficient storage before launch.

### 18.4 Run ID Uniqueness for N=30
- Current scheme: `C{XX}_phase{N}_{dataset}` — no seed component.
- **Must add seed:** `C{XX}_phase{N}_{dataset}_s{S}` to avoid collisions across seeds.
- Update `new_run_id()` in `src/utils/wandb_utils.py` before N=30 launch.

---

## 19. Logging Architecture

### 19.1 Monolithic Orchestrator Log (OBSERVED in N=1)

**Issue:** The N=1 run wrote all config training (Phase 1 + Phase 2), all evaluation, and all statistics output to a single orchestrator log file (`logs/orchestrator_20260823_041331.log`). This file reached 788 KB and contained interleaved output from C01, C07, and C10 -- making it difficult to extract per-config loss curves, timing, or error messages without complex grep patterns. When C00 crashed and restarted 19 times, each restart created a new orchestrator log, but all phases of the resumed config were still interleaved within each log.

**Impact on N=1 analysis:** Extracting per-config training metrics required grepping across 20+ log files and manually correlating timestamps. The C00 baseline data was particularly noisy because 19 crash/restart cycles fragmented its loss trajectory across multiple logs and WandB runs.

### 19.2 Required Logging Structure for N=30

**Principle:** Each run (config x phase x seed) must produce a unique, self-contained log file. The only exception is a safe crash resume, where the resumed run appends to the same log file with a clear `[RESUME from checkpoint-N]` marker.

**N=30 actions:**
- [ ] **Implement per-run log file naming.** Each training, evaluation, and statistics run should write to its own log file using the convention:
  - Training: `logs/train/{config_id}_phase{N}_{dataset}_seed{S}.log`
  - Evaluation: `logs/eval/{config_id}_seed{S}_{benchmark}.log`
  - Statistics: `logs/stats/{step_name}.log`
- [ ] **Modify the orchestrator to redirect subprocess output to per-run files.** The orchestrator currently captures all subprocess stdout/stderr into the monolithic orchestrator log. Instead, it should open a file handle per run and pass it as the subprocess stdout/stderr.
- [ ] **Keep the orchestrator log as a high-level summary only.** The orchestrator log should contain only: config start/stop, phase transitions, crash/recovery events, and completion markers. All per-step training output (loss, grad_norm, lr) goes to the per-run log.
- [ ] **Add a [RESUME] marker on crash recovery.** When a run safely resumes from a checkpoint, append to the existing per-run log file with a clearly delimited resume block:
  ```
  ===== [RESUME from checkpoint-N at timestamp] =====
  ```
  This makes it unambiguous that the log is a continuation, not a fresh run.
- [ ] **Do not create a new log file for a resumed run.** If the orchestrator detects that a per-run log already exists and the run is resuming from a checkpoint, it must append to the existing file. A new file is only created for a fresh run (no existing checkpoint).
- [ ] **Add a log index to the orchestrator state.** Track the per-run log file path in `orchestrator_state.json` so that post-experiment analysis can programmatically locate each run's log without guessing the naming convention.
- [ ] **Test the logging structure with a pilot run.** Before N=30, run 2-3 configs through the new logging pipeline and verify that each config/phase/seed produces a clean, isolated log file with no interleaving.

---

## 20. Summary: What Must Be True Before N=30 Launch

**Training:**
1. All credentials validated (HF token full-length + gated access, WandB key valid)
2. `WANDB_PROJECT` set in environment
3. `HF_HUB_OFFLINE=1` set after model cache populated
4. All training datasets pre-downloaded and verified
5. All 13 YAML configs validated (LR matches DeepSpeed, batch sizes match)
6. `new_run_id()` includes seed component
7. Run display names set in code (not post-hoc)
8. Pre-flight check script passes (`scripts/preflight.sh`)
9. Health check daemon configured (read-only monitoring, alert-only)
10. W&B project strategy decided (free tier limits, offline mode, or paid tier)
11. Checkpoint storage capacity verified
12. No W&B runs deleted from the project (use tags instead)
13. Loss computation path audited and verified comparable between spectral and baseline attention (Section 4)
14. Per-run log files implemented: each config/phase/seed writes to an isolated log file (Section 19)

**Evaluation:**
15. `zstandard` installed in venv and added to workspace template
16. All eval datasets pre-cached (PG-19, Proof-pile, LongBench V1)
17. Pre-eval dataset check passes in orchestrator
18. All relative imports in `src/` audited with fallbacks
19. `past_key_values` (plural) used consistently across all generate() calls
20. Transformers version pinned in `requirements.txt`
21. Stale GPU process cleanup runs before eval launch
22. Eval launches staggered by 30s to avoid memory spikes
23. Eval result verification (JSON exists, valid, all 4 benchmark keys present)
24. GitHub push script wired into orchestrator post-exfil
25. Incremental metrics capture implemented (Section 21)

---

## 21. Resilient Metrics Capture

### 21.1 Problem Statement

During the N=1 run, training time, memory efficiency, compression overhead, and token latency data were only available from two sources:

1. **WandB's `train_runtime` summary** -- written once at the end of training. If a config crashed mid-phase and resumed from checkpoint (as C00 did 19 times, and C11 did after pod eviction), the runtime metric only reflects the final successful segment, not the total wall-clock cost including crash/recovery overhead.

2. **The efficiency benchmark** (`src/eval/efficiency.py`) -- runs last in the eval pipeline, after PG-19, Proof-pile, and LongBench. If any earlier benchmark crashes (as happened for all 12 compressed configs due to the `past_key_value` bug), the efficiency benchmark never runs and zero efficiency data is captured.

**The consequence:** For the N=1 run, only C00 (baseline) has efficiency data. All 12 compressed configs have no memory, latency, or overhead measurements. Training time data for C00 is fragmented across 19 crash/restart cycles and cannot be reliably reconstructed. C11's Phase 2 runtime (3.13h) reflects only the post-eviction resume, not the full training cost including the 638 steps lost to the pod eviction.

For N=30 (390 training runs, 780 phases), this problem scales dramatically. Any crash, eviction, or kill that interrupts a run before the efficiency benchmark destroys irreplaceable efficiency data that cannot be reconstructed from checkpoints alone.

### 21.2 Required Metrics

The following metrics must be captured for every `(config, phase, seed)` combination, regardless of whether the run completes successfully:

**Training metrics (per phase):**
- `phase`: 1 (RedPajama CPT) or 2 (LongAlpaca SFT)
- `start_time`: ISO timestamp when training began
- `end_time`: ISO timestamp when training ended (or was interrupted)
- `wall_clock_seconds`: end_time - start_time (includes crash/recovery gaps for resumed runs)
- `train_runtime_seconds`: HuggingFace Trainer's `train_runtime` (compute time only, excludes gaps)
- `steps_completed`: number of optimizer steps that ran
- `steps_total`: expected total steps for this phase
- `train_loss_final`: final training loss
- `gpu_memory_peak_gb`: peak GPU memory during training (per GPU)
- `gpu_count`: number of GPUs used
- `deep_speed_config`: which DS config was used (standard vs longctx)
- `interrupted`: bool -- was this run killed/crashed before completion?
- `interrupt_reason`: "pod_eviction", "oom", "manual_kill", "crash", or null
- `checkpoint_resumed_from`: path to checkpoint if resumed, or null

**Evaluation metrics (per benchmark, per config):**
- `benchmark`: "pg19", "proof_pile", "longbench", "efficiency"
- `start_time` / `end_time`: ISO timestamps
- `wall_clock_seconds`: benchmark duration
- `completed`: bool -- did this benchmark finish?
- `error`: error message if benchmark failed, or null
- `gpu_memory_peak_gb`: peak GPU memory during this benchmark
- `gpu_id`: which GPU was assigned

**Efficiency-specific metrics (when efficiency benchmark runs):**
- `peak_kv_memory_gb`: peak KV-cache memory usage
- `decoding_latency_ms_per_token`: milliseconds per token during generation
- `compression_overhead_pct`: percentage overhead from spectral transform/reconstruct
- `total_decode_time_s`: total wall-clock time for generation
- `num_tokens_generated`: number of tokens in the generation test
- `theoretical_cache_size_gb`: expected cache size = gamma * baseline_cache_size
- `actual_cache_size_gb`: measured cache size from SpectralDynamicCache
- `cache_compression_ratio`: actual / theoretical (should be ~1.0)

### 21.3 Design: Incremental Metrics File

**Principle:** Write metrics to a persistent JSON file incrementally, after each benchmark completes -- not only at the end of the full eval pipeline. This ensures partial data survives crashes.

**File location:** `results/raw/{config_id}/seed_{seed}/metrics.json`

**Structure:**
```json
{
  "config_id": "C01",
  "seed": 0,
  "transform_type": "dct",
  "filter_type": "fixed",
  "gamma": 0.5,
  "training": {
    "phase1": {
      "start_time": "2026-08-22T19:11:19Z",
      "end_time": "2026-08-22T19:41:01Z",
      "wall_clock_seconds": 1782,
      "train_runtime_seconds": null,
      "steps_completed": 500,
      "steps_total": 1000,
      "train_loss_final": null,
      "interrupted": true,
      "interrupt_reason": "crash",
      "checkpoint_resumed_from": null,
      "attempts": [
        {"start": "...", "end": "...", "steps": 500, "reason": "crash"},
        {"start": "...", "end": "...", "steps": 1000, "reason": null}
      ]
    },
    "phase2": { ... }
  },
  "evaluation": {
    "pg19": {
      "start_time": "...",
      "end_time": "...",
      "wall_clock_seconds": 320,
      "completed": true,
      "result": {"mean_perplexity": 1.12},
      "gpu_memory_peak_gb": 19.3,
      "gpu_id": 0
    },
    "proof_pile": { ... },
    "longbench": {
      "start_time": "...",
      "end_time": null,
      "wall_clock_seconds": null,
      "completed": false,
      "error": "CUDA out of memory",
      "tasks_completed": 2,
      "tasks_total": 14
    },
    "efficiency": {
      "start_time": null,
      "end_time": null,
      "completed": false,
      "error": "not_reached"
    }
  }
}
```

**Write strategy:**
- The metrics file is created (or opened) at the start of each training phase and each eval benchmark.
- After each benchmark completes (success or failure), the corresponding section is updated and the file is flushed to disk immediately (`f.flush()` + `os.fsync(f.fileno())`).
- On crash recovery, the orchestrator reads the existing metrics file to determine which benchmarks have already been captured, avoiding redundant re-runs.
- The file uses atomic writes (write to temp, rename) to prevent corruption from mid-write crashes.

### 21.4 Implementation Points

- [ ] **Add a `MetricsCollector` class** to `src/utils/metrics_collector.py` that manages the incremental metrics file. Methods: `start_phase(phase)`, `end_phase(phase, result)`, `start_benchmark(name)`, `end_benchmark(name, result, error)`, `record_efficiency(metrics)`, `mark_interrupted(reason)`, `flush()`.
- [ ] **Call `MetricsCollector` from the training scripts** (`train_redpajama.py`, `train_longalpaca.py`) at phase start, phase end, and on crash (via `try/finally` or signal handler).
- [ ] **Call `MetricsCollector` from `run_experiment.py`** before and after each benchmark (PG-19, Proof-pile, LongBench, efficiency). Write results immediately after each benchmark completes, not after all four.
- [ ] **Record GPU memory** using `torch.cuda.max_memory_allocated()` at the end of each benchmark. Reset the peak counter at the start of each benchmark with `torch.cuda.reset_peak_memory_stats()`.
- [ ] **Record actual cache size** from `SpectralDynamicCache.get_compression_ratio()` during the efficiency benchmark. Compare to the theoretical `gamma * baseline` to verify the compression is working as expected.
- [ ] **Handle interruption gracefully.** If a process receives SIGTERM or SIGKILL (pod eviction), the `try/finally` block should write `interrupted: true` with a best-effort timestamp. For SIGKILL (which cannot be caught), the absence of an `end_time` in the metrics file serves as the signal that the run was interrupted.
- [ ] **Add metrics file to orchestrator state.** Track the metrics file path in `orchestrator_state.json` alongside the per-run log path (Section 19.2). This lets post-experiment analysis programmatically locate metrics without guessing paths.
- [ ] **Add a metrics aggregation script** (`scripts/aggregate_metrics.py`) that reads all `metrics.json` files across configs and seeds, produces a summary table (training time, memory, latency, overhead per config), and exports to CSV for statistical analysis.
- [ ] **Test the incremental write with a deliberate kill.** Start a training run, kill it mid-phase with `kill -9`, and verify the metrics file contains the partial data (start_time, steps_completed, interrupted=true, no end_time).

### 21.5 Why This Matters for N=30

At N=30 scale (390 runs), crashes are not exceptional -- they are expected. The N=1 run experienced 19 crash/restart cycles for C00 alone, a pod eviction for C11, and multiple eval crashes for all 12 compressed configs. Without incremental metrics capture, any crash that occurs before the efficiency benchmark destroys irreplaceable data.

The metrics file is the only source for:
- **Training cost analysis**: total wall-clock time per config (including recovery overhead), GPU-hours consumed, steps/second
- **Efficiency comparison**: memory savings from compression (the core research question), latency tradeoffs, overhead of spectral transforms
- **Cost-benefit analysis**: does the memory savings from lower gamma justify the perplexity degradation? This requires both quality metrics (perplexity, LongBench) and efficiency metrics (memory, latency) for the same config.

Without this data, the research cannot answer "how much memory does spectral KV-cache compression save, and at what cost in latency and quality?" -- which is the central question of the dissertation.
