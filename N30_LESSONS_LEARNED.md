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
- Added retry logic with exponential backoff (30s, 60s) for 429 errors in `src/eval/proof_pile.py`, so transie

... [OUTPUT TRUNCATED - 45,366 chars omitted out of 95,293 total] ...

shed" status, but the LongBench evaluation is actually complete with valid results on disk. C03 (same run batch) shows as "complete" in WandB with identical data structure.

**Root cause:** WandB marks a run as "crashed" when the Python process exits without calling `wandb.finish()`. During the Aug 28-29 incident (Section 22), the mass process cleanup killed eval processes externally (SIGKILL/SIGTERM). C02's eval process had already written valid results to `all_results.json` and `longbench.json`, but was killed before it reached the `finish_wandb()` call at the end of `run_evaluation()`. C03's process happened to complete the full pipeline including `wandb.finish()` before the cleanup.

The on-disk JSON files are the source of truth. WandB run status is a process-lifecycle signal, not a data-integrity signal.

### 24.2 Verification

C02 on-disk results (verified Aug 29):
- `all_results.json`: all 4 benchmark keys present, 0 errors
- LongBench: 14 tasks, 200 samples each, all with real non-zero `all_scores` arrays
- `overall_mean = 0.0207` (valid for DCT fixed gamma=0.22)
- `longbench.json`: identical data to `all_results.json` longbench section

C03 on-disk results (verified Aug 29):
- Same structure, `overall_mean = 0.0483`
- WandB status: "complete" (process finished cleanly)

The `is_eval_phase_complete()` function correctly identifies C02 as complete because it validates the JSON files on disk, not WandB status.

### 24.3 Why This Matters

A WandB "crashed" status does not mean the evaluation data is invalid. It only means the WandB SDK did not perform its clean shutdown handshake. The actual evaluation work (model loading, benchmark execution, metric computation, file writing) may have completed successfully before the process was killed.

Conversely, a WandB "complete" status does not guarantee valid on-disk results. A process could call `wandb.finish()` but fail to write `all_results.json` if an error occurred between metric computation and file save.

### 24.4 Fix Applied

No code fix needed for the current run — on-disk results are valid and `is_eval_phase_complete()` already checks disk, not WandB. However, the following hardening is needed for N=30.

### 24.5 Remaining Hardening for N=30

- [ ] **Add a `try/finally` block around `finish_wandb()` in `run_evaluation()`.** Currently, if the process is killed between writing results and calling `finish_wandb()`, WandB shows "crashed" even though results are valid. Wrap the entire evaluation in `try/finally` so `finish_wandb()` is called even on exceptions. For SIGKILL (uncatchable), document that on-disk results are authoritative.
- [ ] **Add a WandB status reconciliation script.** After all evals complete, scan each config's on-disk results. If results are valid but the WandB run shows "crashed," log a `wandb.init(resume=...)` + `wandb.finish()` to mark the run as complete. This aligns WandB status with on-disk truth.
- [ ] **Document on-disk results as the source of truth.** Add a comment in `src/orchestrator.py` near `is_eval_complete()` and `is_eval_phase_complete()` stating that on-disk JSON files are authoritative and WandB status is advisory only.
- [ ] **Add a `results_verified` flag to orchestrator_state.json.** After each eval completes, set `results_verified: true` only after validating the JSON files (not just checking process exit code). This separates "process exited 0" from "results are valid JSON with all expected keys."
- [ ] **Log a warning when WandB status disagrees with on-disk results.** If `is_eval_phase_complete()` returns True but the WandB run shows "crashed" or "failed," log a warning so the operator knows to reconcile the WandB status.

## 25. Orchestrator Pipe Deadlock -- C10 Frozen for 9.5 Hours (FIXED Aug 30, 2026)

### 25.1 Issue

During the LongBench evaluation run on Aug 29-30, C10 (fft_learnable, gamma=0.5) deadlocked for approximately 9.5 hours (from ~02:12 to ~11:42 UTC Aug 30). The process was alive (State: S sleeping, wchan=pipe_write) with the model loaded on GPU3 (18.8GB), but GPU3 was at 0% utilization and I/O stats were completely unchanged.

**Root cause:** The orchestrator captures subprocess stdout/stderr via Python subprocess pipes. When multiple subprocesses write to pipes simultaneously, the orchestrator reads them sequentially. C10's pipe buffer filled up while the orchestrator was busy reading C07's buffered output (which had been flushed when C01 completed at 10:38). With nobody draining C10's pipe, C10 blocked on `pipe_write` and could not make any forward progress -- it was deadlocked waiting to emit log output.

This is a classic producer-consumer deadlock: the producer (subprocess) cannot proceed because the consumer (orchestrator) is busy with another producer, and the pipe buffer is full.

### 25.2 Timeline

- Aug 29 14:36 -- C07 (GPU2) and C10 (GPU3) launched as batch 1 by the orchestrator. Output captured via subprocess pipes.
- Aug 29 15:45 -- Batch 2 (C04/C05/C06, C08/C09/C11/C12) launched on the same GPUs, causing severe GPU contention. C07/C10 throughput dropped dramatically.
- Aug 30 ~02:12 -- C10's I/O stats freeze at rchar=739MB. Process enters pipe_write block. GPU3 drops to 0% util.
- Aug 30 10:38 -- C01 completes. Orchestrator begins reading C07's buffered output (pipe flush). C07 resumes normal progress (~25 min/task). C10 remains deadlocked.
- Aug 30 11:41 -- C10 discovered deadlocked. Process killed, GPU3 freed, C10 relaunched standalone with output to file.

### 25.3 Detection

C10's deadlock was detectable through:
- `/proc/<pid>/io` -- rchar unchanged across two checks 9.5 hours apart
- `/proc/<pid>/wchan` -- `pipe_write` (blocked writing to a pipe)
- `/proc/<pid>/status` -- State: S (sleeping)
- `nvidia-smi` -- GPU3 at 0% utilization despite 18.8GB allocated
- No log output from GPU3 slot in the orchestrator log

### 25.4 Fix Applied

1. Killed the deadlocked C10 process (PID 2175961). It became a zombie (parent=orchestrator) but GPU3 memory was freed immediately.
2. Relaunched C10 standalone on GPU3 with `CUDA_VISIBLE_DEVICES=3` and stdout/stderr redirected to a file (`logs/eval_longbench_C10_gpu3_standalone.log`) instead of a pipe.
3. W&B run set to `resume=allow`, so it resumes `C10_eval_seed0` and merges with existing on-disk results.
4. C10 began processing LongBench tasks immediately at normal speed (~45 min/task).

### 25.5 Root Cause: Subprocess Pipe Buffering in the Orchestrator

The orchestrator uses `subprocess.Popen` with `stdout=PIPE, stderr=PIPE` to capture each eval process's output. Python pipes have a finite buffer (typically 64KB on Linux). When the orchestrator is not actively reading a subprocess's pipe, the buffer fills and the subprocess blocks on its next write.

The orchestrator reads subprocess output line-by-line in a loop that processes one GPU slot at a time. When one slot produces a large burst of output (e.g., C07's 9.5 hours of buffered logs flushing at once), the orchestrator spends all its time draining that pipe while other subprocesses (C10) block waiting for their pipes to be drained.

### 25.6 Remaining Hardening for N=30

- [ ] **Replace subprocess pipes with output files.** Launch each eval process with `stdout=open(log_path, 'w'), stderr=subprocess.STDOUT` instead of `stdout=PIPE`. This eliminates the pipe-buffer deadlock entirely -- the subprocess writes to a file and never blocks, and the orchestrator can tail the file when it wants to display progress.
- [ ] **Add a liveness watchdog per subprocess.** Every N minutes, check each running subprocess's `/proc/<pid>/io` rchar. If rchar has not changed in 30+ minutes and the process is in state S with wchan=pipe_write, flag it as deadlocked and restart it.
- [ ] **Use `select` or threading to read all pipes concurrently.** If pipes must be used, read them with `select.select()` or a dedicated reader thread per subprocess so no pipe can fill up while another is being drained.
- [ ] **Set `stderr=subprocess.DEVNULL` or merge to stdout.** Having two pipes per subprocess doubles the buffering surface area. Merging stderr into stdout (one pipe/file) halves it.

## 26. Missing `import os` Causes Crash at Results Save (FIXED Aug 30, 2026)

### 26.1 Issue

C10's standalone relaunch completed all 14 LongBench tasks and logged results to WandB, but crashed at the final save step with `NameError: name 'os' is not defined`. The `_atomic_write_json()` function in `src/run_experiment.py` uses `os.fdopen()`, `os.replace()`, and `os.unlink()`, but `import os` was missing from the module's top-level imports. The function locally imported `tempfile` but forgot to import `os`.

### 26.2 Impact

- All 14 LongBench task scores were computed and logged to WandB successfully.
- The crash occurred in `_atomic_write_json()` when attempting to write `all_results.json` and `longbench.json` to disk.
- No on-disk results were saved -- `results/raw/C10/seed_0/longbench.json` was never created.
- The orchestrator auto-restarted C10, but the running process had already cached the unfixed module, so it would crash again at the same spot.

### 26.3 Fix Applied

1. Added `import os` to the top-level imports in `src/run_experiment.py` (line 19).
2. Verified the fix with `python -c "import src.run_experiment; print('import OK')"`.
3. The fix is on disk and will be picked up by any NEW Python process that imports the module. The currently running process (PID 2922217) will NOT pick it up (Python caches modules in `sys.modules`).
4. Deployed a rescue script (`scripts/c10_rescue.py`) that monitors the running C10 process. When it exits (crash at save), the rescue script:
   a. Checks if `longbench.json` was written (in case the fix was somehow picked up).
   b. If not, fetches LongBench results from the WandB API.
   c. Writes them to `results/raw/C10/seed_0/longbench.json` and merges into `all_results.json`.
   d. Kills any redundant C10 process the orchestrator may have restarted.
5. If the WandB rescue fails, the orchestrator's auto-restart will launch a new process that imports the fixed code and succeeds on the next run.

### 26.4 Root Cause

The `_atomic_write_json()` function was added as an inline nested function inside `run_evaluation()`. It locally imports `tempfile` but uses `os.fdopen()`, `os.replace()`, and `os.unlink()` without importing `os`. The function worked in testing because `os` was often available transitively (imported by other modules in the call chain), but in the standalone relaunch (launched directly as `python src/run_experiment.py`), the module's namespace did not include `os`.

### 26.5 Remaining Hardening for N=30

- [ ] **Move `_atomic_write_json` to a utility module.** It should not be a nested function inside `run_evaluation()`. Move it to `src/utils/file_io.py` with proper imports and unit tests.
- [ ] **Add a smoke test that exercises the save path.** The test should call `_atomic_write_json()` with dummy data to verify that `os` is importable and the function works end-to-end.
- [ ] **Add `import os` to a pre-commit lint check.** Flag any file that uses `os.*` without importing `os` at the module level.
- [ ] **Consider using `pathlib.Path` methods instead of `os` functions.** `Path.write_text()`, `Path.unlink()`, and `Path.replace()` provide the same functionality without needing `import os`.

## 27. Architecture Violation: In-Memory Results Not Crash-Safe (DESIGN LESSON Aug 30, 2026)

### 27.1 Core Principle Violated

The entire end-to-end experiment was designed with a core principle: results must survive sudden power loss at the server. This means every irreplaceable piece of data must be written to disk (with fsync) before the next computation begins. Holding results in memory and writing them only at the end violates this principle.

### 27.2 What Happened

During the N=1 LongBench evaluation, `run_evaluation()` in `src/run_experiment.py` accumulated all benchmark results into a single `all_results` dict in memory. Results were written to disk only at the END of the function, via `_atomic_write_json()`. When C10 completed all 14 LongBench tasks (scoring 200-500 individual samples per task), the results sat in memory. The process then crashed at the save step (NameError: `os` not defined, Section 26).

All individual `all_scores` arrays across 14 tasks were permanently lost. Only per-task mean scores were recoverable from the log file. A full 10-hour re-run was required for the IEEE Transactions paper.

### 27.3 Root Cause: Incremental Patches Over Architecture

The in-memory accumulation was not part of the original design. It was introduced as a side effect of patching other issues:

1. Section 23: `_atomic_write_json()` was added to fix file write races. It writes the ENTIRE `all_results` dict at once, reinforcing the "accumulate then write" pattern.
2. Section 22: The `--benchmarks` flag was added to run subsets of benchmarks. Results from previous runs are loaded and merged -- but still accumulated in memory before the final write.
3. Section 21: The incremental metrics file design (Section 21.3) was proposed but never implemented. It would have written metrics after each benchmark, but the actual result data (scores, samples) was still only written at the end.

Each patch was a reasonable response to an immediate problem, but collectively they created an architecture where irreplaceable data could be lost on crash.

### 27.4 The Lesson: File-Specific Patches Are Not Permitted

Individual file-specific or error-specific patches will not be permitted for the full N=30 run. All fixes must be applied at the experiment level before launch:

1. **No mid-experiment code changes.** Once the experiment starts, the codebase is frozen. Bugs found mid-experiment require stopping, fixing, re-validating, and restarting.
2. **No in-memory-only data paths.** Any data that would be expensive or impossible to recompute must be written to disk (with fsync) before the next computation begins.
3. **Incremental writes for every benchmark and every LongBench task.** After each task completes, write its results to a per-task file. On restart, skip tasks that are already on disk.
4. **Crash-safe resume.** The eval process must be able to resume from any point -- not just from the beginning of a benchmark.
5. **Pre-experiment validation gate.** Before launch, run a full smoke test that includes crash simulation and resume verification.

### 27.5 GitHub Issues Created

The following GitHub issues track the required work at the experiment level:

- **#25**: Crash-safe incremental result persistence (per-benchmark + per-task writes with fsync)
- **#26**: Replace subprocess pipes with file-based output (eliminate pipe deadlock class)
- **#27**: Pre-experiment validation gate (smoke test full pipeline including save + crash resume)
- **#28**: Eliminate run-specific patches (enforce frozen codebase for entire experiment)
- **#29**: Single source of truth for result data (on-disk JSON authoritative, WandB advisory)
- **#30**: Watchdog/orchestrator singleton enforcement (lockfile + pgrep, no duplicate spawns)

### 27.6 What Must Be True Before N=30 Launch

In addition to the checklist in Section 20, the following must be verified:

- [ ] Results are written to disk after each LongBench task, not after all 14 tasks
- [ ] `fsync` is called after every result write
- [ ] A crash mid-eval can be resumed without re-running completed tasks
- [ ] Subprocess output goes to files, never to pipes
- [ ] The codebase is frozen (git tag) and no patches will be applied mid-experiment
- [ ] A full smoke test (including crash + resume) passes before the first config launches
- [ ] The watchdog cannot spawn a duplicate orchestrator
- [ ] On-disk JSON is the single source of truth; WandB is display-only

## 28. End-to-End Design Assessment: From Bug Fixes to Architecture Principles (Aug 30, 2026)

### 28.1 Purpose

Sections 1-27 document specific failures and their immediate fixes. Each contains valuable diagnostic data, timelines, and root-cause analysis. However, fixing individual bugs does not prepare the experiment for N=30. This section steps back to identify the systemic failure modes that produced those bugs, and the design principles that would eliminate or mitigate the entire class.

This is an assessment, not a design document. Implementation issues are tracked in GitHub.

### 28.2 Failure Mode Taxonomy

Every incident in Sections 1-27 maps to one or more of seven systemic failure modes:

---

**FM-1: No durability guarantee for computed results**

Affected sections: 22 (OOM kills lost all results including completed PG-19/Proof-pile), 25 (pipe deadlock froze C10 for 9.5h), 26 (crash at save lost all C10 all_scores), 27 (architecture violation: in-memory-only data paths)

The pattern: The pipeline computes expensive results (model generation, scoring, metrics) and holds them in memory until a single end-of-function write. Any crash, kill, or deadlock between computation and write loses everything.

The bug fixes (atomic writes, phased eval, rescue scripts) address symptoms: they make the final write safer or add recovery after loss. They do not address the root cause -- that results are not persisted at the granularity of computation.

Design principle: Every unit of computation that would be expensive to recompute must be persisted to durable storage (with fsync) before the next unit begins. For LongBench, this means per-task writes. For training, this means per-checkpoint metric snapshots. The question is never "how do we recover if we lose data" but "how do we never lose data in the first place."

GitHub issue: #25

---

**FM-2: No isolation between concurrent processes sharing resources**

Affected sections: 12 (stale processes caused device mismatch), 22 (10 eval processes on 4 GPUs, OOM kills), 23 (manual + orchestrator processes on same GPUs), 25 (pipe deadlock from shared orchestrator read loop)

The pattern: Multiple processes share GPUs, pipes, and files without explicit resource ownership or coordination. When one process dies, its resources (GPU memory, pipe buffers) are not reclaimed. When multiple processes write to the same file, writes race. When the orchestrator reads one process's pipe, others block.

The bug fixes (process scanning, pre-launch re-checks, atomic file writes, file-based output) are point fixes for specific sharing scenarios. They do not establish a resource ownership model.

Design principle: Resources (GPUs, file paths, output streams) must have explicit ownership. A process acquires a resource before using it and releases it on exit (including crash exit). The orchestrator is a resource manager, not just a process launcher. No two processes should ever write to the same file, share a pipe, or land on the same GPU without an explicit allocation.

GitHub issues: #26 (pipes), #30 (singleton). Resource ownership model not yet tracked -- needs a new issue.

---

**FM-3: Mid-experiment code changes create a mixed-codebase environment**

Affected sections: 22 (phased eval flags added mid-run), 23 (atomic writes and process checks added mid-run), 26 (import os added mid-run, not picked up by running process), 25 (C10 killed and relaunched with different code than orchestrator), 27 (user mandate: no mid-experiment patches)

The pattern: A bug is discovered mid-run. The fix is applied to the source file. New processes (relaunched by the orchestrator) pick up the fix. Old processes (still running) do not. The system is now in an inconsistent state where behavior depends on which version of the code a process loaded.

The bug fixes are themselves instances of the problem -- each was a mid-experiment patch. The phased eval strategy (Section 22), atomic writes (Section 23), and import os fix (Section 26) were all applied while eval processes were running.

Design principle: The codebase is frozen at experiment start. The git commit hash is recorded. No source files are modified during the experiment. If a bug is discovered, the experiment is stopped, the fix is applied, the full validation gate is re-run, and the experiment restarts. This is the same principle as "no schema migrations during a database transaction."

GitHub issue: #28

---

**FM-4: No validation gate between pipeline phases**

Affected sections: 1 (truncated HF token caused 381 warnings), 9 (missing zstandard killed all evals), 10 (relative import failed in script mode), 11 (past_key_value naming bug killed 12 configs), 12 (stale GPU processes caused device mismatch), 16 (remaining hardening checklist), 26 (missing import os crashed at save)

The pattern: Each phase of the pipeline (environment setup, training, evaluation, analysis, exfiltration) has implicit prerequisites that are not checked before the phase begins. A missing package, a stale process, a truncated token, or a naming bug is discovered only when the phase fails -- sometimes hours into a 12-hour run.

The bug fixes add individual pre-flight checks (verify HF token, install zstandard, clean GPUs, audit imports). But each check was added reactively, after the failure occurred. There is no comprehensive gate that validates the entire pipeline end-to-end before committing to a run.

Design principle: A single validation gate must exercise the full pipeline (training -> eval -> save -> analysis -> exfil) with minimal inputs (2 training steps, 2 eval samples per benchmark, 1 config) and verify that every phase produces correct output. This gate must pass before any real experiment run begins. No phase may start until the previous phase's outputs are verified.

GitHub issue: #27

---

**FM-5: External dependencies with no offline fallback or circuit breaker**

Affected sections: 1 (HF token 401 warnings during checkpoint save), 2 (WandB deleted run IDs caused silent fallback), 6 (HF cache on ephemeral filesystem wiped on pod recreation), 9 (HuggingFace 429 rate limits on parallel dataset downloads), 9 (proof-pile dataset removed from Hub), 10 (WandB import failure silently dropped results), 13 (no GitHub push mechanism), 24 (WandB "crashed" status misleading)

The pattern: The pipeline depends on external services (HuggingFace Hub, WandB, GitHub, dataset repositories) for credentials, model weights, datasets, metric logging, and result storage. When these services are unavailable, the pipeline either fails silently (WandB import swallowed, results dropped), fails destructively (429 rate limits kill eval processes), or fails confusingly (WandB "crashed" status misrepresents data integrity).

The bug fixes add retries, fallbacks, and pre-caching for specific dependencies. But there is no unified dependency management strategy: no circuit breaker to stop the experiment when a dependency is down, no offline mode that allows the experiment to continue without external services, no clear contract about which data lives where.

Design principle: Every external dependency must have a documented contract: what it provides, what happens when it's unavailable, and whether the experiment can proceed without it. Critical data (results, metrics) must have a local-first storage strategy where the on-disk file is the source of truth and external services (WandB, GitHub, HF) are secondary replicas. Non-critical dependencies (HF Hub for config.json during checkpoint save) must be eliminated (set HF_HUB_OFFLINE=1) or made resilient (circuit breaker, not retry loop).

GitHub issue: #29 (on-disk JSON as source of truth). Full dependency contract not yet tracked -- needs a new issue.

---

**FM-6: No crash recovery contract for the orchestrator**

Affected sections: 5 (orchestrator resilience: atomic state, checkpoint resume), 6 (pod eviction: no autostart, HF cache wiped), 22 (watchdog spawned duplicate orchestrator), 23 (watchdog restart re-queues running configs), 25 (orchestrator pipe deadlock), 27 (crash-safe resume requirement)

The pattern: The orchestrator is designed to survive process crashes (atomic state file, checkpoint resume). But it is not designed to survive infrastructure failures (pod eviction, power loss) or its own bugs (pipe deadlock, duplicate spawn). Each failure mode required a different recovery mechanism, added incrementally.

The bug fixes address specific recovery scenarios: watchdog for pod eviction, PID file for process tracking, .eval_phase for phase preservation, pgrep for duplicate detection. But there is no unified crash recovery contract: what state must survive, what state may be lost, how the system recovers, and how it verifies recovery is correct.

Design principle: The orchestrator must define a crash recovery contract:
1. What state is durable (orchestrator_state.json, result files, checkpoints, logs) -- must survive any crash.
2. What state is ephemeral (in-memory results, pipe buffers, GPU memory) -- may be lost, must be reconstructable from durable state.
3. How recovery works: on restart, read durable state, determine what was completed (by checking result files, not PID files), determine what was in-progress (by checking process table), resume from the last durable checkpoint.
4. How recovery is verified: after recovery, validate that completed work has valid result files and in-progress work is correctly resumed.

GitHub issues: #25 (durable writes), #30 (singleton). Crash recovery contract not yet tracked -- needs a new issue.

---

**FM-7: Provenance and reproducibility gaps**

Affected sections: 4 (100x loss gap: unknown if loss computation is correct), 7 (recovery scripts re-log data with different schemas), 14 (harmless warnings that could mask real issues), 21 (training time metrics fragmented across crashes), 24 (WandB status disagrees with on-disk truth), 26 (C10 results logged to WandB but not on disk)

The pattern: The experiment lacks a single source of truth for "what happened, when, and with what code." Training loss values may be artifacts of a computation bug. Recovery scripts re-log data with different schemas. WandB status disagrees with on-disk files. Metrics are fragmented across crash/restart cycles. When C10's results were lost, there was no way to determine from the system's state alone whether the results had ever been computed.

The bug fixes address individual provenance gaps (document on-disk as source of truth, add metrics file, audit loss path). But there is no unified provenance model: a record that ties each result to the code version, config, seed, GPU, timestamp, and computation that produced it.

Design principle: Every result file must carry provenance metadata: the git commit hash, config ID, seed, GPU ID, start/end timestamps, and a checksum. The experiment produces a manifest that lists every expected result file, its provenance, and its validation status. After the experiment, the manifest is the definitive record of what was computed, with what code, and whether it passed validation. This makes "did we compute C10's results?" a query, not an investigation.

GitHub issue: #29 (single source of truth). Provenance model not yet tracked -- needs a new issue.

### 28.3 Cross-Cutting Observations

**The N=1 run was a debugging session, not an experiment.** Sections 1-27 document 26 distinct bugs discovered during a single N=1 run. Each was fixed mid-experiment, creating a mixed-codebase environment where the system's behavior depended on when each process was launched. The fixes are valuable -- they identify real failure modes -- but the process of discovering and fixing them mid-run is itself a failure mode that must not recur.

**Recovery tooling is a code smell.** Sections 7, 24, and 26 describe recovery scripts (fix_wandb_phase1.py, c10_rescue.py, WandB reconciliation). As noted in Section 7: "Recovery scripts should not be needed. If they are, the experiment design has failed." The existence of recovery tooling indicates that the pipeline does not guarantee durability of its outputs. The goal is not better recovery -- it is eliminating the need for recovery.

**Checklists are necessary but insufficient.** Section 20 provides a 25-item pre-launch checklist. Every item is necessary. But checklists are point-in-time validations: they verify the system is correct at launch, not that it stays correct during the run. The design principles above (frozen codebase, incremental writes, crash recovery contract, provenance) are invariants that hold for the duration of the experiment, not just at launch.

**WandB is not a backup.** Multiple sections treat WandB as a data store (logging results, recovering via API, checking completion status). But WandB is a monitoring and visualization tool. Its data model does not match the experiment's result schema, its status field is a process-lifecycle signal not a data-integrity signal, and its API is not designed for reliable bulk data recovery. On-disk JSON with fsync is the source of truth. WandB is a display layer.

### 28.4 Summary: Design Principles for N=30

| # | Principle | Failure Modes Addressed | GitHub Issues |
|---|-----------|------------------------|---------------|
| P1 | Durability at computation granularity | FM-1 | #25 |
| P2 | Explicit resource ownership | FM-2 | #26, #30, (new) |
| P3 | Frozen codebase, no mid-run patches | FM-3 | #28 |
| P4 | Full-pipeline validation gate | FM-4 | #27 |
| P5 | Local-first storage, external services as replicas | FM-5 | #29, (new) |
| P6 | Crash recovery contract | FM-6 | #25, #30, (new) |
| P7 | Provenance for every result | FM-7 | #29, (new) |

Three new issues are needed for failure modes not yet tracked:
- FM-2: Resource ownership model (GPUs, files, pipes)
- FM-5: External dependency contract (circuit breakers, offline mode)
- FM-6: Crash recovery contract (durable vs ephemeral state, verified resume)
- FM-7: Provenance model (git hash, checksums, manifest)

These will be created as GitHub issues for proper engineering follow-up.