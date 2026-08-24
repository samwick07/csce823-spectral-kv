# N=30 Experiment: Lessons Learned & Hardening Plan

**Source:** N=1 pilot run (C00, C01, C07) — August 22-24, 2026
**Goal:** A fully autonomous N=30 experiment with zero runtime code changes, patches, or mid-run script adjustments.

---

## 1. Pre-Launch Checklist

### 1.1 Credential Validation
- [ ] **Verify HF_TOKEN is full-length (37+ chars), not truncated.** The N=1 run had `hf_qFR...LjlE` (13 chars with literal dots) saved in `.env.spectral`, causing 38 PEFT 401 warnings at every checkpoint save.
- [ ] **Verify HF_TOKEN has gated repo access.** Run `huggingface_hub.HfApi().whoami()` and attempt `hf_hub_download("meta-llama/Llama-3.1-8B-Instruct", "config.json")` before launching.
- [ ] **Verify WANDB_API_KEY is valid.** Run `wandb.Api().whoami()` before launching.
- [ ] **Add a pre-flight check script** (`scripts/preflight.sh`) that validates all credentials and fails fast with a clear message if any are missing/invalid.

### 1.2 Environment Hardening
- [ ] **Set WANDB_PROJECT in .env.spectral.** The N=1 run had WANDB_PROJECT missing, causing the HF Trainer's WandbCallback to default to `project="huggingface"`, sending 3 runs to the wrong project.
- [ ] **Set HF_HUB_OFFLINE=1 after model cache is populated.** PEFT's checkpoint save tries to fetch `config.json` from HuggingFace Hub even when the model is fully cached. Setting `HF_HUB_OFFLINE=1` eliminates these 401 warnings entirely once the model is in local cache.
- [ ] **Validate DeepSpeed config matches YAML.** The N=1 run had `warmup_max_lr=2e-05` in DeepSpeed config vs `learning_rate=1e-05` in YAML, causing all 13 configs to crash on Aug 22.

### 1.3 Data Verification
- [ ] **Pre-download and verify all datasets** (RedPajama-Data-1T streaming, LongAlpaca-12k, PG-19, Proof-pile, LongBench V1).
- [ ] **Verify LongAlpaca column names** (instruction/input/output/file). A column mismatch was fixed in commit 7a7ecf3.
- [ ] **Verify LongBench org name** is THUDM, not THUIAR.
- [ ] **Run dataset verification** per `docs/dataset_verification.md`.

---

## 2. WandB Integration Hardening

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

### 3.3 Dataset Issues
- **RedPajama-Data-1T-Sample removed from HuggingFace.** The training code falls back to streaming the full `RedPajama-Data-1T` dataset. This works but requires network access during Phase 1.
- **LongAlpaca column mismatch (FIXED in 7a7ecf3).** The dataset uses `instruction/input/output/file` columns; the training code was updated to match.

---

## 4. Orchestrator Resilience

### 4.1 Current State
- Atomic `orchestrator_state.json` persists progress across restarts.
- DeepSpeed checkpoint resume every 500 steps (Phase 1) / 100 steps (Phase 2).
- `WANDB_MODE=offline` fallback for VPN drops.
- `scripts/relaunch.sh` is idempotent and auto-restores HF cache.

### 4.2 N=30 Hardening Needed
- [ ] **Health check daemon:** A background process that monitors training progress, GPU utilization, and W&B sync status. Alerts (not stops) if something goes wrong.
- [ ] **Automatic W&B sync recovery:** If `WANDB_MODE` falls back to offline, a watcher should `wandb sync` local runs when connectivity returns.
- [ ] **Config validation before launch:** A script that validates all 13 YAML configs, DeepSpeed configs, and dataset availability before the orchestrator starts.
- [ ] **Per-config isolated W&B runs:** Ensure each `(config, phase, seed)` combination gets a unique, non-colliding W&B run ID. The current deterministic ID scheme works for N=1 but may collide with seeds in N=30.
- [ ] **Run name convention for N=30:** Define and implement a naming scheme like `C{XX} Phase {N} {dataset} (seed {S})` in the code, not post-hoc.

---

## 5. HF Token Security
- [ ] **Never redact tokens in env files.** The N=1 run had the HF_TOKEN literally replaced with `hf_qFR...LjlE` (the redacted form became the actual value). Add a validation check that the token is at least 35 characters.
- [ ] **Store tokens only in `/workspaces/.env.spectral`** (PVC, outside git). Never in code, scripts, or git-tracked files.
- [ ] **Pre-flight token validation:** Verify `whoami()` succeeds and the account has access to `meta-llama/Llama-3.1-8B-Instruct` before launching.

---

## 6. Recovery Scripts (If Needed)

The following recovery scripts exist in `scripts/` and were tested during the N=1 run:
- `fix_wandb_phase1.py` — Re-logs C00 Phase 1 from `c00_phase1_clean_metrics.txt`
- `fix_wandb_phase2.py` — Re-logs C00 Phase 2 from `output.log` + orchestrator log
- `fix_wandb_C01.py` — Re-logs C01 Phase 1+2 from orchestrator log

**N=30 principle:** Recovery scripts should not be needed. If they are, the experiment design has failed. The goal is to prevent the conditions that require recovery, not to improve recovery tooling.

---

## 7. N=30 Scale Considerations

### 7.1 Compute Budget
- 13 configs x 30 seeds = 390 training runs (780 phases)
- Estimated 1,216-1,340 GPU-hours on 8x H200 (~7 days wall-clock)
- At 4x H200 (current allocation): ~14 days

### 7.2 W&B Free Tier Limits
- Free tier: 100 runs per project. With 780 phases, you'll need multiple projects or a paid tier.
- **Alternative:** Use `WANDB_MODE=offline` for all runs, then sync in batches after the experiment. This eliminates sync issues during training entirely.
- **Alternative:** Split into per-config or per-seed W&B projects (e.g., `csce823-spectral-kv-C00`).

### 7.3 Checkpoint Storage
- Each Phase 1 checkpoint: ~50MB (LoRA adapters only)
- Each Phase 2 checkpoint: ~50MB (LoRA adapters only)
- 780 phases x 2 checkpoints avg = ~78GB total
- Verify PVC has sufficient storage before launch.

### 7.4 Run ID Uniqueness for N=30
- Current scheme: `C{XX}_phase{N}_{dataset}` — no seed component.
- **Must add seed:** `C{XX}_phase{N}_{dataset}_s{S}` to avoid collisions across seeds.
- Update `new_run_id()` in `src/utils/wandb_utils.py` before N=30 launch.

---

## 8. Summary: What Must Be True Before N=30 Launch

1. All credentials validated (HF token full-length + gated access, WandB key valid)
2. `WANDB_PROJECT` set in environment
3. `HF_HUB_OFFLINE=1` set after model cache populated
4. All datasets pre-downloaded and verified
5. All 13 YAML configs validated (LR matches DeepSpeed, batch sizes match)
6. `new_run_id()` includes seed component
7. Run display names set in code (not post-hoc)
8. Pre-flight check script passes (`scripts/preflight.sh`)
9. Health check daemon configured (read-only monitoring, alert-only)
10. W&B project strategy decided (free tier limits, offline mode, or paid tier)
11. Checkpoint storage capacity verified
12. No W&B runs deleted from the project (use tags instead)
13. Loss computation path audited and verified comparable between spectral and baseline attention (Section 9)
14. Per-run log files implemented: each config/phase/seed writes to an isolated log file (Section 10)

---

## 9. Loss Computation Verification

### 9.1 The 100x Loss Gap (OBSERVED in N=1)

**Issue:** During the N=1 run, compressed configs (C01, C07, C10) converged to a Phase 1 training loss of ~0.02, while the baseline (C00) maintained a flat loss of ~2.09. In Phase 2, compressed configs reached ~0.003-0.007 while the baseline sat at ~0.52. This 100x gap is abnormally large and could indicate either a genuine effect of spectral compression on the optimization landscape or a discrepancy in how loss is computed for the custom spectral attention forward pass.

**Root cause unknown -- must be verified before N=30.** The spectral attention forward pass overrides `LlamaAttention.forward` with a manual SDPA implementation. If the loss computation path differs from standard causal LM loss (e.g., different masking, different normalization, or the compressed/reconstructed K-V tensors affect the logits in a way that inflates or deflates the loss), then training loss comparisons between compressed and baseline configs are not apples-to-apples.

**N=30 actions:**
- [ ] **Audit the loss computation path in the spectral attention forward pass.** Trace from `SpectralKVAttention.forward()` through the model's `forward()` to the loss function. Verify that the loss function receives the same type of logits, applies the same label masking, and uses the same reduction as the baseline path.
- [ ] **Verify loss is computed on reconstructed (full) logits, not compressed intermediates.** The K/V tensors are compressed and reconstructed, but the output logits fed to the loss function must be full-dimensional. If loss is accidentally computed on compressed representations, it would appear artificially low.
- [ ] **Add a unit test that compares loss output for identical inputs** between `LlamaAttention` (baseline) and `SpectralKVAttention` (gamma=1.0, no compression). With gamma=1.0, the spectral transform should be a no-op and the losses should be numerically identical (within floating point tolerance). If they differ, the forward pass has a bug.
- [ ] **Log the loss computation path explicitly.** Add a one-time log line at training start that prints whether the model is using standard or spectral attention and which loss function is active. This makes it unambiguous in the logs which path produced the loss values.
- [ ] **Do not cite the 100x loss gap as a result until verified.** If the loss computation is confirmed correct and the gap persists, it is a genuine finding. If it is a bug, the corrected loss values may change the relative ranking of configs.

---

## 10. Logging Architecture

### 10.1 Monolithic Orchestrator Log (OBSERVED in N=1)

**Issue:** The N=1 run wrote all config training (Phase 1 + Phase 2), all evaluation, and all statistics output to a single orchestrator log file (`logs/orchestrator_20260823_041331.log`). This file reached 788 KB and contained interleaved output from C01, C07, and C10 -- making it difficult to extract per-config loss curves, timing, or error messages without complex grep patterns. When C00 crashed and restarted 19 times, each restart created a new orchestrator log, but all phases of the resumed config were still interleaved within each log.

**Impact on N=1 analysis:** Extracting per-config training metrics required grepping across 20+ log files and manually correlating timestamps. The C00 baseline data was particularly noisy because 19 crash/restart cycles fragmented its loss trajectory across multiple logs and WandB runs.

### 10.2 Required Logging Structure for N=30

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
