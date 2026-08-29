# FreqKV Causal Rewrite vs. N30 Lessons Learned: Cross-Comparison Checklist

**Date:** August 29, 2026
**Branch:** `feat/freqkv-causal-rewrite`
**Purpose:** Map every action item in `N30_LESSONS_LEARNED.md` to the rewrite's scope, marking items that were accomplished, superseded, or remain open.

---

## How to Read This Checklist

- [x] **DONE** — The rewrite accomplished this item
- [~] **SUPERSEDED** — The rewrite made this item moot (e.g., removed the class the item referenced)
- [ ] **OPEN** — Not addressed by this rewrite; remains for N=30 launch

---

## PART I: TRAINING

### Section 1: Training Pre-Launch Checklist

#### 1.1 Credential Validation
- [ ] Verify HF_TOKEN is full-length (37+ chars), not truncated
- [ ] Verify HF_TOKEN has gated repo access
- [ ] Verify WANDB_API_KEY is valid
- [ ] Add a pre-flight check script (`scripts/preflight.sh`)

#### 1.2 Environment Hardening
- [ ] Set WANDB_PROJECT in .env.spectral
- [ ] Set HF_HUB_OFFLINE=1 after model cache is populated
- [ ] Validate DeepSpeed config matches YAML

#### 1.3 Training Data Verification
- [ ] Pre-download and verify all training datasets
- [ ] Verify LongAlpaca column names
- [ ] Run dataset verification per docs/dataset_verification.md

### Section 2: WandB Integration Hardening (all previously fixed)
- [~] 2.1 Deleted Run ID Problem — previously fixed (commit 9586a28)
- [~] 2.2 Project Routing — previously fixed (commit 9586a28)
- [~] 2.3 Metric Key Consistency — previously fixed (commit a8e900e)
- [~] 2.4 Phase Merging — previously fixed (commit 9586a28)
- [ ] 2.5 Run Naming — still open for N=30

### Section 3: Training Stability (previously fixed)
- [~] 3.1 DeepSpeed Configuration — warnings documented, not code changes needed
- [~] 3.2 Checkpoint Resume — lexicographic sort bug fixed
- [~] 3.3 Training Dataset Issues — RedPajama fallback works, LongAlpaca columns fixed

### Section 4: Loss Computation Verification

#### 4.1 The 100x Loss Gap — ROOT CAUSE FOUND AND FIXED

**Root cause:** Causality violation in spectral KV-cache compression. The DCT/FFT
transform operated on the full sequence dimension, allowing future V information
to leak into past attention outputs via V_recon[j<=t]. This is NOT a loss
computation bug — HF Trainer computes standard cross-entropy on full-dim logits.
The defect is a forward-pass future→past information leak.

**Fix:** FreqKV-aligned chunk-wise compression (branch `feat/freqkv-causal-rewrite`).
Compresses only PAST tokens per chunk, eliminating cross-chunk leakage entirely.
Within-chunk leakage is bounded by chunk_size and eliminated by the causal mask.

**Verification:** 65 tests passing, including smoke test showing compressed loss
ratio = 1.00x baseline (was 0.01x with old code).

- [x] **Audit the loss computation path in the spectral attention forward pass.**
      DONE. The forward pass was completely replaced with `_freqkv_forward_noiterate`
      and `_freqkv_forward_iterate`. The loss path is standard HF Trainer CE on
      full-dim logits — no compression intermediates reach the loss function.

- [x] **Verify loss is computed on reconstructed (full) logits, not compressed
       intermediates.** DONE. The new forward produces full-dimensional output
      `[B, S, num_heads * head_dim]` via `o_proj`. No compressed representation
      reaches the LM head or loss function. Verified by `test_smoke_tiny_model.py`.

- [x] **Add a unit test that compares loss output for identical inputs between
       baseline and compressed (gamma=1.0).** DONE. `test_smoke_tiny_model.py`
      builds identical-weight models with baseline and compressed attention,
      runs forward passes, and compares losses. Additionally tests gamma=0.5,
      0.22, and 0.01 across all 4 config variants (DCT/FFT × fixed/learnable).

- [x] **Log the loss computation path explicitly.** DONE. `apply_spectral_compression()`
      logs: transform type, filter type, gamma, cache_size, sink/recent size,
      and is_iterate mode at model setup time.

- [x] **Do not cite the 100x loss gap as a result until verified.** RESOLVED.
      Root cause identified (causality violation), fix implemented and verified.
      The 100x gap was an artifact of future information leakage. The corrected
      code produces compressed loss ~1.00x baseline (slightly higher due to
      reconstruction error, as expected). N=1 training loss values (~0.02) must
      NOT be cited; eval results remain valid as comparative metrics.

### Section 5: Orchestrator Resilience
- [ ] 5.2 Health check daemon
- [ ] 5.2 Automatic W&B sync recovery
- [ ] 5.2 Config validation before launch
- [ ] 5.2 Per-config isolated W&B runs (seed component in run ID)
- [ ] 5.2 Run name convention for N=30

### Section 6: Pod Eviction (previously fixed)
- [~] 6.3 Fix 1: HF cache moved to persistent PVC — done
- [~] 6.3 Fix 2: Watchdog auto-restart mechanism — done
- [ ] 6.5 Set HF_HOME globally in Coder workspace template
- [ ] 6.5 Add disk pressure pre-flight check
- [ ] 6.5 Kubernetes liveness/readiness probe
- [ ] 6.5 Set HF_HUB_OFFLINE=1 after model load
- [ ] 6.5 Monitor PVC disk usage
- [ ] 6.5 Document save_total_limit interaction

### Section 7: Recovery Scripts
- [~] Recovery scripts exist but should not be needed for N=30

---

## PART II: EVALUATION

### Section 8: Evaluation Pre-Launch Checklist

#### 8.1 Eval Dependencies
- [ ] Install zstandard in venv and add to workspace template
- [ ] Verify LongBench org name is THUDM

#### 8.2 Eval Dataset Pre-Caching
- [ ] Pre-download ALL eval datasets before launching eval
- [ ] Add pre-eval dataset check to orchestrator

#### 8.3 GPU Hygiene
- [ ] Check for stale GPU processes before launching eval

### Section 9: Proof-pile Dataset Crash (previously fixed)
- [~] zstandard installed, trust_remote_code=True, retry logic added, datasets pre-cached

### Section 10: WandB Import Failure in Eval Mode (previously fixed)
- [~] try/except ImportError fallback added
- [ ] Audit all relative imports in src/

### Section 11: past_key_value Naming Bug

**Previously fixed (31 replacements across 5 files).**

This rewrite goes further: `SpectralDynamicCache` is REMOVED entirely. Standard
HF `DynamicCache` is used for all generation. The `past_key_value` vs
`past_key_values` naming issue cannot recur because no custom cache class
is passed to `model.generate()`.

- [~] **Pin the transformers version in requirements.txt.** Still recommended,
      but the specific naming bug is moot — standard DynamicCache uses the
      correct API.
- [~] **Add a unit test that calls model.generate() with SpectralDynamicCache.**
      SUPERSEDED. SpectralDynamicCache no longer exists. Standard DynamicCache
      is tested by HuggingFace's own test suite.

### Section 12: Device Mismatch (previously fixed)
- [~] Stale process cleanup — still recommended
- [~] `model.to(target_device)` after PeftModel loading — still applies (the
      new `SpectralKVCompressor` is registered via `attn.add_module("spectral_cache", ...)`,
      same as before, so the device movement fix in `run_experiment.py` still works)
- [ ] Add stale-process cleanup to orchestrator's eval phase
- [ ] Stagger parallel eval launches by 30s

### Section 13: No GitHub Results Push (previously fixed)
- [~] scripts/push_results_github.sh created, Phase 4b added to orchestrator

### Section 14: Harmless Generation Warnings
- [~] Not a bug, no action needed

### Section 15: Eval Files Changed (historical record)
- [~] Documented

### Section 16: Remaining Eval Hardening for N=30
- [ ] Add zstandard to Coder workspace template
- [ ] Pre-download ALL eval datasets (LongBench V1 still at eval time)
- [ ] Add pre-eval dataset check to orchestrator
- [ ] Audit all relative imports in src/
- [ ] Stagger parallel eval launches by 30s
- [ ] Add eval result verification to orchestrator
- [ ] Pin transformers version in requirements.txt
- [~] **Add a unit test that calls model.generate() with SpectralDynamicCache.**
      SUPERSEDED — SpectralDynamicCache removed, standard DynamicCache used.
- [ ] Add stale-process cleanup to orchestrator's eval phase

---

## PART III: CROSS-CUTTING CONCERNS

### Section 17: HF Token Security
- [ ] Never redact tokens in env files
- [ ] Store tokens only in /workspaces/.env.spectral
- [ ] Pre-flight token validation

### Section 18: N=30 Scale Considerations
- [ ] Compute budget planning (1,216-1,340 GPU-hours)
- [ ] W&B free tier limits (100 runs/project)
- [ ] Checkpoint storage capacity (78GB)
- [ ] Run ID uniqueness (add seed component)

### Section 19: Logging Architecture
- [ ] Implement per-run log file naming
- [ ] Modify orchestrator to redirect subprocess output to per-run files
- [ ] Keep orchestrator log as high-level summary only
- [ ] Add [RESUME] marker on crash recovery
- [ ] Do not create new log file for resumed run
- [ ] Add log index to orchestrator state
- [ ] Test logging structure with pilot run

### Section 20: Summary — What Must Be True Before N=30 Launch

1. [ ] All credentials validated
2. [ ] WANDB_PROJECT set in environment
3. [ ] HF_HUB_OFFLINE=1 set after model cache populated
4. [ ] All training datasets pre-downloaded and verified
5. [ ] All 13 YAML configs validated (LR matches DeepSpeed, batch sizes match)
6. [ ] new_run_id() includes seed component
7. [ ] Run display names set in code
8. [ ] Pre-flight check script passes
9. [ ] Health check daemon configured
10. [ ] W&B project strategy decided
11. [ ] Checkpoint storage capacity verified
12. [ ] No W&B runs deleted from the project
13. [x] **Loss computation path audited and verified comparable between spectral
        and baseline attention (Section 4).** DONE. Root cause identified
        (causality violation), fix implemented (FreqKV chunk-wise compression),
        verified (65 tests, smoke test ratio 1.00x). The old _spectral_forward
        is replaced; the new forward produces standard full-dim logits through
        standard CE loss.
14. [ ] Per-run log files implemented
15. [ ] zstandard installed in venv and workspace template
16. [ ] All eval datasets pre-cached
17. [ ] Pre-eval dataset check passes in orchestrator
18. [ ] All relative imports in src/ audited with fallbacks
19. [~] **past_key_values (plural) used consistently.** SUPERSEDED —
        SpectralDynamicCache removed, standard DynamicCache used.
20. [ ] Transformers version pinned in requirements.txt
21. [ ] Stale GPU process cleanup runs before eval launch
22. [ ] Eval launches staggered by 30s
23. [ ] Eval result verification in orchestrator
24. [ ] GitHub push script wired into orchestrator post-exfil
25. [ ] Incremental metrics capture implemented (Section 21)

### Section 21: Resilient Metrics Capture
- [ ] Add MetricsCollector class to src/utils/metrics_collector.py
- [ ] Call MetricsCollector from training scripts
- [ ] Call MetricsCollector from run_experiment.py before/after each benchmark
- [ ] Record GPU memory using torch.cuda.max_memory_allocated()
- [~] **Record actual cache size from SpectralDynamicCache.get_compression_ratio().**
      SUPERSEDED — SpectralDynamicCache removed. The replacement
      `SpectralKVCompressor` has `get_compression_ratio()` and
      `get_compression_stats()`, but the measurement point (during efficiency
      benchmark) needs to be updated to use the new API.
- [ ] Handle interruption gracefully (try/finally, SIGTERM)
- [ ] Add metrics file to orchestrator state
- [ ] Add metrics aggregation script
- [ ] Test incremental write with deliberate kill

---

## Summary

| Category | Total Items | Done by Rewrite | Superseded | Open |
|----------|------------|-----------------|------------|------|
| Section 4 (Loss Computation) | 5 | 5 | 0 | 0 |
| Section 11 (past_key_value) | 2 | 0 | 2 | 0 |
| Section 16 (Eval Hardening) | 9 | 0 | 1 | 8 |
| Section 20 (N=30 Summary) | 25 | 1 | 1 | 23 |
| Section 21 (Metrics Capture) | 8 | 0 | 1 | 7 |
| All other sections | 34 | 0 | 14 | 20 |
| **TOTAL** | **83** | **6** | **19** | **58** |

**Items accomplished by this rewrite (6):**
1. Section 4.1: Audit the loss computation path — root cause found, forward pass replaced
2. Section 4.1: Verify loss on full logits, not compressed intermediates — verified
3. Section 4.1: Unit test comparing baseline vs compressed loss — 6 parametrized tests
4. Section 4.1: Log the loss computation path — apply_spectral_compression logs config
5. Section 4.1: Do not cite 100x gap until verified — resolved, gap was causality artifact
6. Section 20.13: Loss computation path audited and verified

**Items superseded by this rewrite (3 key):**
1. Section 11: past_key_value naming bug — SpectralDynamicCache removed entirely
2. Section 16: Unit test for model.generate() with SpectralDynamicCache — class removed
3. Section 21: Record cache size from SpectralDynamicCache — replaced by SpectralKVCompressor API

**Additional improvements not in the lessons doc:**
- RoPE inconsistency fix (compress before RoPE in both paths) — documented for publication
- FreqKV mathematical equivalence verified (DCT/IDCT/compress match reference exactly)
- Causality test suite (15 tests) that would have caught the bug pre-launch
- Eval perplexity now uses seq_len-sized chunks (FreqKV protocol) — compression active during eval
- All 13 YAMLs updated with sink_size, recent_size, cache_size, use_flash_attn fields
