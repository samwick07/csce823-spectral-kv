# P1 & P2 Re-Examination (Post-P0)

Date: 2026-08-18
Commit: 166aac8 (P0 complete)
Codebase: 27 Python files, 3,885 lines
Repo: https://github.com/samwick07/csce823-spectral-kv

## P0 Summary (Completed)

All 5 P0 blockers resolved:

1. **Spectral compression integrated into LlamaAttention forward** —
   `src/spectral/attention.py` monkey-patches each layer's forward to
   insert compress/reconstruct between K/V projection and SDPA. Handles
   GQA (8 KV heads → 32 query heads) and both old/new transformers rotary
   embedding APIs.

2. **Spectral-domain cache with true O(γN) storage** —
   `src/spectral/cache.py` (new). `SpectralKVCache` stores compressed
   spectral coefficients, reconstructs on demand. Transform → filter →
   truncate → store; inverse-transform for attention computation.

3. **FlashAttention-2 added to model loading** — all 3 model-loading
   paths (`train_redpajama.py`, `train_longalpaca.py`, `run_experiment.py`)
   use `attn_implementation="flash_attention_2"`.

4. **Switched to Llama-3.1-8B-Instruct** — `config.py` default + all 13
   YAML configs. Same architecture as Llama-3-8B, native 128K context.

5. **Model name from config** — all training/eval scripts accept
   `model_name` as a parameter instead of hardcoding.

Additional P0 work:
- `src/spectral/transform.py` — added `spectral_len()`, `pad_to_len()`
  for correct spectral-domain sizing. DCT inverse zero-pads truncated
  coefficients.
- `src/spectral/filter.py` — renamed `max_seq_len` → `max_spectral_len`
  (DCT=N, FFT=N//2+1) for correct per-transform parameter sizing.
- `src/training/lora_config.py` — `modules_to_save=['spectral_cache']`
  preserves learnable filter params during LoRA training.
- `src/utils/wandb_utils.py` (new) — W&B integration: `init_wandb`,
  `log_spectral_stats`, `log_eval_results`, `create_report`.
- `src/run_experiment.py` — full W&B integration + spectral cache lifecycle.
- Eval scripts (`pg19.py`, `proof_pile.py`, `longbench.py`, `metrics.py`)
  — `model_name` parameter, `reset_all_caches()` between sequences.
- Data collators (`DataCollatorForLanguageModeling`) added to both
  training scripts.

---

## P1 Items — Status After P0

### P1-1: Checkpoint resume (RedPajama → LongAlpaca)
**STATUS: ADDRESSED in P0.**

`train_longalpaca.py` loads Phase 1 checkpoint via
`PeftModel.from_pretrained(model, phase1_checkpoint)`. The
`spectral_cache` module is preserved via `modules_to_save` in LoRA
config. `trainer.train()` no longer passes `resume_from_checkpoint` —
Phase 2 starts fresh with the Phase 1 adapter loaded.

**Remaining concern:** Phase 2 needs to load BOTH the LoRA adapter AND
the `spectral_cache` module from Phase 1. `PeftModel.from_pretrained`
with `modules_to_save` should handle this, but needs runtime verification.
→ Added as P2-6 smoke-test item.

### P1-2: Data collator for training
**STATUS: ADDRESSED in P0.**

Both `train_redpajama.py` and `train_longalpaca.py` use
`DataCollatorForLanguageModeling(tokenizer, mlm=False)`.

### P1-3: Verify dataset names on HuggingFace
**STATUS: STILL NEEDED.** Cannot verify from container (no HF dataset
browsing). Needs to be done on the Coder workspace.

Datasets to verify:
- `Yukang/LongAlpaca-16k` — likely correct but confirm exact HF ID
  (may be `Yukang/LongAlpaca-16k-length` or just `Yukang/LongAlpaca-12k`)
- `EleutherAI/proof-pile` — confirm test split exists
- `THUDM/LongBench` — confirm task names match `LONGENCH_TASKS` dict
- `togethercomputer/RedPajama-Data-1T-Sample` — should be fine

Verify with:
```python
from datasets import load_dataset
ds = load_dataset("Yukang/LongAlpaca-16k", split="train")
print(ds.column_names, len(ds))
```

### P1-4: Implement proper ART ANOVA
**STATUS: STILL NEEDED.** Priority: HIGH.

Location: `src/stats/analyze.py`, function `_art_anova` (line 373).

Current code uses Kruskal-Wallis per factor separately. This CANNOT
detect interaction effects — the core research question (does the
benefit of learnable filtering depend on transform type?).

True ART ANOVA requires:
1. Compute aligned observations: remove the effect of other factors
   (subtract cell mean for the other factor, add grand mean)
2. Rank the aligned observations
3. Run standard ANOVA on the ranks

Options:
- Implement ART manually (rank → align → rank → ANOVA)
- Use `py-art` library if available
- Use R's `ARTool` package via rpy2

The current Kruskal-Wallis approach should be kept as a fallback /
sanity check, but the primary analysis must use true ART.

### P1-5: LongBench needs trust_remote_code
**STATUS: NEW ISSUE.**

`load_dataset("THUDM/LongBench", task_name)` likely needs
`trust_remote_code=True` for the loading script. This needs to be added
to the `load_dataset` call in `src/eval/longbench.py`.

Similarly check if `EleutherAI/proof-pile` needs it.

---

## P2 Items — Status After P0

### P2-1: Unit tests for spectral transforms
**STATUS: STILL NEEDED.**

Tests to write:
- DCT round-trip reconstruction error (forward → inverse ≈ identity)
- FFT round-trip reconstruction error
- Truncation preserves expected fraction (γ·N coefficients)
- Learnable filter gradient flow (backward pass updates filter_logits)
- FixedLowPassFilter output matches expected low-pass behavior
- SpectralKVCache compress → reconstruct shape correctness
- GQA repeat_kv on reconstructed tensors

Suggested location: `tests/test_spectral_transforms.py`

### P2-2: Smoke test script
**STATUS: STILL NEEDED. Priority: HIGH (before any GPU run).**

The attention patching uses monkey-patching on `LlamaAttention.forward`.
Must verify at runtime:
- Wrapped forward receives correct kwargs from the model
- `position_embeddings` correctly handled (transformers ≥4.46 passes
  as kwargs, <4.46 expects `rotary_emb` on the module)
- Gradient flows through the learnable filter
- `repeat_kv` works correctly with reconstructed tensors
- Loss computation succeeds with compressed attention output
- `reset_all_caches()` clears state properly between sequences
- PEFT `modules_to_save` correctly saves/loads `spectral_cache`

Suggested location: `scripts/smoke_test.sh` + `src/tests/smoke_test.py`

Smoke test flow:
1. Load a small model (e.g., `meta-llama/Llama-3.2-1B-Instruct` for speed)
2. Apply spectral compression (FFT + learnable, γ=0.22)
3. Run 10 training steps on dummy data
4. Verify loss decreases or at least doesn't NaN
5. Verify learnable filter params have gradients
6. Save checkpoint, reload, verify spectral_cache is restored
7. Run 1-sample PG-19 perplexity eval
8. Run 1-sample LongBench eval

### P2-3: W&B integration
**STATUS: DONE in P0.**

`src/utils/wandb_utils.py` created with:
- `init_wandb(config, phase)` — initializes run with full experiment config
- `log_spectral_stats(model)` — logs learnable filter masks as images
- `log_eval_results(results, config_id, seed)` — logs metrics
- `finish_wandb(summary)` — finalizes run
- `create_report(config_ids)` — generates W&B report for sharing

Integrated into `run_experiment.py` for both train and eval phases.

### P2-4: FA2 compatibility with modified KV cache
**STATUS: NEW ISSUE. Priority: MEDIUM.**

The spectral forward (`_spectral_forward` in attention.py) uses manual
SDPA (`torch.matmul` + `F.softmax`) instead of
`F.scaled_dot_product_attention`. This means FlashAttention-2 is NOT
used for the compressed attention computation — only for the Q/K/V
projections (which don't need it since they're just linear layers).

The manual implementation is necessary because FA2 cannot handle
variable-length KV tensors (the reconstructed length may differ from
the input length). This is expected and matches FreqKV's approach.

However, the model loading requests
`attn_implementation="flash_attention_2"` which may cause confusion:
- transformers may set internal flags expecting SDPA-based attention
- But since we override `forward` entirely, the internal attention
  implementation flag is irrelevant

**Recommendation:** Switch to `attn_implementation="eager"` for clarity,
since we override the forward anyway. Add a comment explaining why FA2
is not used for the compressed attention path. This avoids confusion
and prevents potential transformers version-specific issues.

Files to update:
- `src/training/train_redpajama.py` (line with `attn_implementation`)
- `src/training/train_longalpaca.py` (line with `attn_implementation`)
- `src/run_experiment.py` (line with `attn_implementation`)

### P2-5: trust_remote_code for LongBench
**STATUS: Same as P1-5.**

### P2-6: Verify checkpoint loading at runtime
**STATUS: NEW ITEM.**

Part of the smoke test (P2-2). Specifically verify:
- Phase 1 LoRA adapter saves `spectral_cache` module
- Phase 2 loads Phase 1 checkpoint and `spectral_cache` is restored
- Learnable filter parameters persist across phases
- `modules_to_save` mechanism works as expected with PEFT

---

## Recommended Implementation Order

1. **P1-4: ART ANOVA** — academically critical for interaction effects
2. **P2-2: Smoke test script** — catches integration bugs before GPU runs
3. **P1-3/P1-5: Dataset verification + trust_remote_code** — quick, on Coder workspace
4. **P2-4: FA2 → eager attention** — clarity fix, prevents version issues
5. **P2-1: Unit tests** — verify spectral transform correctness
6. **P2-6: Checkpoint verification** — fold into smoke test

## Key Files Reference

| File | Role |
|------|------|
| `src/spectral/transform.py` | DCT/FFT transforms with truncation |
| `src/spectral/filter.py` | Fixed low-pass + learnable sigmoid mask |
| `src/spectral/cache.py` | SpectralKVCache: compress, store, reconstruct |
| `src/spectral/attention.py` | LlamaAttention forward override + apply function |
| `src/training/lora_config.py` | LoRA config with modules_to_save |
| `src/training/train_redpajama.py` | Phase 1: RedPajama CPT |
| `src/training/train_longalpaca.py` | Phase 2: LongAlpaca SFT |
| `src/eval/pg19.py` | PG-19 sliding-window perplexity |
| `src/eval/proof_pile.py` | Proof-pile perplexity |
| `src/eval/longbench.py` | LongBench V1 generation tasks |
| `src/eval/metrics.py` | Sliding-window PPL + efficiency metrics |
| `src/stats/analyze.py` | 7-step statistical pipeline (ART needs fix) |
| `src/utils/wandb_utils.py` | W&B integration |
| `src/utils/config.py` | YAML config loading |
| `src/run_experiment.py` | Main orchestration |
| `configs/experiment_C00.yaml` – `C12.yaml` | 13 experiment configs |
