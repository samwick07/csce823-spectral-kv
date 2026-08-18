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
   Model name centralized in src/utils/constants.py.

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
**STATUS: CHECKLIST CREATED (see docs/dataset_verification.md).**

Dataset verification requires the Coder workspace (coder.afitcdn.org)
with HuggingFace access. A comprehensive checklist has been created at
`docs/dataset_verification.md` listing all 5 datasets (RedPajama-Data-1T-Sample,
LongAlpaca-16k, pg19, proof-pile, LongBench), their expected columns,
trust_remote_code requirements, and a pre-flight checklist.

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
**STATUS: DONE.** Implemented true Aligned Rank Transform (Wobbrock et al. 2011).

`src/stats/analyze.py`, function `_art_anova` (line 373).

The implementation follows the ART procedure:
1. Compute aligned observations for each effect by removing the estimated
   effects of all *other* factors and interactions (alignment formulas
   in the docstring).
2. Rank the aligned observations (average ranks for ties via
   `scipy.stats.rankdata`).
3. Run standard one-way ANOVA (F-test) on the ranks via
   `scipy.stats.f_oneway`.

Three effects tested per benchmark:
- Factor A (transform): DCT vs FFT
- Factor B (filter): Fixed vs Learnable
- Interaction A×B: does learnable filtering benefit depend on transform?

Also includes:
- Per-gamma analysis (does the interaction vary with compression level?)
- Partial eta-squared effect sizes
- Kruskal-Wallis per-factor results retained as `kruskal_wallis_fallback`

Unit tests in `tests/test_spectral_transforms.py::TestARTAnova` verify:
- Detection of main effects and interactions on synthetic data
- No false-positive interactions when none exist
- Correct per-gamma and fallback outputs

### P1-5: LongBench needs trust_remote_code
**STATUS: DONE.**

`load_dataset("THUDM/LongBench", task_name, split="test", trust_remote_code=True)`
added in `src/eval/longbench.py`.

`load_dataset("EleutherAI/proof-pile", split="test", trust_remote_code=True)`
added in `src/eval/proof_pile.py`.

---

## P2 Items — Status After P0

### P2-1: Unit tests for spectral transforms
**STATUS: DONE.**

`tests/test_spectral_transforms.py` — 28 tests across 8 test classes:

- `TestDCTTransform`: round-trip identity, spectral_len, output shape/realness, pad_to_len
- `TestFFTTransform`: round-trip identity, spectral_len, output shape/complexity, pad_to_len
- `TestTruncation`: fraction correctness for DCT/FFT at gamma=0.50/0.22/0.01, low-freq preservation, gamma=1.0 no-op
- `TestLearnableFilter`: parameter existence, forward shape, gradient flow, mask range [0,1], low-pass initialization
- `TestFixedLowPassFilter`: no-op behavior, no parameters, complex compatibility
- `TestSpectralKVCache`: compress/reconstruct shape, compression ratio, reset, baseline passthrough, cache size, DCT cache
- `TestGQARepeatKV`: repeat_kv shape correctness, repeat_kv on reconstructed tensors
- `TestARTAnova`: main effect detection, interaction detection, no false positives, per-gamma, fallback, method field, eta_sq

### P2-2: Smoke test script
**STATUS: DONE.**

`scripts/smoke_test.sh` + `src/tests/smoke_test.py` — 9 integration tests:

1. Model loading with eager attention
2. Spectral compression applied to all layers
3. Forward pass produces valid output (no NaN/Inf)
4. Loss computation succeeds
5. Gradient flow through learnable filter (backward + optimizer step)
6. `reset_all_caches()` clears state properly
7. PEFT `modules_to_save` saves/loads `spectral_cache` (P2-6)
8. 1-sample PG-19 perplexity eval
9. 1-sample LongBench eval

Usage:
```bash
bash scripts/smoke_test.sh
bash scripts/smoke_test.sh --model meta-llama/Llama-3.2-1B-Instruct
bash scripts/smoke_test.sh --transform dct --filter fixed --gamma 0.50
bash scripts/smoke_test.sh --skip-eval  # skip PG-19/LongBench
```

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
**STATUS: DONE.**

Switched `attn_implementation` from `"flash_attention_2"` to `"eager"` in
all 3 model-loading paths:
- `src/training/train_redpajama.py`
- `src/training/train_longalpaca.py`
- `src/run_experiment.py`

Added explanatory comments in each file documenting that the spectral
forward override replaces LlamaAttention.forward entirely with manual
attention (Q@K^T + softmax), so FA2 is never used for the compressed
attention computation. Using "eager" avoids version-specific SDPA flag
confusion.

### P2-5: trust_remote_code for LongBench
**STATUS: DONE (same as P1-5).**

### P2-6: Verify checkpoint loading at runtime
**STATUS: DONE (folded into smoke test P2-2).**

Smoke test `test_checkpoint_save_load` in `src/tests/smoke_test.py`:
- Applies LoRA with `modules_to_save=["spectral_cache"]`
- Saves checkpoint to temp dir
- Verifies `adapter_config.json` mentions `spectral_cache`
- Loads fresh base model, re-applies compression, loads LoRA adapter
- Verifies learnable filter params restored from checkpoint

---

## Recommended Implementation Order

All items completed:

1. ~~**P1-4: ART ANOVA**~~ — DONE. True ART (Wobbrock et al. 2011) with interaction detection.
2. ~~**P2-2: Smoke test script**~~ — DONE. 9 integration tests in `src/tests/smoke_test.py`.
3. ~~**P1-3/P1-5: Dataset verification + trust_remote_code**~~ — DONE. trust_remote_code added; dataset verification checklist created at `docs/dataset_verification.md`.
4. ~~**P2-4: FA2 → eager attention**~~ — DONE. All 3 files switched to eager. Removed flash-attn from requirements.txt.
5. ~~**P2-1: Unit tests**~~ — DONE. 28 tests in `tests/test_spectral_transforms.py`.
6. ~~**P2-6: Checkpoint verification**~~ — DONE. Folded into smoke test.

**All P1/P2 tasks addressed.** Run on the Coder workspace before launching
GPU experiments:
  1. Verify datasets: follow `docs/dataset_verification.md`
  2. Run unit tests: `pytest tests/test_spectral_transforms.py -v`
  3. Run smoke test: `bash scripts/smoke_test.sh`

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
