# Training Loss Computation Investigation: Root Cause Analysis

**Date:** August 29, 2026
**Branch:** `investigation/loss-computation-audit`
**Status:** Root cause identified; fixes proposed for future full experiment (N=30)

---

## Executive Summary

The 100x training loss gap between baseline (C00, loss ~2.09) and compressed
configs (C01-C12, loss ~0.02) observed during the N=1 run is caused by a
**causality violation in the spectral KV-cache compression**. The DCT/FFT
transform operates on the full sequence dimension, and after truncation +
reconstruction, the V values at position j (for j <= t) contain information
from tokens at positions j+1, ..., N-1 (future tokens). This allows the model
to "cheat" during next-token prediction by exploiting future information
embedded in the reconstructed V tensor.

This is NOT a bug in the loss computation itself — the HF Trainer computes
standard cross-entropy loss on full-dimensional logits. The problem is in the
forward pass: the spectral reconstruction creates an information channel from
future tokens to past attention outputs.

---

## Detailed Root Cause Analysis

### 1. The Causality Violation

**Standard causal attention (baseline, C00):**
- At position t, the model attends to positions 0, 1, ..., t (causal mask).
- V[j] for j <= t only contains information about token j.
- The attention output at position t is a function of tokens 0..t only.

**Spectral compressed attention (C01-C12):**
- The spectral transform (DCT or FFT) operates on the FULL sequence dimension.
- `compress()`: K, V [B, H, S, D] → transform → truncate → store
- `reconstruct()`: stored spectral → inverse transform → K_recon, V_recon [B, H, S, D]
- After truncation, V_recon[j] for ANY j contains a mixture of information
  from ALL positions 0..S-1, including future positions j+1..S-1.
- The causal attention mask prevents Q[t] from attending to K_recon[j] for j > t,
  but does NOT prevent future information embedded in V_recon[j] (for j <= t)
  from flowing into the attention output at position t.

### 2. Code Path Trace

The information flow is:

```
_spectral_forward() (attention.py:221)
  ↓
  K, V projected from hidden_states (lines 254-256)
  ↓
  spectral_cache.compress(K, V)  (line 310)
    → transform.forward(K)       # FFT/DCT on full sequence [B,H,S,D]
    → transform.truncate(...)    # Keep gamma fraction of coefficients
    → filter(...)                # Apply learnable mask (if learnable)
    → store compressed spectral
  ↓
  K_recon, V_recon = spectral_cache.reconstruct()  (line 313)
    → transform.inverse(stored, target_len=S)
    → V_recon[j] now contains mixed info from all positions
  ↓
  repeat_kv(V_recon, ...)  (line 324)  # GQA expansion
  ↓
  _compute_attention(Q, K_recon, V_recon, causal=True)  (line 327)
    → causal mask on Q @ K_recon^T  ✓  (prevents attending to future K)
    → BUT: softmax(QK^T) @ V_recon uses V_recon[j<=t] which contains future info ✗
  ↓
  attn_output → o_proj → logits → cross-entropy loss
```

The loss function receives logits that were computed using V_recon values
containing future information. The cross-entropy loss is standard, but the
logits themselves are "contaminated."

### 3. Diagnostic Verification

Three diagnostic tests were run (see branch commit) confirming:

**Test 1 - Cross-position leakage:**
Modifying V at position 50 changes V_recon at position 10:
- DCT (gamma=0.5): change of 0.0724 (should be 0 if causal)
- FFT (gamma=0.5): change of 0.1563 (should be 0 if causal)
- At gamma=1.0 (no truncation): round-trip is exact, but the mixing still
  exists — it's just perfectly invertible so no information is lost.

**Test 2 - Attention output leakage:**
Modifying V at position 40 changes attention output at position 10:
- Baseline (no compression): 0.0000000000 (causal mask blocks it)
- Compressed (DCT, gamma=0.5): 0.0454 (future info leaks through V_recon)

**Test 3 - Predictive correlation:**
Cosine similarity between attn_output[t] and token[t+1]:
- Baseline: 0.0224 (random, no predictive power)
- DCT gamma=0.5: 0.0629 (2.8x higher, mild leakage)
- DCT gamma=0.01: 0.1410 (6.3x higher, strong leakage at high compression)

The leakage increases with compression (lower gamma) because more high-frequency
coefficients are discarded, causing more cross-position mixing in the reconstruction.

### 4. Why This Produces a 100x Loss Gap

During training with cross-entropy loss on next-token prediction:

1. The model receives V_recon[j<=t] that contains information about token t+1
2. Through LoRA fine-tuning, the model learns to extract this future information
3. The attention output at position t becomes predictive of token t+1
4. The logits at position t can predict token t+1 with much higher accuracy
5. Cross-entropy loss drops dramatically (from ~2.0 to ~0.02)

The 100x gap is not a measurement artifact — it reflects the model genuinely
learning to exploit the future information channel. The compressed configs
are solving an easier task: "predict token t+1 given tokens 0..t AND a
lossy spectral mixture of all tokens including t+1."

### 5. Why the Baseline (C00) Shows Higher Loss

C00 uses `transform_type="none"`, which sets `config.is_baseline = True`.
In this case, `apply_spectral_compression()` returns immediately (line 142-144
of attention.py), and the standard LlamaAttention.forward is used with SDPA.
No spectral transform is applied, no reconstruction occurs, and the causal mask
perfectly prevents future information leakage. The loss of ~2.09 is the genuine
CPT loss on RedPajama.

### 6. Additional Findings

#### 6.1 DCT Phase Discarding

The DCT transform (line 182 of transform.py) computes `(V * phase).real`,
discarding the imaginary part. This is mathematically correct for DCT-II
(the output is real-valued), but it means the DCT inverse must reconstruct
the Hermitian symmetry from the real coefficients alone (lines 238-246).
This is handled correctly — the inverse DCT is exact at gamma=1.0 (verified
in tests).

#### 6.2 FFT Complex Tensor Handling

The FFT transform uses `torch.fft.rfft` (line 319 of transform.py) which
returns complex coefficients, and `torch.fft.irfft` for reconstruction
(line 332). The complex tensor flows through the filter correctly:
- LearnableSpectralFilter multiplies complex spectral by real sigmoid mask (line 140 of filter.py)
- This is mathematically valid: `complex * real = complex`

No complex tensor bugs were found. The gradient flow through the filter is
correct (verified in test_filter_gradient_flow).

#### 6.3 SDPA vs Manual Attention

The spectral forward pass uses `_compute_attention()` which has a fallback
chain: FlashAttention-2 → SDPA → manual. During training, SDPA is used
(with `is_causal=True` for q_len > 1). The causal mask is correctly applied
to the attention weights. The issue is not in the attention computation
itself but in the V_recon tensor that feeds into it.

#### 6.4 No Loss Computation Bug

The HF Trainer computes loss using the model's built-in `CausalLMLoss`
(standard cross-entropy with label shifting). The `model_accepts_loss_kwargs=False`
setting (line 231 of train_redpajama.py) correctly prevents the transformers 4.57
`num_items_in_batch` kwarg injection. The loss function is not the problem —
the logits fed to it are.

---

## Proposed Fixes

### Fix A: Chunked Spectral Compression (Recommended)

Instead of transforming the full sequence at once, apply the spectral
transform to local chunks (windows) of the K/V tensors. This preserves
causality: each chunk only mixes information within its window, and the
causal mask prevents attention from accessing future chunks' V values.

```python
# In SpectralKVCache.compress():
# Instead of:
#   k_spectral = self.transform(key_states)
# Do:
#   chunk_size = 256  # or configurable
#   k_spectral = torch.cat([
#       self.transform(key_states[:, :, start:start+chunk_size, :])
#       for start in range(0, seq_len, chunk_size)
#   ], dim=2)
```

**Pros:** Preserves causality, aligns with FreqKV's local window approach,
minimal code change.
**Cons:** Slightly reduces compression efficiency (chunk boundaries create
spectral artifacts).

### Fix B: Causal Spectral Transform

Modify the transform to only use past information when reconstructing V[j].
This could be done by:
1. Transform the full sequence (as now)
2. Before reconstruction, zero out spectral coefficients that would carry
   future information to position j
3. This is complex to implement correctly for both DCT and FFT

**Pros:** Theoretically optimal (no information loss).
**Cons:** Difficult to implement, may not be well-defined for FFT.

### Fix C: Per-Position Reconstruction (FreqKV Approach)

Following the original FreqKV paper more closely, compress only the KV cache
for positions that have already been attended to (past positions), and keep
the current position's K/V in spatial domain. This is the approach used in
generation (K=1 incremental caching) but applied during training.

**Pros:** Exactly matches FreqKV protocol, no causality issues.
**Cons:** Requires significant refactoring of the training forward pass.

### Fix D: Diagnostic Validation (Minimum Viable)

Before implementing any fix, add a validation test that checks for causality:

```python
def test_spectral_causality():
    """Verify that V_recon[j] does not contain information from positions > j."""
    # Modify V at future position, check V_recon at past position is unchanged
    ...
```

This should be added to `tests/test_spectral_transforms.py` and made a
blocking test before any future experiment run.

---

## Impact on Current N=1 Results

### Training Loss
The 100x loss gap is NOT a genuine finding — it is an artifact of the causality
violation. The compressed configs' training loss values (~0.02 in Phase 1,
~0.003-0.007 in Phase 2) are artificially low and should not be cited as
results in the paper or compared to the baseline.

### Evaluation Results
The evaluation results (PG-19, Proof-pile, LongBench) are still valid as
comparative metrics between configs, because:
- During evaluation, each window/sample is compressed independently
- The spectral mixing still occurs within each window, but since evaluation
  measures perplexity/generation quality (not training loss), the "cheating"
  advantage is reduced (the model can't exploit future V during generation
  because the SpectralDynamicCache uses K=1 incremental updates)
- However, the LoRA weights were trained with the causality violation, so
  the learned weights may be subtly different from what they would be with
  a causal transform

### Statistical Analysis
The relative ranking of configs in the evaluation may still hold, but the
training loss curves should not be used as evidence of compression quality.
The paper should focus on evaluation metrics (perplexity, LongBench scores)
rather than training loss.

---

## Recommendations for N=30

1. **Implement Fix A (chunked spectral compression) before N=30.** This is
   the minimum change needed to produce valid training loss comparisons.

2. **Add the causality test (Fix D) to the test suite.** This should be a
   blocking CI check.

3. **Re-run at least 3 configs (C00, C01, C10) with the fix** to verify the
   loss gap narrows to a reasonable range (compressed loss should be higher
   than baseline, not lower, due to reconstruction error).

4. **Do not cite N=1 training loss values in the paper.** Use only evaluation
   metrics (PG-19, Proof-pile, LongBench, efficiency).

5. **Document the causality violation as a known issue** in the lessons
   learned document and the paper's limitations section.

---

## Files Examined

| File | Lines | Role |
|------|-------|------|
| `src/spectral/transform.py` | 359 | DCT/FFT transforms (forward, inverse, truncate) |
| `src/spectral/filter.py` | 158 | Fixed and learnable spectral filters |
| `src/spectral/cache.py` | 442 | SpectralKVCache: compress, reconstruct, append |
| `src/spectral/attention.py` | 477 | _spectral_forward: K/V projection → compress → reconstruct → attend |
| `src/training/train_redpajama.py` | 266 | Phase 1 CPT training (standard HF Trainer) |
| `src/training/train_longalpaca.py` | 278 | Phase 2 SFT training (standard HF Trainer) |
| `src/training/lora_config.py` | 55 | LoRA config with modules_to_save for spectral_cache |
| `src/run_experiment.py` | 398 | Experiment entry point: train + eval orchestration |
| `src/eval/metrics.py` | 186 | Sliding window perplexity (standard cross-entropy) |
| `src/eval/pg19.py` | 115 | PG-19 evaluation |
| `src/utils/config.py` | 107 | ExperimentConfig dataclass and YAML loader |
| `tests/test_spectral_transforms.py` | 684 | Unit tests (no causality test) |
| `docs/N30_LESSONS_LEARNED.md` | 612 | Lessons learned (Section 4 flags this issue) |
| 13 config YAMLs | — | C00-C12 experiment configurations |
