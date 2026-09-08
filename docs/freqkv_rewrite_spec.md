# FreqKV Causal Rewrite Implementation Spec

**Date:** August 29, 2026
**Branch:** `feat/freqkv-causal-rewrite`
**Status:** Design complete (25 decisions via grilling), implementation in progress

---

## Executive Summary

Rewrite the spectral KV-cache compression to mirror FreqKV (arXiv:2505.00570, ICLR 2026)
for causally-correct chunk-wise attention during both training and inference. The current
`_spectral_forward` operates on the full sequence at once, causing a causality violation
(future V leaks into past attention outputs via spectral reconstruction). This rewrite
replaces it with FreqKV's chunk-wise approach where compression only applies to PAST tokens.

The project's novel contributions (learnable filter, FFT transform) are preserved as
extensions to the FreqKV framework.

---

## Design Decisions (25 questions, 5 rounds)

### Round 1 — Foundational Architecture

| Q | Decision |
|---|----------|
| Q1 | **Full replacement** of `_spectral_forward`. No parallel path. Old code preserved in git history. |
| Q2 | **Spatial KV storage + on-the-fly compression.** No spectral coefficient storage. KV stored as standard spatial tensors, compressed per-chunk during forward pass. |
| Q3 | **sink_size=4, recent_size=8 as config params** with FreqKV defaults. Configurable for ablation. |
| Q4 | **cache_size as config param** (2048 for Phase 1, 8192 for Phase 2). Phase 1 seq_len (2048) <= cache_size, so no compression during CPT — matches FreqKV protocol. |
| Q5 | **Shared learnable filter applied per chunk independently.** Same filter weights reused across chunks. Per-chunk variation is a future ablation. |

### Round 2 — Implementation Mechanics

| Q | Decision |
|---|----------|
| Q6 | **Ortho-normalized DCT with sqrt(L/N) scaling** (option a). Also test `torch.fft.dct` when available for comparison (option c). Replace custom DCT in transform.py. |
| Q7 | **Compress BEFORE RoPE** (un-rotated keys). Match FreqKV's iterate path. Apply RoPE to reconstructed/compressed keys per chunk. |
| Q8 | **flash_attn_varlen_kvpacked_func per chunk** (match FreqKV). With non-flash SDPA/manual fallback path. |
| Q9 | **Rename SpectralKVCache → SpectralKVCompressor.** Stateless compression utility. Keep module name "spectral_cache" on attn layer for LoRA modules_to_save compatibility. Remove SpectralDynamicCache (use standard DynamicCache). |
| Q10 | **Phase 1: no compression** (seq_len=2048 <= cache_size=2048). Matches FreqKV: model learns compression during SFT (Phase 2 at 16K with cache_size=8192). |

### Round 3 — Compression Mechanics and Integration

| Q | Decision |
|---|----------|
| Q11 | **Rewrite DCTTransform with ortho norm + sqrt(L/N) + precomputed constants cache.** Not FreqKV's verbatim functions — keep performance optimizations and SpectralTransform abstraction. Verify mathematical equivalence via unit test. |
| Q12 | **Generalize compress logic into SpectralTransform base class.** `compress(x, compress_len)` calls forward → filter → truncate → inverse → scale. Each subclass implements forward/inverse/truncate only. DRY for DCT+FFT. |
| Q13 | **Filter on FULL spectrum BEFORE truncation.** Filter sees all coefficients, can learn to retain high-freq. Truncation enforces compression budget. This makes the learnable filter strictly more expressive than fixed low-pass. |
| Q14 | **Standard DynamicCache, forward manages compression.** Cache stays bounded at cache_size. Iterate forward stores compressed group K/V back into DynamicCache. Match FreqKV's forward_iterate. |
| Q15 | **seq_len-sized chunks with 256-token sliding windows within** for perplexity eval. Configurable seq_len as ablation (option c). Compression triggers when seq_len > cache_size. Rewrite compute_sliding_window_perplexity. |

### Round 4 — Forward Pass Structure and Fallbacks

| Q | Decision |
|---|----------|
| Q16 | **Both flash_attn and non-flash paths.** Match FreqKV's `forward_*_flashattn` / `forward_*_noflashattn` pattern. Flash primary, SDPA/manual fallback. |
| Q17 | **Both no-iterate and iterate forward paths.** No-iterate for training/perplexity eval (fresh compression per chunk). Iterate for generation (compounding compression, bounded cache). |
| Q18 | **Add sink_size, recent_size, cache_size, use_flash_attn to ExperimentConfig.** New YAMLs get new fields. Old YAMLs backwards compatible (pick up defaults). |
| Q19 | **Stateless compress(x, compress_len) API.** SpectralKVCompressor is a pure compression utility. Chunk-wise logic (sink, recent, boundaries) lives in the forward pass. |
| Q20 | **Minimal + comprehensive + tiny-model smoke tests.** Mathematical equivalence test vs FreqKV reference functions. Causality verification. Tiny-model loss sanity check (not 100x below baseline). |

### Round 5 — Final Frontier

| Q | Decision |
|---|----------|
| Q21 | **Fix RoPE inconsistency: BOTH paths compress before RoPE.** FreqKV's no-iterate path compresses AFTER RoPE (inconsistent with iterate path). Our fix: both compress before RoPE. **Document this for publication.** |
| Q22 | **Iterate path for SFT training** (is_iterate=True). Model learns to handle compounding compression. Matches FreqKV. |
| Q23 | **File plan confirmed.** 3 rewrites, 12 patches, 3 new test files. |
| Q24 | **Pass is_iterate as parameter to apply_spectral_compression().** Matches FreqKV's replace_llama_attn(use_flash_attn, is_iterate, ...) pattern. |
| Q25 | **Layer-by-layer implementation with verification gates.** (1) transform.py + tests → (2) compressor + tests → (3) attention forward + tests → (4) config + training + eval → (5) full smoke test. |

---

## Critical Architectural Finding

FreqKV's `dct_compress` returns an **L-length output** (NOT an N-length reconstruction).
The compressed KV has fewer positions — L compressed "tokens" represent the low-frequency
structure of the original N tokens. The cache always stays bounded at:

```
cache_size = sink_size + fft_size + recent_size + chunk_size
           = sink_size + int(fft_span * gamma) + recent_size + (fft_span - fft_size)
           = sink_size + fft_span + recent_size
           = cache_size  (tautology — bounded by construction)
```

where `fft_span = cache_size - sink_size - recent_size`.

The `sqrt(L/N)` scaling compensates for the amplitude reduction when going from N to L samples
in the ortho-normalized DCT/IDCT pair.

---

## RoPE Inconsistency (Publication Note)

FreqKV's reference code has an inconsistency between its two forward paths:

- `forward_iterate` (training + generation): compresses **un-rotated** K, then applies RoPE
  to the compressed K per chunk via `apply_rotary_pos_emb_varlen`.
- `forward_noiterate` (perplexity eval): applies RoPE to the **full sequence first**,
  then compresses the already-rotated keys.

This creates a train/eval mismatch. Our implementation fixes this by compressing before
RoPE in **both** paths, consistent with the theoretically sounder approach (compress raw
key structure, then apply positional encoding to the approximation).

**For publication:** Document as "We resolve an inconsistency in the FreqKV reference
implementation where the no-iterate evaluation path applies rotary position embeddings
before spectral compression, while the iterate training path compresses before RoPE.
Our implementation consistently compresses before RoPE in both paths, ensuring
train/eval consistency."

---

## File Change Map

### Files to REWRITE (3)

1. **`src/spectral/transform.py`**
   - Rewrite `DCTTransform` with `norm='ortho'` DCT/IDCT and `sqrt(L/N)` scaling
   - Add `compress(x, compress_len)` method to `SpectralTransform` base class:
     forward → filter → truncate → inverse → scale
   - Adapt `FFTTransform` to support the same `compress()` interface
   - Keep precomputed constants cache for performance
   - `torch.fft.dct` detection for optional comparison path

2. **`src/spectral/cache.py`**
   - Rename `SpectralKVCache` → `SpectralKVCompressor`
   - Remove KV state storage (compress/reconstruct/append methods)
   - Keep filter parameter storage (LearnableSpectralFilter)
   - Add `compress(x, compress_len)` delegating to transform
   - Remove `SpectralDynamicCache` (replaced by standard DynamicCache)
   - Keep `CompressionConfig` — add sink_size, recent_size, cache_size, use_flash_attn

3. **`src/spectral/attention.py`**
   - Replace `_spectral_forward` with `_freqkv_forward_noiterate` and `_freqkv_forward_iterate`
   - `apply_spectral_compression(model, config, is_iterate=True)` installs the right forward
   - Both paths: project Q/K/V → chunk-wise compress raw K → apply RoPE per chunk → attend
   - Flash path: `flash_attn_varlen_kvpacked_func` with `unpad_input`/`pad_input`
   - Non-flash path: SDPA with manual causal mask per chunk
   - Remove `create_spectral_dynamic_cache`, `SpectralDynamicCache` usage
   - Update `get_spectral_caches` → `get_spectral_compressors`

### Files to PATCH (12)

4. **`src/utils/config.py`** — Add sink_size, recent_size, cache_size, use_flash_attn
5. **`src/training/lora_config.py`** — modules_to_save stays "spectral_cache" (unchanged)
6. **`src/training/train_redpajama.py`** — Pass new config fields; is_iterate=True
7. **`src/training/train_longalpaca.py`** — Pass new config fields; is_iterate=True
8. **`src/run_experiment.py`** — Remove SpectralDynamicCache; use DynamicCache; is_iterate per mode
9. **`src/eval/metrics.py`** — Rewrite compute_sliding_window_perplexity for seq_len chunks
10. **`src/eval/pg19.py`** — Pass cache_size; chunk-based perplexity
11. **`src/eval/proof_pile.py`** — Same as pg19.py
12. **`src/eval/longbench.py`** — Replace SpectralDynamicCache with DynamicCache
13. **`src/eval/efficiency.py`** — Replace SpectralDynamicCache with DynamicCache
14. **`src/spectral/__init__.py`** — Update exports
15. **All 13 config YAMLs (C00-C12)** — Add new fields with defaults

### Files to CREATE (3)

16. **`tests/test_freqkv_transforms.py`** — Ortho DCT/IDCT, round-trip, truncation, sqrt(L/N),
    mathematical equivalence to FreqKV reference `dct`/`idct`/`dct_compress`
17. **`tests/test_spectral_kv_compressor.py`** — Compressor unit tests: L-length output,
    filter application, DCT and FFT paths
18. **`tests/test_freqkv_forward.py`** — Integration: no-iterate and iterate forward,
    causality verification, cache boundedness, tiny-model smoke test

---

## Implementation Order (Layer-by-Layer with Verification Gates)

### Gate 1: transform.py + tests
- Rewrite DCTTransform with ortho norm
- Add compress() to base class
- Adapt FFTTransform
- Write test_freqkv_transforms.py
- **Verify:** DCT round-trip exact, truncation produces L-length, sqrt(L/N) scaling correct,
  mathematical equivalence to FreqKV reference

### Gate 2: cache.py (compressor) + tests
- Rename to SpectralKVCompressor
- Remove KV state, add compress() delegation
- Write test_spectral_kv_compressor.py
- **Verify:** compress returns L-length, filter applies before truncation, both DCT+FFT work

### Gate 3: attention.py (forward) + tests
- Implement _freqkv_forward_noiterate and _freqkv_forward_iterate
- Flash + non-flash paths
- Write test_freqkv_forward.py
- **Verify:** output shapes correct, causality (no future leakage), cache bounded at cache_size

### Gate 4: config + training + eval integration
- Patch config.py, training files, eval files, run_experiment.py
- Update YAMLs
- **Verify:** imports resolve, config loads, training script initializes

### Gate 5: full smoke test
- Tiny model (1 layer, 2 heads, 64 dim) through one training step
- **Verify:** loss is reasonable (not 100x below baseline), no crashes

---

## FreqKV Reference

- Paper: arXiv:2505.00570 (ICLR 2026)
- Code: https://github.com/LUMIA-Group/FreqKV
- Key file: `llama_attn_replace_dct_mempe.py`
- Training: `supervised-fine-tune.py` (is_iterate=True, cache_size=4096/8192)
- Eval: `eval.py` (is_iterate=False, sliding_window=256)
