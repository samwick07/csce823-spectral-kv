# Experiment Specification: Spectral KV-Cache Compression

**Project:** Spectral KV-Cache Compression via Complex FFT with Learnable Frequency Selection
**Course:** CSCE 823 Final Project (preliminary work for journal article and dissertation)
**Author:** Samuel Chadwick
**Repo:** github.com/samwick07/csce823-spectral-kv
**Date:** August 29, 2026 (unified from compute_resources.md, dataset_verification.md, experiment_recap_N1.md)
**Supersedes:** docs/compute_resources.md, docs/dataset_verification.md, docs/experiment_recap_N1.md

> **Authoritative sources:** The GitHub issue tracker and `docs/N30_LESSONS_LEARNED.md`
> are the authoritative source of truth for action items and hardening. This doc is
> the authoritative reference for the experimental design as implemented on
> `feat/freqkv-causal-rewrite`.

---

## 1. Hardware and Compute Budget

### 1.1 Cluster

| Spec                  | Value                          |
|-----------------------|--------------------------------|
| GPU                   | NVIDIA H200 (SXM5)            |
| HBM3e per GPU         | 141 GB                         |
| Memory bandwidth      | 4.8 TB/s                       |
| FP16/BF16 compute     | 1,979 TFLOPS                  |
| Interconnect          | NVLink 4.0 (900 GB/s)         |
| Allocation            | 4x H200 (current) or 8x H200  |

### 1.2 Model Memory Footprint (Llama-3.1-8B-Instruct, LoRA r=8)

| Component                  | Phase 1 (2K ctx) | Phase 2 (16K ctx) |
|---------------------------|------------------|-------------------|
| Model weights (BF16)      | ~16 GB           | ~16 GB            |
| LoRA adapters (r=8)       | ~21 MB           | ~21 MB            |
| Optimizer (ZeRO-2 split)  | ~32 GB total     | ~4 GB/GPU         |
| Activations               | ~40 GB           | ~60 GB            |
| KV cache                  | ~4 GB            | ~2 GB             |
| **Total per GPU**         | **~55-65 GB**    | **~82 GB**        |
| **Headroom (of 141 GB)**  | **~76-86 GB**    | **~59 GB**        |

### 1.3 Training Budget

| Phase   | Dataset       | Steps/Epochs | Global Batch | Seq Len  |
|---------|---------------|--------------|--------------|----------|
| Phase 1 | RedPajama     | 1,000 steps  | 64           | 2,048    |
| Phase 2 | LongAlpaca-16k| 5 epochs (940 steps) | 64   | 16,384   |

13 configs (12 compressed + 1 baseline), trained sequentially via DeepSpeed ZeRO-2.

### 1.4 Evaluation Budget

| Benchmark   | Samples | Method                |
|-------------|---------|-----------------------|
| PG-19       | 100     | Sliding window PPL    |
| Proof-pile  | 100     | Sliding window PPL    |
| LongBench   | 14 tasks| Generation (256 tok)  |
| Efficiency  | 1       | 128 token generation  |

| Scope       | Configs | Seeds | Total eval runs |
|-------------|---------|-------|-----------------|
| N=1 (class) | 13      | 1     | 13              |
| N=30 (pub)  | 13      | 30    | 390             |

### 1.5 Storage

| Item                        | Size      |
|----------------------------|-----------|
| Base model (Llama-3.1-8B)   | ~16 GB    |
| LoRA checkpoints (13)       | ~300 MB   |
| HF dataset cache            | ~50 GB    |
| Eval results (JSON)         | ~500 MB   |
| **Total**                   | **~72 GB**|

---

## 2. Datasets

### 2.1 Training Datasets

#### RedPajama (Phase 1 — CPT)
- **HF ID:** `togethercomputer/RedPajama-Data-1T` (streaming; the `-Sample` variant was removed from HF)
- **Split:** `train`
- **Columns:** `text`, `meta`
- **Notes:** ~1B tokens sample. Streamed into memory (~80K samples). Pre-download to PVC cache.

#### LongAlpaca-16k (Phase 2 — SFT)
- **HF ID:** `Yukang/LongAlpaca-16k`
- **Split:** `train`
- **Columns:** `instruction`, `input`, `output`, `file`
- **Notes:** ~16k long-context instruction pairs. `input` field omitted when empty (Alpaca convention).

### 2.2 Evaluation Datasets

#### PG-19 (Perplexity)
- **HF ID:** `deepmind/pg19` (or `ozilla/pg19-test` parquet mirror)
- **Split:** `test` (~100 books)
- **Columns:** `text`, `short_book_title`, `publication_date`
- **Protocol:** seq_len-sized chunks (default 2048) with 256-token sliding windows within each chunk. Compression triggers when seq_len > cache_size.

#### Proof-pile (Perplexity)
- **HF ID:** `hoskinson-center/proof-pile` (primary), `EleutherAI/proof-pile-2` (fallback)
- **Split:** `test` (~46,251 rows)
- **trust_remote_code:** True (for proof-pile-2)
- **Notes:** Requires `zstandard` package for `.jsonl.zst` decompression. Pre-download to avoid 429 rate limits.

#### LongBench V1 (Generation)
- **HF ID:** `THUDM/LongBench` (NOT THUIAR)
- **Split:** `test`
- **Tasks (14):** narrativeqa, qasper, multifieldqa_en, hotpotqa, 2wikimqa, musique, gov_report, qmsum, multi_news, trec, triviaqa, samsum, passage_retrieval_en, lcc
- **trust_remote_code:** True
- **Columns:** `context`, `input`, `answers`, `length`, `dataset`
- **Decoding:** temp=0.7, top_p=0.9, max_new_tokens=256

### 2.3 Pre-Flight Checklist

Before launching:
- [ ] HF_TOKEN set (37+ chars, gated access to meta-llama/Llama-3.1-8B-Instruct)
- [ ] WANDB_API_KEY valid
- [ ] WANDB_PROJECT=csce823-spectral-kv set in environment
- [ ] All 5 datasets load successfully
- [ ] All 14 LongBench tasks accessible
- [ ] trust_remote_code=True accepted for proof-pile and LongBench
- [ ] zstandard installed in venv
- [ ] HF cache on persistent PVC (not ephemeral overlay)
- [ ] Smoke test passes: `bash scripts/smoke_test.sh`

---

## 3. Experiment Matrix

2x2 factorial x 3 compression ratios + 1 baseline = 13 configs.

| Config | Transform | Filter    | Gamma | Compression  | 2x2 Cell   |
|--------|-----------|-----------|-------|--------------|------------|
| C00    | none      | none      | 1.0   | 1x (baseline)| Baseline   |
| C01    | DCT       | Fixed LP  | 0.50  | 2x           | A: DCT+Fix |
| C02    | DCT       | Fixed LP  | 0.22  | 4.5x         | A: DCT+Fix |
| C03    | DCT       | Fixed LP  | 0.01  | 100x         | A: DCT+Fix |
| C04    | DCT       | Learnable | 0.50  | 2x           | B: DCT+Lrn |
| C05    | DCT       | Learnable | 0.22  | 4.5x         | B: DCT+Lrn |
| C06    | DCT       | Learnable | 0.01  | 100x         | B: DCT+Lrn |
| C07    | FFT       | Fixed LP  | 0.50  | 2x           | C: FFT+Fix |
| C08    | FFT       | Fixed LP  | 0.22  | 4.5x         | C: FFT+Fix |
| C09    | FFT       | Fixed LP  | 0.01  | 100x         | C: FFT+Fix |
| C10    | FFT       | Learnable | 0.50  | 2x           | D: FFT+Lrn |
| C11    | FFT       | Learnable | 0.22  | 4.5x         | D: FFT+Lrn |
| C12    | FFT       | Learnable | 0.01  | 100x         | D: FFT+Lrn |

**Factors:**
- **Transform:** DCT (real-valued, ortho-normalized, discards phase) vs FFT (complex rfft, preserves phase) — novel contribution beyond FreqKV
- **Filter:** Fixed low-pass (parameter-free truncation) vs Learnable sigmoid mask (per-layer per-head, backpropagated) — novel contribution beyond FreqKV
- **Gamma:** 0.50, 0.22, 0.01 (fraction of spectral coefficients retained)

**Novel contributions** (vs FreqKV which only tests DCT + fixed + gamma=0.5):
1. Complex FFT preserving phase information
2. Learnable frequency selection via sigmoid mask
3. 2x2 factorial isolating phase preservation from adaptive filtering
4. Three compression ratios including extreme (gamma=0.01, 100x)

---

## 4. Architecture (feat/freqkv-causal-rewrite)

### 4.1 Base Model
- **Model:** meta-llama/Llama-3.1-8B-Instruct (gated, ~16GB, BF16)
- **Fine-tuning:** LoRA rank 8, alpha 16, dropout 0.05
- **Distributed:** DeepSpeed ZeRO-2, 4x H200, torch_adam (no nvcc needed)
- **Stack:** Python 3.12, CUDA 13.0, PyTorch 2.13.0+cu130, transformers 4.57.6, datasets 2.21.0, deepspeed 0.19.5

### 4.2 Spectral Mechanism (src/spectral/)

The architecture mirrors FreqKV (arXiv:2505.00570, ICLR 2026) with extensions.

#### transform.py
- **DCTTransform:** Ortho-normalized DCT-II via FFT interleaving. Real-valued. Precomputed/cached index+phase tensors per (N, device). Includes `sqrt(L/N)` scaling on compression.
- **FFTTransform:** Complex rfft. Preserves phase. Returns N//2+1 complex coefficients.
- **SpectralTransform (base):** `compress(x, compress_len)` method: forward → filter → truncate → inverse → scale. Shared by both DCT and FFT.

#### filter.py
- **FixedLowPassFilter:** No-op; truncation handled by transform's truncate() method.
- **LearnableSpectralFilter:** Per-layer, per-head sigmoid soft mask over frequency bins. Applied to FULL spectrum BEFORE truncation (can learn to retain high-freq that truncation would discard). Parameters: `filter_logits [num_heads, max_spectral_len]`. Backpropagated during training.

#### cache.py (renamed from SpectralKVCache → SpectralKVCompressor)
- **SpectralKVCompressor:** Stateless compression utility. Holds transform + filter + config. Provides `compress(x, compress_len)` method. No KV state storage — compression is on-the-fly per chunk.
- **CompressionConfig:** sink_size=4, recent_size=8, cache_size=2048 (Phase 1) or 8192 (Phase 2), use_flash_attn=True.
- Module name on attn layer stays `"spectral_cache"` for LoRA `modules_to_save` compatibility.

#### attention.py (FreqKV-aligned chunk-wise forward)
Two forward paths, matching FreqKV:
- **`_freqkv_forward_noiterate`** (perplexity eval): Processes full sequence in one forward pass. Chunk-wise compression from full spatial K/V tensor. Fresh compression per chunk.
- **`_freqkv_forward_iterate`** (training + generation): Uses standard `DynamicCache`. Stores compressed group K/V back. Cache stays bounded at `cache_size`. Compounding compression — model learns to handle it.

**Chunk-wise compression flow:**
1. Project Q, K, V
2. First `cache_size` tokens: standard causal attention
3. Later chunks of size `(fft_span - fft_size)` attend to:
   - Sink tokens (uncompressed, first `sink_size` positions)
   - Compressed past (DCT/FFT → filter → truncate → IDCT/irfft → scale, compressed to `fft_size` positions)
   - Recent tokens (uncompressed, last `recent_size` positions)
   - Current chunk queries with causal self-attention
4. Apply RoPE AFTER compression (both paths — fixes FreqKV's no-iterate inconsistency)

**Cache budget:** `cache_size = sink_size + fft_span + recent_size` where `fft_span = cache_size - sink_size - recent_size` and `fft_size = int(fft_span * gamma)`.

**Attention backends:** FlashAttention-2 (primary, via `flash_attn_varlen_kvpacked_func`) with SDPA/manual fallback (for environments without flash-attn).

### 4.3 Causality Guarantee

The old `_spectral_forward` compressed the full sequence at once, causing future V information to leak into past attention outputs via V_recon[j<=t]. The new chunk-wise approach compresses ONLY PAST tokens per chunk, eliminating cross-chunk leakage entirely. Within-chunk leakage is bounded by chunk_size and eliminated by the causal mask.

**Verified:** 65 tests passing, including smoke test showing compressed loss ratio = 1.00x baseline (was 0.01x with old code).

---

## 5. Training (2 phases per config)

### Phase 1 — RedPajama CPT
- **Dataset:** RedPajama-Data-1T (streaming)
- **Steps:** 1,000
- **Global batch:** 64 (4 GPU x 8 micro_bs x 2 grad_accum)
- **Learning rate:** 1e-5, 100 warmup steps, constant schedule
- **Sequence length:** 2,048
- **Compression:** None (seq_len=2048 <= cache_size=2048, matches FreqKV protocol)
- **Checkpointing:** Every 500 steps, save_total_limit=2

### Phase 2 — LongAlpaca SFT
- **Dataset:** LongAlpaca-16k
- **Epochs:** 5 (940 steps)
- **Global batch:** 64 (4 GPU x 1 micro_bs x 16 grad_accum)
- **Learning rate:** 1e-5
- **Sequence length:** 16,384
- **Compression:** Active (seq_len=16384 > cache_size=8192, triggers chunk-wise compression)
- **Training path:** Iterate (is_iterate=True) — model learns compounding compression
- **Checkpointing:** Every 100 steps, save_total_limit=5

### Bug Fixes Applied (13 total, commits 77f2262 → 5f3a482)
1. DS warmup_max_lr mismatch → scheduler/optimizer="auto"
2. FusedAdam ImportError (no nvcc) → torch_adam=true
3. Loss not scalar → model_accepts_loss_kwargs=False
4. Loss grad chain broken → enable_input_require_grads()
5. CUDA OOM Phase 1 → RedPajama seq_len 16384→2048
6. LongAlpaca KeyError → column fix instruction/input/output
7. Checkpoint lexicographic sort → numeric sort
8. Phase 2 per-epoch saves → per-100-step saves
9. HF cache ephemeral → auto-restore from PVC backup
10. PeftModel is_trainable=False → is_trainable=True
11. Phase 2 OOM (softmax 64 GiB) → micro_bs 2→1, grad_accum 8→16
12. Phase 2 OOM (eager O(S^2)) → SDPA O(S) attention
13. Phase 1 also switched to SDPA for consistency

---

## 6. Evaluation

### 6.1 PG-19 Perplexity
- **Samples:** 100 books (random, seed-controlled)
- **Protocol:** seq_len-sized chunks (default 2048) with 256-token sliding windows within each chunk. Compression triggers when seq_len > cache_size.
- **Metrics:** mean, median, std, min, max perplexity

### 6.2 Proof-pile Perplexity
- **Samples:** 100 docs (random, seed-controlled)
- **Protocol:** Same as PG-19
- **Metrics:** Same

### 6.3 LongBench V1
- **Tasks:** 14 tasks across 5 categories (single-doc QA, multi-doc QA, summarization, few-shot, code completion)
- **Decoding:** Stochastic — temp=0.7, top_p=0.9, max_new_tokens=256
- **Max input length:** 8,192 per task
- **Scoring (simplified, not official LongBench metrics):**
  - QA tasks: token-level F1
  - Summarization: simplified ROUGE-L (LCS-based)
  - Code completion: character-level edit similarity

### 6.4 Efficiency
- **Prompt:** "The quick brown fox jumps over the lazy dog."
- **Generate:** 128 tokens, greedy (do_sample=False)
- **Metrics:** peak KV memory (GB), decoding latency (ms/token), compression overhead (%)

### 6.5 Eval Orchestration
- Per (config, seed) pair, distributed across 4 GPUs in a flat work queue
- Each eval: load Phase 2 checkpoint, apply spectral compression, run all 4 benchmarks
- Standard DynamicCache for generation (FreqKV iterate forward manages compression internally)
- Spectral compressors reset between samples/windows

---

## 7. Statistical Analysis (src/stats/)

7-step pipeline (for N=30):
1. Aggregate results across configs/seeds
2. Point estimates (mean, std, confidence intervals)
3. Experiment matrix construction
4. ART ANOVA (2x2 factorial: transform type x filter type) — Wobbrock et al. 2011
5. Interaction effects analysis
6. Post-hoc comparisons (Wilcoxon signed-rank, Friedman, Nemenyi)
7. Summary report with Holm-Bonferroni correction

For N=1: point estimates only (mean, delta vs baseline).

---

## 8. Comparison with FreqKV

### Directly Comparable
- **PG-19 perplexity:** Same dataset, same 256-token sliding window. FreqKV reports at 2K/4K/8K/16K/32K; we eval at 16K.
- **Proof-pile perplexity:** Same dataset, same protocol.

### Partially Comparable
- **LongBench:** Same benchmark (14 tasks), but our scorers are simplified (F1/ROUGE-L/edit-sim), not official LongBench metrics.

### Key Differences
1. **Model:** FreqKV uses LLaMA-2-7B and LLaMA-3-8B; we use Llama-3.1-8B-Instruct
2. **Training:** FreqKV extends context window (4K→8K/16K/32K); we train at 2K then 16K directly
3. **Compression:** Both use chunk-wise compression with sink tokens (S=4). We add FFT and learnable filter.
4. **Compression ratio:** FreqKV only gamma=0.5; we test 0.5, 0.22, 0.01
5. **Transform:** FreqKV DCT only; we test DCT AND FFT (novel)
6. **Filter:** FreqKV parameter-free; we test fixed AND learnable (novel)
7. **RoPE:** We fix FreqKV's inconsistency — both paths compress before RoPE (documented for publication)

C01 (DCT + Fixed + gamma=0.5) is the closest FreqKV reproduction.

---

## 9. Orchestrator (src/orchestrator.py)

4 phases: Train → Eval → Analyze → Exfil (HF Hub + GitHub)
- Idempotent state file (results/orchestrator_state.json)
- Crash-safe: checkpoint resume, 3x retry, 30s retry delay
- Marker files: .training_complete
- Pilot mode: 3 configs (C00, C07, C10) x N seeds
- Flat work queue for eval: distributes (config, seed) pairs across GPUs
- Phased evaluation: `--eval-phase quick` (PG-19 + Proof-pile + efficiency) or `--eval-phase longbench`

---

## 10. Key Files Reference

| File | Role |
|------|------|
| `src/spectral/transform.py` | DCT/FFT transforms with ortho norm, compress() base method |
| `src/spectral/filter.py` | Fixed low-pass + learnable sigmoid mask |
| `src/spectral/cache.py` | SpectralKVCompressor: stateless compression utility |
| `src/spectral/attention.py` | FreqKV-aligned chunk-wise forward (no-iterate + iterate) |
| `src/training/lora_config.py` | LoRA config with modules_to_save |
| `src/training/train_redpajama.py` | Phase 1: RedPajama CPT |
| `src/training/train_longalpaca.py` | Phase 2: LongAlpaca SFT |
| `src/eval/pg19.py` | PG-19 sliding-window perplexity |
| `src/eval/proof_pile.py` | Proof-pile perplexity |
| `src/eval/longbench.py` | LongBench V1 generation tasks |
| `src/eval/metrics.py` | Sliding-window PPL (seq_len chunks) + efficiency metrics |
| `src/eval/efficiency.py` | KV memory, latency, compression overhead |
| `src/stats/analyze.py` | 7-step statistical pipeline (ART ANOVA) |
| `src/stats/point_estimates.py` | N=1 point estimate table |
| `src/utils/config.py` | ExperimentConfig with all compression params |
| `src/utils/wandb_utils.py` | W&B integration |
| `src/run_experiment.py` | Main orchestration (train + eval) |
| `src/orchestrator.py` | Autonomous experiment runner |
| `configs/experiment_C00.yaml` – `C12.yaml` | 13 experiment configs |
| `tests/test_freqkv_transforms.py` | 34 tests: DCT/FFT, ortho norm, FreqKV equivalence |
| `tests/test_spectral_kv_compressor.py` | 16 tests: compressor, all 4 config variants |
| `tests/test_freqkv_forward.py` | 9 tests: forward paths, causality, boundedness |
| `tests/test_smoke_tiny_model.py` | 6 tests: loss sanity across all variants |
