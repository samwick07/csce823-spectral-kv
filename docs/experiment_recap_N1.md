# CSCE 823 Spectral KV-Cache Experiment — As-Coded Recap (N=1)

**Date:** 2026-08-22
**Repo:** samwick07/csce823-spectral-kv
**Purpose:** Reference for future N=30 distributional analysis

---

## Model

- **Base model:** meta-llama/Llama-3.1-8B-Instruct (gated, ~16GB, bf16)
- **Fine-tuning:** LoRA rank 8, alpha 16, dropout 0.05
- **Distributed:** DeepSpeed ZeRO-2, 4x H200 (140 GiB each), torch_adam (no nvcc)
- **Stack:** Python 3.12, CUDA 13.0, torch 2.13.0+cu130, transformers 4.57.6, datasets 2.21.0, deepspeed 0.19.5

---

## Experiment Matrix: 14 configs (C00–C12), 1 seed (seed 0)

2x2 factorial x 3 compression ratios + 1 baseline.

| Config | Transform | Filter | Gamma | Description |
|--------|-----------|--------|-------|-------------|
| C00    | none      | none   | 1.0   | Baseline (no compression) |
| C01    | DCT       | fixed  | 0.50  | FreqKV reproduction |
| C02    | DCT       | fixed  | 0.22  | |
| C03    | DCT       | fixed  | 0.01  | Most aggressive |
| C04    | DCT       | learnable | 0.50 | |
| C05    | DCT       | learnable | 0.22 | |
| C06    | DCT       | learnable | 0.01 | |
| C07    | FFT       | fixed  | 0.50  | Phase-preserving |
| C08    | FFT       | fixed  | 0.22  | |
| C09    | FFT       | fixed  | 0.01  | |
| C10    | FFT       | learnable | 0.50 | Full proposed method |
| C11    | FFT       | learnable | 0.22 | |
| C12    | FFT       | learnable | 0.01 | |

**Factors:**
- Transform: DCT (real, discards phase via `.real`) vs FFT (complex rfft, preserves phase)
- Filter: Fixed low-pass (no-op passthrough; transform.truncate() does the cut) vs Learnable sigmoid mask (per-layer per-head, init as smooth low-pass decay)
- Gamma: 0.50, 0.22, 0.01 (fraction of spectral coefficients retained)

---

## Spectral Mechanism (src/spectral/)

### transform.py
- **DCTTransform:** DCT-II via FFT interleaving trick. Real-valued, discards phase. Precomputed/cached index+phase tensors per (N, device).
- **FFTTransform:** Complex rfft. Preserves phase. Returns N//2+1 complex coefficients.

### filter.py
- **FixedLowPassFilter:** No-op; truncation handled by transform's truncate() method (keeps first gamma*N coefficients).
- **LearnableSpectralFilter:** Per-layer, per-head sigmoid soft mask over frequency bins. Initialized as smooth decay from 1.0 (low freq) to 0.0 (high freq). Parameters: filter_logits [num_heads, max_spectral_len]. Backpropagated during training.

### attention.py
Overrides LlamaAttention.forward:
1. Compute Q, K, V projections
2. Apply RoPE to Q, K
3. Transform K, V to spectral domain (DCT or FFT)
4. Apply filter (fixed passthrough or learnable mask)
5. Truncate to gamma fraction of coefficients
6. Reconstruct K, V from compressed spectral representation
7. Repeat KV heads for GQA (8 KV heads -> 32 query heads)
8. Standard attention: softmax(Q @ K^T / sqrt(d)) @ V
9. Output projection

Attention backend fallback chain: FlashAttention-2 -> SDPA -> manual.

### cache.py
SpectralDynamicCache for incremental K=1 generation caching: prefill compresses full prompt K/V; decode appends one token, reconstructs old+new, recompresses. O(N log N) per step instead of O(N^2).

---

## Training (2 phases per config)

### Phase 1 — RedPajama CPT
- **Dataset:** RedPajama (Together Computer)
- **Steps:** 1000
- **Global batch:** 64 (4 GPU x 8 micro_bs x 2 grad_accum)
- **Learning rate:** 1e-5, 100 warmup steps, constant schedule
- **Sequence length:** 2048 (reduced from 16K to fix CUDA OOM)
- **Attention:** SDPA (changed from eager; O(S) vs O(S^2) memory)
- **LoRA:** Applied via get_peft_model() (fresh adapters)
- **Checkpointing:** Every 500 steps, save_total_limit=2
- **Completion:** .training_complete marker file

### Phase 2 — LongAlpaca SFT
- **Dataset:** LongAlpaca-12k (instruction/input/output columns)
- **Epochs:** 5 (940 steps)
- **Global batch:** 64 (4 GPU x 1 micro_bs x 16 grad_accum)
- **Learning rate:** 1e-5
- **Sequence length:** 16384 (16K context)
- **Attention:** SDPA
- **LoRA:** Loaded via PeftModel.from_pretrained(is_trainable=True) from Phase 1 checkpoint
- **Checkpointing:** Every 100 steps, save_total_limit=5
- **Memory:** ~68-94 GiB per GPU (of 140 GiB H200)

### Bug Fixes Applied (13 total, commits 77f2262 -> 5f3a482)
1. DS warmup_max_lr mismatch -> scheduler/optimizer="auto"
2. FusedAdam ImportError (no nvcc) -> torch_adam=true
3. Loss not scalar -> model_accepts_loss_kwargs=False
4. Loss grad chain broken -> enable_input_require_grads()
5. CUDA OOM Phase 1 -> RedPajama seq_len 16384->2048
6. LongAlpaca KeyError -> column fix instruction/input/output
7. Checkpoint lexicographic sort -> numeric sort
8. Phase 2 per-epoch saves -> per-100-step saves
9. HF cache ephemeral -> auto-restore from PVC backup
10. PeftModel is_trainable=False -> is_trainable=True
11. Phase 2 OOM (softmax 64 GiB) -> micro_bs 2->1, grad_accum 8->16
12. Phase 2 OOM (eager O(S^2)) -> SDPA O(S) attention
13. Phase 1 also switched to SDPA for consistency

---

## Evaluation (4 benchmarks)

### 1. PG-19 Perplexity
- **Dataset:** emozilla/pg19-test (parquet mirror of deepmind/pg19)
- **Samples:** 100 books (random selection, seed-controlled)
- **Protocol:** 256-token sliding window, non-overlapping stride (stride=window_size)
- **Tokenization:** Truncate to 16384 max length
- **Metrics:** mean, median, std, min, max perplexity
- **Code:** src/eval/pg19.py, src/eval/metrics.py (compute_sliding_window_perplexity)

### 2. Proof-pile Perplexity
- **Dataset:** hoskinson-center/proof-pile (or EleutherAI/proof-pile-2 fallback)
- **Samples:** 100 docs (random selection, seed-controlled)
- **Protocol:** 256-token sliding window, non-overlapping stride
- **Tokenization:** Truncate to 16384 max length
- **Metrics:** mean, median, std, min, max perplexity
- **Code:** src/eval/proof_pile.py

### 3. LongBench V1
- **Dataset:** Xnhyacinth/LongBench (parquet mirror of THUDM/LongBench)
- **Tasks:** 14 tasks across 5 categories:
  - Single-doc QA: narrativeqa, qasper, multifieldqa_en
  - Multi-doc QA: hotpotqa, 2wikimqa, musique
  - Summarization: gov_report, qmsum, multi_news
  - Few-shot: trec, triviaqa, samsum
  - Code completion: passage_retrieval_en, lcc
- **Decoding:** Stochastic — temp=0.7, top_p=0.9, max_new_tokens=256
- **Max input length:** 8192 per task
- **Scoring (SIMPLIFIED — not official LongBench metrics):**
  - QA tasks: token-level F1
  - Summarization: simplified ROUGE-L (LCS-based)
  - Code completion: character-level edit similarity
- **Code:** src/eval/longbench.py

### 4. Efficiency
- **Prompt:** "The quick brown fox jumps over the lazy dog."
- **Generate:** 128 tokens, greedy (do_sample=False)
- **Metrics:** peak KV memory (GB), decoding latency (ms/token), compression overhead (%)
- **Code:** src/eval/efficiency.py, src/eval/metrics.py (compute_efficiency_metrics)

### Eval Orchestration
- Per (config, seed) pair, distributed across 4 GPUs in a flat work queue
- N=1 = 1 seed = 14 eval runs total
- Each eval: load Phase 2 checkpoint, apply spectral compression, run all 4 benchmarks
- Spectral caches reset between samples/windows (no cross-sample leakage)

---

## Statistical Analysis (src/stats/)

7-step pipeline:
1. Aggregate results across configs/seeds
2. Point estimates (mean, std, confidence intervals)
3. Experiment matrix construction
4. ANOVA (2x2 factorial: transform type x filter type)
5. Interaction effects analysis
6. Post-hoc comparisons
7. Summary report

---

## Orchestrator (src/orchestrator.py)

4 phases: Train -> Eval -> Analyze -> Exfil (HF Hub upload)
- Idempotent state file (results/orchestrator_state.json)
- Crash-safe: checkpoint resume, 3x retry, 30s retry delay
- Marker files: .training_complete (Phase 1), .training_complete (Phase 2)
- Pilot mode: 3 configs (C00, C07, C10) x 5 seeds
- Flat work queue for eval: distributes (config, seed) pairs across GPUs

---

## Comparison with FreqKV (arXiv:2505.00570, ICLR 2026)

### Directly Comparable

**PG-19 perplexity:** YES
- Same dataset, same 256-token sliding window, same non-overlapping stride
- FreqKV reports at context lengths 2K/4K/8K/16K/32K; we eval at 16K
- Compare our 16K numbers against their 16K column

**Proof-pile perplexity:** YES
- Same dataset, same 256-token sliding window protocol
- Same context length comparison available

### Partially Comparable

**LongBench:** PARTIALLY
- Same benchmark (14 tasks, same categories)
- BUT: our scorers are simplified (F1/ROUGE-L/edit-sim), not official LongBench metrics
- FreqKV reports per-category averages; we compute per-task then overall_mean
- Can compare trends but not absolute numbers directly

### Not Comparable

- **Needle-in-a-Haystack:** We don't have it; FreqKV does (Figure 3)
- **Decoding latency:** Different methodology (128 tokens, single prompt vs their setup)

### Key Differences Affecting Comparison

1. **Model:** FreqKV uses LLaMA-2-7B and LLaMA-3-8B; we use Llama-3.1-8B-Instruct
   - Different base model means different absolute perplexity baselines
   - Our C00 baseline numbers won't match their "Full FT" row

2. **Training approach:**
   - FreqKV: CPT on RedPajama AT the target extended length (4K->8K/16K/32K)
   - Us: CPT on RedPajama at 2048 (original length), then SFT on LongAlpaca at 16K
   - FreqKV extends context window; we train at 16K directly

3. **Compression architecture:**
   - FreqKV: iterative compression — cache fills to preset window N, then compresses gamma*(N-S) tokens, S=4 sink tokens kept uncompressed
   - Us: compress the full sequence at once during training; iterative only during generation
   - FreqKV has sink tokens (S=4); our code has no sink token handling

4. **Compression ratio:**
   - FreqKV: only gamma=0.5 in main results
   - Us: gamma=0.5, 0.22, 0.01 (more aggressive exploration)

5. **Transform:**
   - FreqKV: DCT only (real-valued, discards phase)
   - Us: DCT AND FFT (complex, preserves phase) — our novel contribution

6. **Filter:**
   - FreqKV: parameter-free (fixed low-pass only)
   - Us: fixed AND learnable sigmoid mask — our novel contribution

7. **Eval context lengths:**
   - FreqKV: evaluates at 2K, 4K, 8K, 16K, 32K (shows extrapolation beyond training length)
   - Us: evaluate at 16K only (training length)

8. **LongBench metrics:**
   - FreqKV: official LongBench metrics
   - Us: simplified F1/ROUGE-L/edit-similarity (code explicitly notes this)

### FreqKV Published Results (for reference)

**PG-19 Perplexity (LLaMA-2-7B, training length 8192):**
| Method | Cache | 2K | 4K | 8K | 16K | 32K |
|--------|-------|-----|-----|-----|-----|-----|
| Full FT | Full | 7.55 | 7.21 | 6.98 | - | - |
| LongLoRA | Full | 7.70 | 7.35 | 7.14 | - | - |
| LoCoCo* | Compressed | 8.15 | 8.08 | 7.27 | - | - |
| FreqKV | Compressed | 7.45 | 7.12 | 7.04 | 7.02 | 7.02 |

**PG-19 Perplexity (LLaMA-2-7B, training length 16384):**
| Method | Cache | 2K | 4K | 8K | 16K | 32K |
|--------|-------|-----|-----|-----|-----|-----|
| LongLoRA | Full | 7.65 | 7.28 | 7.02 | 6.86 | - |
| FreqKV | Compressed | 7.46 | 7.13 | 7.03 | 6.99 | 6.98 |

**PG-19 Perplexity (LLaMA-2-7B, training length 32768):**
| Method | Cache | 2K | 4K | 8K | 16K | 32K |
|--------|-------|-----|-----|-----|-----|-----|
| LongLoRA | Full | 8.29 | 7.83 | 7.54 | 7.35 | 7.22 |
| FreqKV | Compressed | 7.47 | 7.14 | 7.04 | 7.00 | 6.98 |

**FreqKV training setup:**
- LoRA rank 8, lr 1e-6 to 2e-5 (20 warmup steps, constant)
- Batch size 64 (GPU_num x batch_per_device x grad_accum)
- LongAlpaca SFT for 5 epochs (6.28K samples)
- gamma=0.5, S=4 sink tokens
- Context window extension: 4K->8K/16K/32K

### Bottom Line

C01 (DCT + Fixed + gamma=0.5) is the closest FreqKV reproduction. PG-19 and Proof-pile perplexity at 16K context length use the same protocol, same datasets, same eval window — directly comparable. Absolute numbers won't match exactly (different model: 3.1 vs 2/3, different training: 2048 CPT vs extended-length CPT, no sink tokens), but the relative comparison (compressed vs our own baseline C00) is valid and meaningful.

For LongBench, swap in official LongBench metrics (from THUDM/LongBench repo) to directly compare against their published scores. Simplified scorers show trends but not absolute comparability.

The real value-add is the 2x2 factorial (DCT vs FFT, fixed vs learnable) across 3 ratios — FreqKV only tests DCT+fixed at one ratio (gamma=0.5). This experiment shows whether phase preservation (FFT) and adaptive filtering (learnable) improve over the FreqKV baseline, which is the novel contribution for the paper/dissertation.

---

## Notes for N=30 Runs

- Configs are identical; only `--seeds` changes (0-29 instead of 0)
- Stochastic decoding (temp=0.7, top_p=0.9) means LongBench scores will vary across seeds
- PG-19/Proof-pile perplexity is deterministic (greedy, no sampling) — variation comes from random sample selection (seed-controlled)
- Statistical analysis pipeline (src/stats/) is designed for N=30: ANOVA, confidence intervals, distributional analysis
- Pilot recommendation: C00, C07, C10 x 5 seeds first to validate pipeline before committing ~1200 GPU-hours
- Eval flat work queue: N=30 means 14 configs x 30 seeds = 420 eval runs, distributed 4-at-a-time across GPUs
