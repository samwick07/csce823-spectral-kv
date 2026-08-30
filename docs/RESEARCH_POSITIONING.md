# Research Positioning and Expanded Factorial Design

**Project:** Spectral KV-Cache Compression via Complex FFT with Learnable Frequency Selection
**Author:** Samuel Chadwick
**Repo:** github.com/samwick07/csce823-spectral-kv
**Date:** August 30, 2026
**Status:** Planning — for journal article and dissertation

---

## 1. Competitive Landscape: FreqKV Reception Assessment

### 1.1 FreqKV Publication Status

FreqKV (arXiv:2505.00570, Kai et al., LUMIA Lab / SJTU / Huawei Noah's Ark) was:
- **Rejected** from ICLR 2025 (Sept 2024 submission, forum KscheKSYrh)
- **Accepted as ICLR 2026 poster** (not oral/spotlight) after substantial revision (v1→v3, May 2025→Jan 2026)
- Key additions in revision: RULER, LongGenBench, prefilling-stage experiments, broader baselines (ThinK, LaCache, D2O, GemFilter)

### 1.2 Community Engagement: Low

| Signal | Value |
|--------|-------|
| GitHub stars | 2 |
| HuggingFace comments | 0 |
| HuggingFace collections | 0 |
| Reddit threads | 0 (no dedicated discussion on r/LocalLLaMA or r/MachineLearning) |
| alphaXiv discussion | 0 comments |
| Semantic Scholar citations | 3 (as of Aug 2026) |
| Secondary coverage | Auto-generated blog summaries only (papernotes, themoonlight, emergentmind, chatpaper) |

### 1.3 Follow-Up Work (Research Traction Growing)

Several papers have built on or cited FreqKV, confirming the research direction is active:

1. **FAEDKV** (Li et al., Jul 2025, arXiv:2507.20030) — Directly addresses FreqKV's iterative compression bias. Introduces Infinite-Window DFT (IWDFT) for unbiased, position-agnostic retention. Outperforms eviction methods by up to 22% on LongBench. Uses complex DFT instead of DCT.

2. **FlashCache** (Yang et al., Nov 2025) — Extends frequency-domain KV compression to multimodal models. 80% memory reduction, <1% accuracy drop.

3. **Fourier Compressor** (Wang, Kai, Lin, 2025) — Same group (FreqKV first author Jushi Kai) extending to vision-language models.

4. **InfoKV** (Kai et al., 2026) — Same first author's follow-up. Entropy-aware, information-theoretic compression signals.

5. **CacheTune** (arXiv:2605.24022) — Leverages FreqKV's frequency-domain observations for KV cache reuse.

6. **Multimodal Outlier-KV** (arXiv:2511.16786) — Builds on FreqKV's frequency observation for outlier-KV-aware compression.

### 1.4 FreqKV Known Weaknesses (Our Entry Points)

| # | Weakness | Source | Our Counter |
|---|----------|--------|-------------|
| W1 | DCT discards phase information | Architectural choice | FFT preserves phase (our Factor 1) |
| W2 | Progressive degradation of older tokens (up to 127 compression cycles at 256K) | Admitted in paper; FAEDKV addresses this | Unbiased retention via IWDFT (new Factor) |
| W3 | Fixed, uniform low-pass filter across all layers/heads | Paper's own Appendix B shows head-wise variation | Learnable per-head filter (our Factor 2) |
| W4 | Performance degradation on mathematical/symbolic content (Proof-pile) | Admitted in Appendix E | If FFT phase preservation helps symbolic precision, we show it |
| W5 | RoPE train/eval inconsistency (v1/v2) | We discovered this; v3 claims fix | Both paths compress before RoPE (documented) |
| W6 | Only tested on 7B/8B models | Paper limitation | Same scale, but systematic ablation is the contribution |
| W7 | Still requires fine-tuning despite "parameter-free" claim | "Parameter-free" = no compression modules, not zero training | We are transparent about LoRA requirement |
| W8 | Never tests complex FFT vs DCT or learnable vs fixed | No ablation performed | Our entire 2x2 factorial fills this gap |
| W9 | No comparison to unbiased retention strategies | FAEDKV published after FreqKV v3 | Our expanded factorial includes IWDFT |

---

## 2. FAEDKV Technical Summary

**Paper:** FAEDKV: Infinite-Window Fourier Transform for Unbiased KV Cache Compression
**Authors:** Runchao Li, Yao Fu, Mu Sheng, Xianxuan Long, Haotian Yu, Pan Li (Case Western Reserve University)
**arXiv:** 2507.20030 (Jul 2025)
**Venue:** Preprint (not yet peer-reviewed as of Aug 2026)

### 2.1 Core Innovation: IWDFT

The Infinite-Window DFT (IWDFT) is a recursive update rule for maintaining a frequency-domain KV cache that grows with each new token, without re-compressing old tokens:

```
S_{t+1}[k] = W_k * ((N-1)/N * S_t[k] + 1/N * x[t+1])
```

where:
- `S_t[k]` = frequency-domain state at bin k after t tokens
- `W_k = e^{-j*2*pi*k/M}` = complex exponential (twiddle factor)
- `N` = current sequence length (approximate (N-1)/N ≈ 1 for N > 1000)
- `x[t+1]` = new token's K or V vector

**Key properties:**
- O(M) per decoding step (M = number of retained frequency bins)
- No time-domain storage of past tokens (unlike sliding-window DFT)
- No hard cutoff of old information (unlike eviction or iterative re-compression)
- Normalization factor prevents floating-point overflow on long sequences
- **Unbiased:** every token contributes equally to the compressed representation

### 2.2 Why DFT over DCT (FAEDKV's Argument)

FAEDKV explicitly chooses DFT over DCT because:
> "DCT's implicit symmetric signal extension (mirroring) contributes to its strong emphasis on lower-frequency bins. It could lead to the loss of critical higher-frequency details."

This is independent corroboration that the transform choice (DCT vs complex DFT/FFT) matters — validating our Factor 1.

### 2.3 Frequency Ablation Study

FAEDKV performs a layer-wise frequency ablation:
1. Partition N frequency bins into C contiguous chunks
2. For each layer l, chunk c: zero out chunk c's DFT coefficients, reconstruct via IDFT, measure PPL increase Δ_{l,c}
3. Select top r*C most important chunks per layer (greedy, not low-pass)
4. Optimal C=22 (stabilizes at this granularity)

**Key finding:** Many high-frequency chunks yield significant Δ values, suggesting simple low-pass filtering is suboptimal. This directly supports our learnable filter hypothesis — a learnable mask can discover which high-frequency components matter, rather than blindly truncating.

### 2.4 FAEDKV Limitations (Our Opportunities)

1. **Training-free only** — never fine-tunes the model to adapt to compressed representations. FreqKV showed fine-tuning helps significantly. Our learnable filter + LoRA tests both.
2. **No phase ablation** — uses complex DFT but never compares against DCT to isolate phase contribution. Our DCT-vs-FFT factorial does exactly this.
3. **No learnable filter** — uses a static greedy selection from ablation. A learnable mask could adapt during training.
4. **Single A6000 GPU** — limited model scale (7B/8B only), same as us but less compute.
5. **Does not extend context window** — focused on compression within existing context length. FreqKV and our work extend beyond training length.
6. **Chunk-size hyperparameter** — requires a one-time ablation study (C=22 chunks) which adds setup complexity.

---

## 3. Expanded Factorial Design

### 3.1 Design Philosophy

The current 2x2 factorial isolates two design choices FreqKV left unexamined:
- Transform type (DCT vs FFT)
- Filter adaptivity (Fixed vs Learnable)

Adding IWDFT as a third transform type and unbiased retention as a design dimension creates a richer factorial that positions our work as the definitive ablation of the frequency-domain KV compression paradigm.

### 3.2 Three Factors

**Factor 1: Transform Type (3 levels)**
| Level | Transform | Phase | Basis | Source |
|-------|-----------|-------|-------|--------|
| DCT | Ortho-normalized DCT-II | Discarded (.real) | Cosine | FreqKV |
| FFT | Complex rfft | Preserved | Complex exponential | Our contribution |
| IWDFT | Infinite-Window DFT (recursive) | Preserved | Complex exponential + incremental update | FAEDKV |

**Factor 2: Filter Type (2 levels)**
| Level | Filter | Params | Description |
|-------|--------|--------|-------------|
| Fixed | Low-pass truncation | 0 | Retain first gamma*N bins |
| Learnable | Sigmoid soft mask | per-layer per-head | Backpropagated selection |

**Factor 3: Compression Ratio (3 levels)**
| Level | Gamma | Compression |
|-------|-------|-------------|
| Mild | 0.50 | 2x |
| Moderate | 0.22 | 4.5x |
| Extreme | 0.01 | 100x |

### 3.3 Retention Strategy Dimension

IWDFT inherently provides **unbiased retention** — old tokens are not re-compressed. DCT and FFT in the current design use **iterative re-compression** (FreqKV-style), where older tokens undergo progressively more compression cycles.

Two design options for incorporating unbiased retention:

**Option A: IWDFT as third transform (3x2x3 = 18 + 1 = 19 configs)**

IWDFT is added as a third transform level. Unbiased retention is an inherent property of IWDFT. The factorial cleanly compares three transforms x two filters x three gammas. The full method (SpecKV) = IWDFT + Learnable, which naturally incorporates unbiased retention.

Advantage: Clean factorial, fewer configs, IWDFT is self-contained.
Disadvantage: Cannot isolate "unbiased retention" from "IWDFT transform" — they are coupled.

**Option B: Retention strategy as fourth factor (non-full factorial)**

Add retention strategy as an explicit factor for DCT and FFT:
- DCT-iterative, DCT-unbiased, FFT-iterative, FFT-unbiased, IWDFT (inherently unbiased)

This creates 5 transform/retention combinations x 2 filters x 3 gammas = 30 + 1 = 31 configs.

"DCT-unbiased" and "FFT-unbiased" would implement the same incremental recursive update as IWDFT but using DCT or FFT basis functions — a novel contribution that isolates the transform from the retention strategy.

Advantage: Full isolation of all three design dimensions (transform, retention, filter).
Disadvantage: 31 configs is expensive; "DCT-unbiased" and "FFT-unbiased" are novel implementations not from any existing paper.

### 3.4 Recommended Design: Option A (19 configs) for Class, Option B (31 configs) for Publication

For the CSCE 823 class project (N=1): Option A (19 configs, 19 eval runs)
For the journal article / dissertation (N=30): Option B (31 configs, 930 eval runs)

### 3.5 Expanded Config Matrix (Option A — 19 configs)

| Config | Transform | Filter    | Gamma | Compression  | Cell       | Status |
|--------|-----------|-----------|-------|--------------|------------|--------|
| C00    | none      | none      | 1.0   | 1x (baseline)| Baseline   | Existing |
| C01    | DCT       | Fixed LP  | 0.50  | 2x           | A: DCT+Fix | Existing |
| C02    | DCT       | Fixed LP  | 0.22  | 4.5x         | A: DCT+Fix | Existing |
| C03    | DCT       | Fixed LP  | 0.01  | 100x         | A: DCT+Fix | Existing |
| C04    | DCT       | Learnable | 0.50  | 2x           | B: DCT+Lrn | Existing |
| C05    | DCT       | Learnable | 0.22  | 4.5x         | B: DCT+Lrn | Existing |
| C06    | DCT       | Learnable | 0.01  | 100x         | B: DCT+Lrn | Existing |
| C07    | FFT       | Fixed LP  | 0.50  | 2x           | C: FFT+Fix | Existing |
| C08    | FFT       | Fixed LP  | 0.22  | 4.5x         | C: FFT+Fix | Existing |
| C09    | FFT       | Fixed LP  | 0.01  | 100x         | C: FFT+Fix | Existing |
| C10    | FFT       | Learnable | 0.50  | 2x           | D: FFT+Lrn | Existing |
| C11    | FFT       | Learnable | 0.22  | 4.5x         | D: FFT+Lrn | Existing |
| C12    | FFT       | Learnable | 0.01  | 100x         | D: FFT+Lrn | Existing |
| C13    | IWDFT     | Fixed LP  | 0.50  | 2x           | E: IWDFT+Fix | NEW |
| C14    | IWDFT     | Fixed LP  | 0.22  | 4.5x         | E: IWDFT+Fix | NEW |
| C15    | IWDFT     | Fixed LP  | 0.01  | 100x         | E: IWDFT+Fix | NEW |
| C16    | IWDFT     | Learnable | 0.50  | 2x           | F: IWDFT+Lrn | NEW |
| C17    | IWDFT     | Learnable | 0.22  | 4.5x         | F: IWDFT+Lrn | NEW |
| C18    | IWDFT     | Learnable | 0.01  | 100x         | F: IWDFT+Lrn | NEW |

### 3.6 Expanded Config Matrix (Option B — 31 configs, publication)

Adds 12 configs (C19-C30) for DCT-unbiased and FFT-unbiased retention strategies:

| Config | Transform | Retention | Filter    | Gamma | Cell |
|--------|-----------|-----------|-----------|-------|------|
| C19    | DCT       | Unbiased  | Fixed LP  | 0.50  | G: DCT-U+Fix |
| C20    | DCT       | Unbiased  | Fixed LP  | 0.22  | G: DCT-U+Fix |
| C21    | DCT       | Unbiased  | Fixed LP  | 0.01  | G: DCT-U+Fix |
| C22    | DCT       | Unbiased  | Learnable | 0.50  | H: DCT-U+Lrn |
| C23    | DCT       | Unbiased  | Learnable | 0.22  | H: DCT-U+Lrn |
| C24    | DCT       | Unbiased  | Learnable | 0.01  | H: DCT-U+Lrn |
| C25    | FFT       | Unbiased  | Fixed LP  | 0.50  | I: FFT-U+Fix |
| C26    | FFT       | Unbiased  | Fixed LP  | 0.22  | I: FFT-U+Fix |
| C27    | FFT       | Unbiased  | Fixed LP  | 0.01  | I: FFT-U+Fix |
| C28    | FFT       | Unbiased  | Learnable | 0.50  | J: FFT-U+Lrn |
| C29    | FFT       | Unbiased  | Learnable | 0.22  | J: FFT-U+Lrn |
| C30    | FFT       | Unbiased  | Learnable | 0.01  | J: FFT-U+Lrn |

Note: IWDFT (C13-C18) is inherently unbiased, so no separate IWDFT-iterative exists.

---

## 4. SpecKV: The Full Method

The full method (SpecKV) incorporates all three novel contributions:

**SpecKV = IWDFT transform + Learnable filter + Unbiased retention**

This combines:
- Phase preservation (from FFT/IWDFT, addressing FreqKV W1)
- Adaptive per-head frequency selection (our learnable filter, addressing W3)
- Unbiased token retention (from IWDFT, addressing W2)
- No re-compression degradation (inherent in IWDFT's recursive update)

In Option A, SpecKV = C16/C17/C18 (IWDFT + Learnable at three gammas).
In Option B, SpecKV = C28/C29/C30 (FFT-unbiased + Learnable), OR C16/C17/C18 (IWDFT + Learnable) — the publication would compare both and report the best.

### 4.1 SpecKV vs FreqKV vs FAEDKV

| Property | FreqKV | FAEDKV | SpecKV (ours) |
|----------|--------|--------|---------------|
| Transform | DCT (real) | DFT (complex) | IWDFT or FFT-unbiased (complex) |
| Phase | Discarded | Preserved | Preserved |
| Filter | Fixed low-pass | Greedy ablation-selected | Learnable per-head sigmoid |
| Retention | Iterative (biased) | Unbiased (IWDFT) | Unbiased |
| Fine-tuning | LoRA r=8 | None (training-free) | LoRA r=8 |
| Context extension | Yes (8K→256K) | No (within existing window) | Yes (8K→256K) |
| Per-head adaptation | No | No (per-layer only) | Yes |
| Compression ratios tested | 0.5 only | 0.05-0.25 | 0.50, 0.22, 0.01 |

---

## 5. Research Narrative for Publication

### 5.1 Positioning Statement

> FreqKV established frequency-domain KV cache compression as a viable paradigm. FAEDKV demonstrated that unbiased retention via IWDFT outperforms iterative re-compression. However, no prior work has systematically ablated the key design choices: transform type (real vs complex), filter adaptivity (fixed vs learnable), and retention strategy (iterative vs unbiased). We present the first comprehensive factorial study isolating these three dimensions across three compression ratios, including extreme (100x) compression not tested by any prior work.

### 5.2 Key Claims (Supported by Experimental Design)

1. **Phase preservation matters:** FFT/IWDFT (complex) outperforms DCT (real) at high compression ratios, especially on symbolic content (Proof-pile). [Tested by Factor 1: C01-C06 vs C07-C12 vs C13-C18]

2. **Learnable filtering outperforms fixed low-pass:** Per-head adaptive frequency selection improves quality at all compression ratios. [Tested by Factor 2: C01/C04, C07/C10, C13/C16 etc.]

3. **Unbiased retention outperforms iterative re-compression:** IWDFT and unbiased DCT/FFT maintain better long-context fidelity than FreqKV-style iterative compression. [Tested by Factor 3 in Option B: C01 vs C19, C07 vs C25, etc.]

4. **The combination is greater than the sum of its parts:** SpecKV (IWDFT + Learnable + Unbiased) achieves the best performance across all benchmarks, with significant interaction effects. [Tested by ART ANOVA interaction terms]

5. **Extreme compression (100x) is viable:** Prior work tests at most 4-20x. We show that phase preservation + learnable filtering + unbiased retention enables functional performance even at gamma=0.01. [Tested by Factor 3: all configs at gamma=0.01]

### 5.3 Framing Against Prior Work

- **vs FreqKV:** "We extend FreqKV's frequency-domain paradigm with the first systematic ablation of transform type and filter adaptivity, and demonstrate that phase-preserving transforms and learnable filtering provide significant gains, particularly at extreme compression ratios."

- **vs FAEDKV:** "FAEDKV introduced unbiased retention via IWDFT but did not fine-tune the model or test learnable filters. We incorporate IWDFT into our factorial and show that combining unbiased retention with learnable per-head filtering and LoRA adaptation yields further improvements."

- **Concise one-liner:** "We present the first comprehensive factorial ablation of frequency-domain KV cache compression, isolating transform type, filter adaptivity, and retention strategy — three design dimensions that prior work chose individually but never compared systematically."

---

## 6. Implementation Notes

### 6.1 IWDFTTransform (new class in transform.py)

```python
class IWDFTTransform(SpectralTransform):
    """Infinite-Window DFT (FAEDKV, arXiv:2507.20030).
    
    Recursive frequency-domain update that provides unbiased token retention.
    Unlike iterative DCT/FFT compression, old tokens are not re-compressed;
    they contribute equally to a growing spectral representation.
    
    S_{t+1}[k] = W_k * ((N-1)/N * S_t[k] + 1/N * x[t+1])
    where W_k = exp(-j * 2*pi*k / M)
    """
```

Key implementation considerations:
- Stateful: maintains S_t[k] across decoding steps (unlike stateless DCT/FFT)
- Complex-valued: returns M complex coefficients (M = retained bins)
- Pre-filling: use standard DFT (Eq. 4 in FAEDKV) for initial context
- Decoding: use IWDFT recursive update (Eq. 9) for each new token
- Reconstruction: Sparse IDFT (only non-zero retained bins) for attention
- Approximation: (N-1)/N ≈ 1 for N > 1000 (per FAEDKV paper)

### 6.2 Unbiased DCT/FFT (Option B only)

For DCT-unbiased and FFT-unbiased, implement a recursive update analogous to IWDFT but using DCT or FFT basis functions. This is a novel contribution — no prior paper has done this. The key insight: IWDFT's normalization trick ((N-1)/N factor) can be applied to any spectral basis, not just complex exponentials.

### 6.3 Frequency Ablation Integration

FAEDKV's layer-wise frequency ablation (Section 3.2) can optionally replace or inform the learnable filter's initialization:
- Run ablation study → get Δ_{l,c} importance scores per layer per chunk
- Initialize learnable filter logits proportional to Δ scores (warm start)
- Let fine-tuning refine from there

This bridges FAEDKV's static ablation approach with our learnable approach.

### 6.4 Config YAML Changes

New configs C13-C18 (Option A) or C13-C30 (Option B) need:
- `transform: iwdft` (new value for transform field)
- `retention: unbiased` (new field, only for Option B)
- All other fields (filter, gamma, cache_size, etc.) unchanged

### 6.5 Compute Budget Impact

| Design | Configs | N=1 evals | N=30 evals | Est. GPU-hours (eval) |
|--------|---------|-----------|------------|----------------------|
| Current (2x2) | 13 | 13 | 390 | ~130 (quick) + ~1,560 (longbench) |
| Option A (3x2) | 19 | 19 | 570 | ~190 (quick) + ~2,280 (longbench) |
| Option B (5x2) | 31 | 31 | 930 | ~310 (quick) + ~3,720 (longbench) |

Training adds ~40 min/config Phase 1 + ~3h/config Phase 2 = ~3.7 GPU-hours/config.
- Option A: 19 x 3.7 = ~70 GPU-hours training
- Option B: 31 x 3.7 = ~115 GPU-hours training

Total for Option A N=30: ~70 + ~2,470 = ~2,540 GPU-hours (~11 days on 4x H200)
Total for Option B N=30: ~115 + ~4,030 = ~4,145 GPU-hours (~18 days on 4x H200)

---

## 7. Animation/Presentation Impact

The existing Manim animation includes three groups: FreqKV, ablation (2 groups), SpecKV. With the expanded design:
- FreqKV scene: unchanged (DCT + Fixed LP, iterative)
- Ablation scene: expand to show 3 transforms (DCT, FFT, IWDFT) x 2 filters
- SpecKV scene: now uses IWDFT + Learnable + Unbiased retention, showing the recursive update mechanism

The IWDFT recursive update (Eq. 9) has a natural visual: a frequency-domain state vector that absorbs each new token with a rotation (W_k twiddle factor) and normalization, contrasted with FreqKV's iterative compress→append→re-compress cycle.

---

## 8. Citation List for Expanded Related Work

Add to related work section:
- FAEDKV (Li et al., 2025) — IWDFT, unbiased retention, frequency ablation
- FlashCache (Yang et al., 2025) — multimodal frequency-domain compression
- Fourier Compressor (Wang, Kai, Lin, 2025) — VLM frequency compression
- InfoKV (Kai et al., 2026) — entropy-aware compression (same group as FreqKV)
- CacheTune (2026) — frequency-domain KV cache reuse
- Multimodal Outlier-KV (2025) — frequency-guided outlier identification

---

## 9. Statistical Analysis Plan

### 9.1 Hardware Budget

- **Hardware:** 8x NVIDIA H200 GPUs
- **Reserved compute window:** 90 days continuous
- **Total budget:** 17,280 GPU-hours
- **Bottleneck:** LongBench evaluation (~12 GPU-hours per config per seed)

### 9.2 Sample Size Determination

N is determined by power analysis, not assumed. The process:

1. **Pilot phase (post-class project):** Run 3-5 seeds for all configs.
2. **Compute observed effect sizes** (Cohen's d) for each factor from pilot data.
3. **Plug into power formulas** (below) to determine required N.
4. **Commit to final N** based on the most demanding factor that clears the power threshold.

Power analysis formulas (alpha=0.05, two-sided):

| Effect Size | Power | N (raw t-test) | N (Holm-5) | N (Wilcoxon+Holm-5) |
|-------------|-------|----------------|------------|---------------------|
| Large (d=0.8) | 0.80 | 13 | 19 | 22 |
| Large (d=0.8) | 0.90 | 17 | 24 | 28 |
| Medium (d=0.5) | 0.80 | 32 | 47 | 55 |
| Medium (d=0.5) | 0.90 | 43 | 60 | 69 |
| Small (d=0.2) | 0.80 | 197 | 292 | 336 |

Holm-Bonferroni correction is applied to the **k pre-registered main hypotheses only** (k=3-5), not to all pairwise comparisons. Wilcoxon signed-rank requires ~15% more samples than the paired t-test (Noether 1987 efficiency factor).

### 9.3 Multiple Comparison Correction Strategy

**Holm-Bonferroni is NOT applied across the full study.** Rationale:

- The full study includes 78 pairwise comparisons (C(13,2)) plus ANOVA tests. Holm correction across all 81 comparisons yields effective alpha = 0.0006, requiring N=292+ for medium effects — infeasible at 11,433+ GPU-hours.
- Additional planned factors (base model variants, KV cache window size extrapolations) further inflate the comparison count, making full correction even less practical.

**Tiered correction approach:**

| Tier | Tests | Correction | Reporting |
|------|-------|------------|-----------|
| **Primary (pre-registered)** | 3-5 main hypotheses | Holm-Bonferroni | Corrected p-values, effect size, 95% CI |
| **Secondary (exploratory)** | All other pairwise + interactions | None | Raw p-values, effect size (Cohen's d), 95% CI |

Pre-registered main hypotheses (to be finalized after pilot):

1. FFT/IWDFT outperforms DCT (phase preservation, Factor 1 main effect)
2. Learnable outperforms Fixed LP (adaptive filtering, Factor 2 main effect)
3. Compression ratio degrades performance (Factor 3 main effect)
4. SpecKV (full method) outperforms best single-component config
5. Interaction: learnable filtering benefits more from phase preservation than fixed LP

### 9.4 Target N and Compute Cost

**Target: N=60, Option A (19 configs)**

- Power = 0.90 for Holm-5 corrected medium effects (d=0.5, Wilcoxon)
- Power = 0.97 for uncorrected medium effects
- Minimum detectable effect: d=0.36 (raw, power=0.80), d=0.44 (Holm-5, power=0.80)
- Compute: ~7,747 GPU-hours, ~72.6 days wall on 8x H200
- Buffer: ~17 days for crashes, restarts, pilot, and analysis

Compute breakdown by N:

| N | GPU-hours | Wall days (8x H200) | Power (Holm-5, d=0.5) |
|---|-----------|---------------------|------------------------|
| 10 | 2,383 | 12.4 | 0.16 |
| 15 | 3,539 | 18.4 | 0.26 |
| 20 | 4,695 | 24.5 | 0.37 |
| 30 | 7,007 | 36.5 | 0.56 |
| 50 | 11,613 | 60.6 | 0.83 |
| 60 | 13,869 | 72.6 | 0.90 |
| 69 | 16,025 | 83.5 | 0.94 |

Option B (31 configs) does NOT fit in 90 days at the required N:
- N=55 (power=0.80, Holm-5, Wilcoxon): 108.7 days — over budget
- N=45 (max affordable): power=0.78 — below 0.80 threshold

### 9.5 Literature Justification

This tiered approach is endorsed by:

- **REFORMS checklist** (Roberts et al., Science Advances 2026): "reporting uncertainty is better than performing statistical tests alone"
- **Colas et al. (2018)**: bootstrap tests should not be used with fewer than 20 samples; N should exceed power analysis prescriptions
- **NeurIPS 2026 Paper Checklist**: requires error bars and multiple seeds, does not mandate full-study correction
- **"When +1% Is Not Enough" (arXiv:2511.19794)**: conservative paired bootstrap protocol endorses effect-size-with-CI reporting for small gains under tight compute budgets
- **Bouthillier et al. (2021)**: vary sources of randomness, report variance, adapt N to observed variance

### 9.6 Statistical Tests

| Comparison Type | Test | Notes |
|-----------------|------|-------|
| Pairwise config comparisons | Wilcoxon signed-rank test | Paired across seeds; nonparametric; no normality assumption |
| Factorial main effects + interactions | ART ANOVA (Aligned Rank Transform) | Nonparametric factorial; handles interactions |
| Effect size | Cohen's d (paired) | Reported with 95% bootstrap CI (10,000 resamples) |
| Multiple comparison (primary) | Holm-Bonferroni (k=3-5) | Applied to pre-registered hypotheses only |
| Multiple comparison (secondary) | None | Raw p-values reported as exploratory |

### 9.7 Additional Planned Factors

Beyond the core 3-factor design, the following factors will be added for the publication:

- **Base model variants:** Test generalization across model families (e.g., Llama-3.1-8B, Mistral-7B, Qwen-7B). These enter the exploratory tier (effect sizes + CIs, no Holm correction).
- **KV cache window size extrapolations:** Test compression behavior at context lengths beyond training (e.g., 16K → 64K → 256K). Also exploratory tier.

These factors increase the total comparison count, reinforcing the decision to restrict Holm correction to pre-registered primary hypotheses.

---

## 10. Action Items

- [ ] Implement IWDFTTransform in src/spectral/transform.py
- [ ] Add configs C13-C18 YAMLs (Option A)
- [ ] Implement unbiased DCT/FFT variants (Option B, if pursuing publication)
- [ ] Add IWDFT tests to tests/test_freqkv_transforms.py
- [ ] Update EXPERIMENT_SPECIFICATION.md Section 3 with expanded matrix
- [ ] Update src/stats/analyze.py for 3x2 or 5x2 ART ANOVA
- [ ] Update Manim SpecKV scene to show IWDFT recursive update
- [ ] Run Option A N=1 evals for C13-C18 (6 new configs)
- [ ] Run pilot (3-5 seeds) to estimate observed effect sizes
- [ ] Compute required N from pilot effect sizes (target: N=60)
- [ ] Pre-register 3-5 main hypotheses before full study launch
- [ ] Commit to Option A vs Option B based on pilot + compute budget
- [ ] Update Section 3.4 with finalized N and design decision after pilot
