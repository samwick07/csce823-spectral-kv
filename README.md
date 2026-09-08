# Spectral KV-Cache Compression via Complex FFT with Learnable Frequency Selection

A 2×2 factorial ablation of phase preservation and adaptive filtering for
KV-cache compression, run on Llama-3.1-8B-Instruct (LoRA rank 8) across three
compression ratios. CSCE 823 final project (Air Force Institute of
Technology); preliminary work for a forthcoming journal article.

**Status: complete (N=1, seed 0).** All 13 configurations trained and
evaluated; results are on HuggingFace Hub and analyzed in the accompanying
paper (`paper/`).

## Headline Finding: Causal Leakage

The central result is a **causality defect** discovered through an implausible
training-loss anomaly: at 2× compression, all variants *beat the uncompressed
baseline* (92× lower final loss in Phase 1). A lossy transform cannot improve
next-token prediction over an uncompressed baseline. The root cause: the
spectral transform runs over the **full sequence** before truncation, so
reconstructed values at position *t* carry information from future tokens —
bypassing the causal mask (diagnostics in `scripts/leakage_diagnostics.py`).

Consequences for interpreting results:
- **Moderate-compression (2×, 4.5×) comparisons are confounded** — every
  transform/filter comparison at these ratios reflects leak exploitation as
  much as compression quality.
- **100× results are the most trustworthy**: the 1% coefficient retention
  destroys the leaked signal, so those numbers largely measure real
  compression behavior (PG-19 perplexity ≈ 1,845–1,891; LongBench retains
  81–88% of baseline).
- A chunked causal-safe implementation exists (`src/spectral/chunked_cache.py`)
  but is **untested** — re-running the sweep under it is the top future-work
  item.

## Experiment Matrix

13 configurations: 2 (transform) × 2 (filter) × 3 (ratio) + 1 baseline.

| Config | Transform | Filter | γ | Compression |
|--------|-----------|--------|------|-------------|
| C00 | — | — | 1.0 | 1× (baseline) |
| C01–C03 | DCT | Fixed LP | 0.50/0.22/0.01 | 2×/4.5×/100× |
| C04–C06 | DCT | Learnable | 0.50/0.22/0.01 | 2×/4.5×/100× |
| C07–C09 | Complex FFT | Fixed LP | 0.50/0.22/0.01 | 2×/4.5×/100× |
| C10–C12 | Complex FFT | Learnable | 0.50/0.22/0.01 | 2×/4.5×/100× |

- **DCT** (real-valued, discards phase) vs **complex FFT** (rfft, preserves phase)
- **Fixed low-pass** (no-op; truncation does the cut) vs **learnable sigmoid
  mask** (per-layer, per-head; initialized as smooth low-pass decay)
- Training: Phase 1 = RedPajama CPT (1,000 steps, seq 2048); Phase 2 =
  LongAlpaca-12k SFT (5 epochs, seq 16K). DeepSpeed ZeRO-2 on 4× H200.

## Results Summary (seed 0)

| Benchmark | C00 baseline | 2× | 4.5× | 100× |
|-----------|-------------|-----|------|------|
| PG-19 ppl (↓) | 15.88 | 1.07–1.12 (leak) | 2.13–12.11 (leak) | 1,845–1,891 |
| Proof-pile ppl (↓) | 6.14 | 1.13–1.15 (leak) | 2.93–6.93 (leak) | 2,062–2,140 |
| LongBench (↑) | 0.057 | 0.003–0.005 | 0.005–0.021 | 0.046–0.050 |

Retrieval is non-monotonic in compression and tracks final training loss at
Pearson r = 0.97 — consistent with leak exploitation, not compression
quality, driving the moderate-compression numbers. Full data:
`results/all_eval_data.json`; per-benchmark scores in
`results/all_ppl_scores.json` and `results/all_longbench_scores.json`.

## Repository Layout

```
csce823-spectral-kv/
├── src/
│   ├── spectral/          # DCT/FFT transforms, filters, compressed attention,
│   │                      #   cache.py (incremental decode), chunked_cache.py (causal-safe, untested)
│   ├── training/          # RedPajama + LongAlpaca fine-tuning pipelines
│   ├── eval/              # PG-19, Proof-pile, LongBench, efficiency
│   ├── stats/             # Aggregation, point estimates, 7-step pipeline
│   ├── utils/             # Config, constants, W&B, checkpointing
│   ├── orchestrator.py    # Crash-safe 4-phase driver (Train→Eval→Analyze→Exfil)
│   └── run_experiment.py  # Single-config entry point
├── configs/               # 13 experiment YAMLs + DeepSpeed ZeRO-2 configs
├── scripts/
│   ├── run.sh             # Orchestrator entry (tmux, --seeds, --pilot, --status)
│   ├── setup_env.sh       # venv, deps, model+dataset download, smoke tests
│   ├── leakage_diagnostics.py  # 3 repeatable causal-leak tests (CPU, deterministic)
│   └── monitor.sh / smoke_test.sh / exfil.sh / watchdog.sh
├── results/               # all_eval_data.json, per-benchmark scores, efficiency,
│                          #   orchestrator state, figures/, raw/, aggregated/
├── tests/                 # 43 spectral unit tests
├── docs/                  # Experiment spec, research positioning, lessons learned
└── paper/                 # IEEE-format final paper + figures + report_statistics.py
```

## Reproducing

```bash
# Environment (H200 cluster, HF_TOKEN + WANDB_API_KEY required)
bash scripts/setup_env.sh

# Full pipeline (13 configs; --seeds 0 for the N=1 run reproduced here)
bash scripts/run.sh --seeds 0

# Leakage diagnostics only (CPU-only, no GPU needed)
python scripts/leakage_diagnostics.py --test 1   # cross-position sensitivity
python scripts/leakage_diagnostics.py --test 2   # attention-output sensitivity
python scripts/leakage_diagnostics.py --test 3   # predictive correlation

# Paper: verify every reported number against results/
python paper/report_statistics.py               # 51 checks, exits nonzero on failure
```

Orchestrator state lives in `results/orchestrator_state.json` (atomic writes);
crashes resume from DeepSpeed checkpoints with no manual intervention — re-run
the same command.

## Data & Artifacts

- **HuggingFace Hub:** `samwick07/spectral-kv-results` — packaged results,
  LoRA adapters, filter-mask arrays, reproducibility manifest
- **Weights & Biases:** `samwick07-afit/csce823-spectral-kv` — training logs
- **Paper:** `paper/` — IEEE Transactions-format writeup with all figures
  regenerable via `paper/make_figures.py` and every numeric claim gated by
  `paper/report_statistics.py`

## Key Design Decisions

- **Eager/SDPA attention**: FlashAttention-2 is incompatible with the
  overridden `LlamaAttention.forward`; manual SDPA is used throughout.
- **DeepSpeed auto-selection**: GPU count at runtime picks the matching
  ZeRO-2 config; Phase 2 (16K sequences) uses the `_longctx` variants
  (micro-batch 2).
- **Seed count is a runtime argument**: 1 seed → point estimates; 2+ seeds →
  full 7-step statistical pipeline (Wilcoxon, Friedman + Nemenyi,
  Kolmogorov–Smirnov, Anderson–Darling, ART ANOVA, Holm–Bonferroni).
- **DCT initialization detail**: the learnable filter's logits start as a
  smooth low-pass decay so early training behaves like the fixed filter.

## Author

Samuel Chadwick — Air Force Institute of Technology

## License

Apache-2.0
