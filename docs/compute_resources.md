# CSCE 823 — Compute Resource Scoping
## 8× NVIDIA H200 (141 GB HBM3e, 4.8 TB/s) — AFIT CCR AI Cluster

**Date:** 2026-08-19
**Author:** Samuel Chadwick

---

## 1. Hardware Overview

| Specification               | Value                          |
|-----------------------------|--------------------------------|
| GPU                         | NVIDIA H200 (SXM5)            |
| GPUs available              | 8                              |
| HBM3e memory per GPU        | 141 GB                         |
| Aggregate HBM               | 1,128 GB (1.1 TB)             |
| Memory bandwidth per GPU    | 4.8 TB/s                       |
| FP16/BF16 compute per GPU   | 1,979 TFLOPS                  |
| Aggregate FP16/BF16         | 15,832 TFLOPS (~15.8 PFLOPS)  |
| Interconnect                | NVLink 4.0 (900 GB/s)         |

---

## 2. Model Memory Footprint

### Llama-3-8B-Instruct

| Component                  | Size (BF16)   | Notes                         |
|---------------------------|---------------|-------------------------------|
| Model weights             | ~16 GB        | 8B params × 2 bytes           |
| LoRA adapters (r=8)       | ~21 MB        | Negligible                    |
| Optimizer states (AdamW)  | ~32 GB        | 2× model weights (ZeRO-2 splits) |
| Gradient states           | ~16 GB        | ZeRO-2 partitions across GPUs |
| Activations (8K ctx, bs=8)| ~40 GB        | Gradient checkpointing reduces ~60% |
| KV cache (8K, bs=8)       | ~4 GB         | 32 layers × 8 heads × 8192 × 128 × 2 (K+V) × 2 bytes |
| **Total per GPU (ZeRO-2)**| **~55-65 GB** | Well within 141 GB H200      |
| **Headroom**              | **~76-86 GB** | Available for larger batches  |

### With 16K context (LongAlpaca SFT)

| Component                  | Size (BF16)   |
|---------------------------|---------------|
| Model + LoRA              | ~16 GB        |
| Optimizer (ZeRO-2)        | ~4 GB/GPU     |
| Activations (16K, bs=2)   | ~60 GB        |
| KV cache (16K, bs=2)      | ~2 GB         |
| **Total per GPU**         | **~82 GB**    |
| **Headroom**              | **~59 GB**    |

### Conclusion: 8× H200 is Overkill for Training (Intentionally)

Training Llama-3-8B with LoRA r=8 is comfortable on 4× H200. With 8× H200, you have
two options:

1. **Use all 8 GPUs for training** — faster throughput, larger effective batch
2. **Use 4 GPUs for training, 4 for parallel evaluation** — maximizes pipeline throughput

Recommendation: Use all 8 for training (ZeRO-2 across 8 GPUs gives ~55 GB/GPU with
large headroom for batch size tuning). Then switch to parallel eval mode using all 8
GPUs independently (1 config per GPU, seeds run sequentially within each).

---

## 3. Training Compute Budget

### Per-Configuration Training

| Phase             | Dataset         | Steps/Epochs     | Batch Size | Est. Time (8 GPU) |
|-------------------|-----------------|------------------|------------|-------------------|
| Phase 1: CPT      | RedPajama       | 1,000 steps      | 64         | ~1.5-2 hours      |
| Phase 2: SFT      | LongAlpaca-16k  | 5 epochs         | 64         | ~3-4 hours        |
| **Total per config** |              |                  |            | **~5-6 hours**    |

### All 14 Configurations

| Item                    | Value          |
|------------------------|----------------|
| Configurations         | 14 (13 + baseline) |
| Training time per config | ~5-6 hours   |
| Sequential training    | ~70-84 hours   |
| **Wall clock (sequential)** | **~3-3.5 days** |

Training must be sequential (each config uses all 8 GPUs via DeepSpeed).
However, the baseline (C00) doesn't need spectral compression training — it
can use the stock Llama-3-8B-Instruct checkpoint, saving ~6 hours.

**Adjusted training estimate: ~64-78 hours (~2.7-3.3 days)**

---

## 4. Evaluation Compute Budget

### Per-Seed Evaluation

| Benchmark    | Samples | Method              | Est. Time (1 GPU) |
|-------------|---------|---------------------|-------------------|
| PG-19       | 100     | Sliding window PPL  | ~20 min           |
| Proof-pile  | 100     | Sliding window PPL  | ~20 min           |
| LongBench   | 14 tasks| Generation (256 tok)| ~60 min           |
| Efficiency  | 1       | 128 token gen       | ~2 min            |
| **Total per seed** | |                     | **~1.7 hours**    |

### Full Evaluation Matrix

| Item                         | Value              |
|-----------------------------|--------------------|
| Configurations              | 14                 |
| Seeds per config            | 30                 |
| Total eval runs             | 420                |
| Time per eval run (1 GPU)   | ~1.7 hours         |
| **Total GPU-hours (eval)**  | **~714 GPU-hours** |

### Parallelization Across 8 GPUs

| Strategy                    | Wall Clock         |
|-----------------------------|--------------------|
| 1 GPU (sequential)          | ~714 hours (~30 days) |
| 4 GPUs                      | ~179 hours (~7.5 days) |
| **8 GPUs (recommended)**    | **~89 hours (~3.7 days)** |

With 8 GPUs, distribute 14 configs across GPUs (2 per GPU, 1 GPU gets 1 config + baseline):
- GPU 0: C00 (baseline, fast) + C01
- GPU 1: C02 + C03
- GPU 2: C04 + C05
- GPU 3: C06 + C07
- GPU 4: C08 + C09
- GPU 5: C10 + C11
- GPU 6: C12
- GPU 7: Reserved (efficiency, overflow, monitoring)

---

## 5. Total Compute Budget Summary

| Phase         | GPU-Hours   | Wall Clock (8 GPU)   |
|--------------|-------------|----------------------|
| Training     | ~500-624    | ~2.7-3.3 days        |
| Evaluation   | ~714        | ~3.7 days            |
| Analysis     | ~2          | ~10 min              |
| **Total**    | **~1,216-1,340** | **~6.4-7.0 days** |

### Pilot Run (Recommended First)

Before the full sweep, run a pilot with 3 configs × 5 seeds:

| Item                    | Value          |
|------------------------|----------------|
| Configs                | C00, C07, C10  |
| Seeds per config       | 5              |
| Eval runs              | 15             |
| Training (3 configs)   | ~15-18 hours   |
| Evaluation (15 runs)   | ~25.5 hours    |
| **Pilot total**        | **~41 GPU-hours** |
| **Pilot wall clock**   | **~8 hours**   |

This validates the pipeline end-to-end and catches bugs before committing
to the full 1,200+ GPU-hour run.

---

## 6. Storage Requirements

| Item                          | Size          |
|------------------------------|---------------|
| Base model (Llama-3-8B)      | ~16 GB        |
| LoRA checkpoints (14 configs)| ~300 MB total |
| HuggingFace dataset cache     | ~50 GB        |
| Evaluation results (JSON)    | ~500 MB       |
| W&B / TensorBoard logs       | ~5 GB         |
| **Total storage**            | **~72 GB**    |

H200 nodes typically have 3-7 TB NVMe local storage. Storage is not a constraint.

---

## 7. Network Bandwidth

| Operation                     | Data Transfer     | Impact                |
|------------------------------|-------------------|-----------------------|
| Model download (HF)          | ~16 GB            | One-time, ~5 min on CCR network |
| Dataset download             | ~50 GB            | One-time, ~15 min     |
| DeepSpeed all-reduce (8 GPU) | ~32 GB/step       | NVLink 900 GB/s: negligible |
| Checkpoint save              | ~16 GB/config     | NVMe: ~30 sec         |

Network is not a bottleneck thanks to NVLink 4.0 and CCR's high-speed internet.

---

## 8. Power and Thermal

| Specification               | Value              |
|-----------------------------|--------------------|
| H200 TDP per GPU            | 700W               |
| 8× H200 peak power          | 5,600W (5.6 kW)    |
| Cooling                     | Liquid-cooled SXM5 |
| Thermal concern             | None — H200 SXM5 is datacenter-grade |

---

## 9. Recommended Execution Plan

### Phase A: Environment Setup (Day 0)
1. Verify 8× H200 visibility: `nvidia-smi`
2. Create venv, install requirements.txt
3. HuggingFace login (request Llama-3 access if not already approved)
4. Pre-download datasets: RedPajama, LongAlpaca-16k, PG-19, Proof-pile, LongBench
5. Smoke test: load model, run 10-step training, run 1-sample eval

### Phase B: Pilot Run (Day 1)
1. Train C00 (baseline), C07 (FFT+fixed), C10 (FFT+learnable) at gamma=0.50
2. Evaluate with 5 seeds each
3. Verify aggregation and statistical analysis pipeline
4. Estimated time: ~8 hours

### Phase C: Full Training (Days 2-4)
1. Train all 14 configurations sequentially
2. ~5-6 hours per config, ~70-84 hours total
3. Monitor via W&B

### Phase D: Full Evaluation (Days 5-8)
1. Distribute 14 configs across 8 GPUs (2 per GPU)
2. 30 seeds per config, ~1.7 hours per seed
3. ~89 hours wall clock
4. Monitor via results/ directory

### Phase E: Analysis (Day 8)
1. Aggregate results: `python -m src.stats.aggregate`
2. Run 7-step statistical analysis: `python -m src.stats.analyze`
3. Generate figures and tables

**Total elapsed time: ~8 days** (with pilot, training, eval, analysis)

---

## 10. Contingency: 4× H200

If only 4 H200s are available (instead of 8):

| Phase         | 4× H200 Wall Clock | 8× H200 Wall Clock |
|--------------|--------------------|--------------------|
| Training     | ~5.4-6.6 days      | ~2.7-3.3 days      |
| Evaluation   | ~7.5 days          | ~3.7 days          |
| **Total**    | **~13 days**       | **~7 days**        |

The project is feasible on 4× H200 but takes roughly 2× longer. The 8× H200
configuration is recommended for the 30-seed statistical analysis to complete
within a reasonable timeframe.
