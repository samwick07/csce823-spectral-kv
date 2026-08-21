# CSCE 823 — Compute Resource Scoping
## 4× NVIDIA H200 (141 GB HBM3e, 4.8 TB/s) — AFIT CCR AI Cluster

**Date:** 2026-08-19
**Author:** Samuel Chadwick

---

## 1. Hardware Overview

| Specification               | Value                          |
|-----------------------------|--------------------------------|
| GPU                         | NVIDIA H200 (SXM5)            |
| GPUs available              | 4                              |
| HBM3e memory per GPU        | 141 GB                         |
| Aggregate HBM               | 564 GB                         |
| Memory bandwidth per GPU    | 4.8 TB/s                       |
| FP16/BF16 compute per GPU   | 1,979 TFLOPS                  |
| Aggregate FP16/BF16         | 7,916 TFLOPS (~7.9 PFLOPS)   |
| Interconnect                | NVLink 4.0 (900 GB/s)         |

---

## 2. Model Memory Footprint

### Llama-3.1-8B-Instruct

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

### Conclusion: 4× H200 is Sufficient for Training

Training Llama-3.1-8B with LoRA r=8 is comfortable on 4× H200. With 4× H200, the
model trains with large headroom for batch size tuning.

Recommendation: Use all 4 for training (ZeRO-2 across 4 GPUs gives ~65 GB/GPU with
large headroom). Then switch to parallel eval mode using all 4
GPUs independently (1 config per GPU, seeds run sequentially within each).

---

## 3. Training Compute Budget

### Per-Configuration Training

| Phase             | Dataset         | Steps/Epochs     | Batch Size | Est. Time (4 GPU) |
|-------------------|-----------------|------------------|------------|-------------------|
| Phase 1: CPT      | RedPajama       | 1,000 steps      | 64         | ~4-5 hours        |
| Phase 2: SFT      | LongAlpaca-16k  | 5 epochs         | 64         | ~6-8 hours        |
| **Total per config** |              |                  |            | **~9-12 hours**   |

### All 13 Configurations

| Item                    | Value          |
|------------------------|----------------|
| Configurations         | 13 (12 + baseline) |
| Training time per config | ~9-12 hours   |
| Sequential training    | ~117-156 hours |
| **Wall clock (sequential)** | **~4.9-6.5 days** |

Training must be sequential (each config uses all 4 GPUs via DeepSpeed).
However, the baseline (C00) doesn't need spectral compression training — it
can use the stock Llama-3.1-8B-Instruct checkpoint, saving ~12 hours.

**Adjusted training estimate: ~114-156 hours (~4.8-6.5 days)**

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
| Configurations              | 13                 |
| Seeds per config            | 30                 |
| Total eval runs             | 390                |
| Time per eval run (1 GPU)   | ~1.7 hours         |
| **Total GPU-hours (eval)**  | **~663 GPU-hours** |

### Parallelization Across 4 GPUs

| Strategy                    | Wall Clock         |
|-----------------------------|--------------------|
| 1 GPU (sequential)          | ~663 hours (~27.6 days) |
| 2 GPUs                      | ~332 hours (~13.8 days) |
| **4 GPUs (recommended)**    | **~166 hours (~6.9 days)** |

With 4 GPUs, distribute 13 configs across GPUs:
- GPU 0: C00 (baseline, fast) + C01 + C02 + C03
- GPU 1: C04 + C05 + C06
- GPU 2: C07 + C08 + C09
- GPU 3: C10 + C11 + C12

---

## 5. Total Compute Budget Summary

| Phase         | GPU-Hours   | Wall Clock (4 GPU)    |
|--------------|-------------|----------------------|
| Training     | ~480-600    | ~4.8-6.5 days        |
| Evaluation   | ~663        | ~6.9 days            |
| Analysis     | ~2          | ~10 min              |
| **Total**    | **~1,145-1,265** | **~11.7-13.4 days** |

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
| Base model (Llama-3.1-8B)     | ~16 GB        |
| LoRA checkpoints (13 configs)| ~300 MB total |
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
1. Verify 4× H200 visibility: `nvidia-smi`
2. `bash scripts/setup_env.sh` — creates venv, installs deps, downloads model + datasets
3. HuggingFace login (request Llama-3.1 access if not already approved)
4. Smoke test: `bash scripts/smoke_test.sh`

### Phase B: Pilot Run (Day 1)
1. `bash scripts/run.sh --pilot` — trains C00, C07, C10 at gamma=0.50, evals 5 seeds each
2. Verify aggregation and statistical analysis pipeline
3. Estimated time: ~22 hours

### Phase C: Full Training (Days 2-7)
1. `bash scripts/run.sh --phase train` — trains all 13 configurations sequentially
2. ~9-12 hours per config, ~117-156 hours total
3. Monitor via `bash scripts/monitor.sh --watch` or W&B dashboard

### Phase D: Full Evaluation (Days 8-15)
1. `bash scripts/run.sh --phase eval` — evaluates 13 configs across 4 GPUs
2. 30 seeds per config, ~1.7 hours per seed
3. ~166 hours wall clock
4. Monitor via `bash scripts/monitor.sh --watch`

### Phase E: Analysis (Day 15)
1. `bash scripts/run.sh --phase analyze`
2. Aggregates results: `python -m src.stats.aggregate`
3. Runs 7-step statistical analysis: `python -m src.stats.analyze`
4. Generates figures and tables

**Total elapsed time: ~15 days** (with pilot, training, eval, analysis)

Note: The orchestrator auto-resumes after crashes. If the workspace
loses power, simply re-run `bash scripts/run.sh` and it will skip
completed work and resume from the last checkpoint.

---

## 10. Contingency: 2× H200

If only 2 H200s are available (instead of 4):

| Phase         | 2× H200 Wall Clock | 4× H200 Wall Clock |
|--------------|--------------------|--------------------|
| Training     | ~9.6-13 days       | ~4.8-6.5 days      |
| Evaluation   | ~15 days           | ~7.5 days          |
| **Total**    | **~25 days**       | **~14 days**       |

The project is feasible on 2× H200 but takes roughly 2× longer.
