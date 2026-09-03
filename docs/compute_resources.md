# CSCE 823 — Compute Resource Scoping
## NVIDIA H200 (141 GB HBM3e, 4.8 TB/s) — AFIT CCR AI Cluster

**Date:** 2026-08-19  
**Author:** Samuel Chadwick  
**Revised:** 2026-08-22 — Removed wall-clock estimates (preliminary; not benchmarked)

---

## 1. Hardware Overview

| Specification               | Value                          |
|-----------------------------|--------------------------------|
| GPU                         | NVIDIA H200 (SXM5)            |
| HBM3e memory per GPU        | 141 GB                         |
| Memory bandwidth per GPU    | 4.8 TB/s                       |
| FP16/BF16 compute per GPU   | 1,979 TFLOPS                  |
| Interconnect                | NVLink 4.0 (900 GB/s)         |

---

## 2. Model Memory Footprint

### Llama-3.1-8B-Instruct

| Component                  | Size (BF16)   | Notes                         |
|---------------------------|---------------|-------------------------------|
| Model weights             | ~16 GB        | 8B params x 2 bytes           |
| LoRA adapters (r=8)       | ~21 MB        | Negligible                    |
| Optimizer states (AdamW)  | ~32 GB        | 2x model weights (ZeRO-2 splits) |
| Gradient states           | ~16 GB        | ZeRO-2 partitions across GPUs |
| Activations (8K ctx, bs=8)| ~40 GB        | Gradient checkpointing reduces ~60% |
| KV cache (8K, bs=8)       | ~4 GB         | 32 layers x 8 heads x 8192 x 128 x 2 (K+V) x 2 bytes |
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

### Conclusion: H200 is Sufficient for Training

Training Llama-3.1-8B with LoRA r=8 is comfortable on 4x or 8x H200.
The DeepSpeed config is auto-selected based on detected GPU count.

---

## 3. Training Compute Budget

### Per-Configuration Training

| Phase             | Dataset         | Steps/Epochs     | Batch Size |
|-------------------|-----------------|------------------|------------|
| Phase 1: CPT      | RedPajama       | 1,000 steps      | 64         |
| Phase 2: SFT      | LongAlpaca-16k  | 5 epochs         | 64         |

### All 13 Configurations

| Item                    | Value          |
|------------------------|----------------|
| Configurations         | 13 (12 + baseline) |
| Training               | Sequential (each config uses all GPUs via DeepSpeed) |

The baseline (C00) is trained through the same pipeline as C01-C12
(no compression applied, but identical data, LoRA config, and hyperparams)
to keep the independent variable clean.

---

## 4. Evaluation Compute Budget

### Per-Seed Evaluation

| Benchmark    | Samples | Method              |
|-------------|---------|---------------------|
| PG-19       | 100     | Sliding window PPL  |
| Proof-pile  | 100     | Sliding window PPL  |
| LongBench   | 14 tasks| Generation (256 tok)|
| Efficiency  | 1       | 128 token gen       |

### Evaluation Matrix

| Item                         | N=1 (class)  | N=30 (archive)  |
|-----------------------------|--------------|-----------------|
| Configurations              | 13           | 13              |
| Seeds per config            | 1            | 30              |
| Total eval runs             | 13           | 390             |
| Parallelization             | 4-way (1 GPU each) | 4-way or 8-way |

> The N=30 publication protocol is documented in the
> [spectral-kv archive](https://github.com/samwick07/spectral-kv).

---

## 5. Storage Requirements

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

## 6. Network Bandwidth

| Operation                     | Data Transfer     | Impact                |
|------------------------------|-------------------|-----------------------|
| Model download (HF)          | ~16 GB            | One-time              |
| Dataset download             | ~50 GB            | One-time              |
| DeepSpeed all-reduce         | ~32 GB/step       | NVLink 900 GB/s: negligible |
| Checkpoint save              | ~16 GB/config     | NVMe                  |

Network is not a bottleneck thanks to NVLink 4.0 and CCR high-speed internet.

---

## 7. Recommended Execution Plan

### Phase A: Environment Setup
1. Verify GPU visibility: `nvidia-smi`
2. `bash scripts/setup_env.sh` — creates venv, installs deps, downloads model + datasets
3. HuggingFace login (request Llama-3.1 access if not already approved)
4. Smoke test: `bash scripts/smoke_test.sh`

### Phase B: Pilot Run
1. `bash scripts/run.sh --pilot` — trains C00, C07, C10 at gamma=0.50, evals 5 seeds each
2. Verify aggregation and statistical analysis pipeline

### Phase C: Full Training
1. `bash scripts/run.sh --phase train` — trains all 13 configurations sequentially
2. Monitor via `bash scripts/monitor.sh --watch` or W&B dashboard

### Phase D: Full Evaluation
1. `bash scripts/run.sh --phase eval` — evaluates 13 configs across GPUs
2. Monitor via `bash scripts/monitor.sh --watch`

### Phase E: Analysis
1. `bash scripts/run.sh --phase analyze`
2. N=1 -> point-estimate table: `results/point/point_table.{csv,md}`
3. N=30 (archive) -> 7-step pipeline: aggregate + analyze

Note: The orchestrator auto-resumes after crashes. If the workspace
loses power, simply re-run `bash scripts/run.sh` and it will skip
completed work and resume from the last checkpoint.
