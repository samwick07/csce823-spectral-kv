# P1-3: HuggingFace Dataset Verification Checklist

**Status:** READY FOR CODER WORKSPACE EXECUTION

This document lists every HuggingFace dataset used by the spectral-kv
project. Each must be verified accessible from the AFIT CCR Coder
workspace (coder.afitcdn.org) before launching experiments.

## Verification Procedure

Run the following Python snippet on the Coder workspace for each dataset.
All commands assume `HF_TOKEN` is set in the environment.

```python
from datasets import load_dataset

# Replace DATASET_ID, SPLIT, and SUBSET as needed
ds = load_dataset("DATASET_ID", name="SUBSET", split="SPLIT", trust_remote_code=True)
print(f"OK: {len(ds)} samples, columns={ds.column_names}")
```

## Datasets

### 1. togethercomputer/RedPajama-Data-1T-Sample
- **Used in:** `src/training/train_redpajama.py` (LoRA fine-tuning)
- **Split:** `train`
- **Subset:** None (single config)
- **trust_remote_code:** Not needed (standard parquet dataset)
- **Verification:**
  ```python
  ds = load_dataset("togethercomputer/RedPajama-Data-1T-Sample", split="train")
  print(f"OK: {len(ds)} samples, columns={ds.column_names}")
  ```
- **Expected columns:** `text`, `meta` (dict with `redpajama_subset` key)
- **Notes:** ~1B tokens sample. Large download (~10GB). Pre-download
  before experiment launch to avoid bottleneck.

### 2. Yukang/LongAlpaca-16k
- **Used in:** `src/training/train_longalpaca.py` (long-context LoRA)
- **Split:** `train`
- **Subset:** None
- **trust_remote_code:** Not needed
- **Verification:**
  ```python
  ds = load_dataset("Yukang/LongAlpaca-16k", split="train")
  print(f"OK: {len(ds)} samples, columns={ds.column_names}")
  ```
- **Expected columns:** `question`, `answer`, `length` (or similar)
- **Notes:** ~16k long-context instruction pairs. Used for Phase 2
  fine-tuning with extended context windows.

### 3. deepmind/pg19
- **Used in:** `src/eval/pg19.py` (perplexity evaluation)
- **Split:** `test`
- **Subset:** None
- **trust_remote_code:** Not needed
- **Verification:**
  ```python
  ds = load_dataset("deepmind/pg19", split="test")
  print(f"OK: {len(ds)} samples, columns={ds.column_names}")
  ```
- **Expected columns:** `text`, `short_book_title`, `publication_date`
- **Notes:** ~100 full books in test split. Used for sliding-window
  perplexity (window=256 tokens). No token limit on raw text.

### 4. EleutherAI/proof-pile
- **Used in:** `src/eval/proof_pile.py` (mathematical perplexity)
- **Split:** `test`
- **Subset:** None
- **trust_remote_code:** True (required)
- **Verification:**
  ```python
  ds = load_dataset("EleutherAI/proof-pile", split="test", trust_remote_code=True)
  print(f"OK: {len(ds)} samples, columns={ds.column_names}")
  ```
- **Expected columns:** `text`, `meta` (dict with source info)
- **Notes:** Mathematical/symbolic text. Critical for evaluating
  whether learnable filters retain high-frequency components needed
  for symbolic content.

### 5. THUDM/LongBench
- **Used in:** `src/eval/longbench.py` (14-task long-context benchmark)
- **Split:** `test`
- **Subset:** Task name (one of 14: narrativeqa, qasper, multifieldqa_en,
  hotpotqa, 2wikimqa, musique, gov_report, qmsum, multi_news, trec,
  triviaqa, samsum, passage_retrieval_en, lcc)
- **trust_remote_code:** True (required)
- **Verification (all 14 tasks):**
  ```python
  tasks = [
      "narrativeqa", "qasper", "multifieldqa_en",
      "hotpotqa", "2wikimqa", "musique",
      "gov_report", "qmsum", "multi_news",
      "trec", "triviaqa", "samsum",
      "passage_retrieval_en", "lcc",
  ]
  for task in tasks:
      ds = load_dataset("THUDM/LongBench", task, split="test", trust_remote_code=True)
      print(f"OK: {task} - {len(ds)} samples, columns={ds.column_names}")
  ```
- **Expected columns:** `context`, `input`, `answers`, `length`, `dataset`
- **Notes:** Each task is a separate subset. Download all 14 before
  experiment launch. Used with stochastic decoding (temp=0.7, top_p=0.9).

## Pre-Flight Checklist

Before launching the 2x2 factorial experiment:

- [ ] HF_TOKEN set in environment (`export HF_TOKEN=hf_...`)
- [ ] All 5 datasets load successfully (run verification snippets above)
- [ ] All 14 LongBench tasks accessible
- [ ] `trust_remote_code=True` accepted for proof-pile and LongBench
- [ ] Sufficient disk space for dataset caches (~50GB recommended)
- [ ] `datasets` library version >= 2.20 (check: `pip show datasets`)
- [ ] Network access to `huggingface.co` from CCR compute nodes
- [ ] Run `python -m src.tests.smoke_test --skip-eval` to verify model
      loading + compression before full eval runs

## Troubleshooting

- **trust_remote_code error:** Ensure `datasets>=2.16` and run
  `huggingface-cli login` with your HF token.
- **Dataset not found:** Some datasets may have been renamed or moved.
  Search at `https://huggingface.co/datasets` for the current canonical name.
- **Connection timeout from CCR:** Use `HF_DATASETS_OFFLINE=0` and
  pre-download via `datasets-cli download` on the login node, then
  set `HF_DATASETS_CACHE` to the shared storage path.
- **Out of memory during LongBench:** Reduce `max_new_tokens` or
  use `num_samples=N` to subsample.