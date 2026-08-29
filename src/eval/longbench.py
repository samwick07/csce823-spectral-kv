"""LongBench V1 evaluation: 14 tasks across 5 categories.

Categories: single-document QA, multi-document QA, summarization,
few-shot learning, code completion.

Uses stochastic decoding (temperature=0.7, top-p=0.9) to capture
generation variance across seeds.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from transformers import AutoTokenizer

from ..utils.constants import DEFAULT_MODEL_NAME

logger = logging.getLogger(__name__)

# LongBench task definitions
LONGBENCH_TASKS = {
    # Single-document QA
    "narrativeqa": {"category": "single_doc_qa", "max_length": 8192},
    "qasper": {"category": "single_doc_qa", "max_length": 8192},
    "multifieldqa_en": {"category": "single_doc_qa", "max_length": 8192},
    # Multi-document QA
    "hotpotqa": {"category": "multi_doc_qa", "max_length": 8192},
    "2wikimqa": {"category": "multi_doc_qa", "max_length": 8192},
    "musique": {"category": "multi_doc_qa", "max_length": 8192},
    # Summarization
    "gov_report": {"category": "summarization", "max_length": 8192},
    "qmsum": {"category": "summarization", "max_length": 8192},
    "multi_news": {"category": "summarization", "max_length": 8192},
    # Few-shot learning
    "trec": {"category": "few_shot", "max_length": 8192},
    "triviaqa": {"category": "few_shot", "max_length": 8192},
    "samsum": {"category": "few_shot", "max_length": 8192},
    # Code completion
    "passage_retrieval_en": {"category": "code_completion", "max_length": 8192},
    "lcc": {"category": "code_completion", "max_length": 8192},
}


def evaluate_longbench(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
    tasks: list[str] | None = None,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    device: str = "cuda",
    seed: int = 0,
    hf_token: str | None = None,
    longbench_dir: str | None = None,
    num_samples: int | None = None,
    past_key_value=None,
) -> dict:
    """Evaluate model on LongBench V1 benchmark.

    Args:
        model: The language model (possibly with spectral compression).
        tokenizer: Tokenizer (loaded if None).
        model_name: Model name for loading tokenizer if not provided.
        tasks: List of task names to evaluate. None = all 14 tasks.
        max_new_tokens: Maximum generation length.
        temperature: Sampling temperature (0.7 per protocol).
        top_p: Nucleus sampling threshold (0.9 per protocol).
        device: Device to run on.
        seed: Random seed for generation.
        hf_token: HuggingFace token.
        longbench_dir: Path to LongBench evaluation code (for metrics).
        num_samples: If set, evaluate only the first N samples per task
                     (useful for smoke tests). None = full dataset.
        past_key_value: Optional DynamicCache for KV caching during
                        KV caching during generation. If provided,
                        enables K=1 incremental updates (O(N log N) per
                        step instead of O(N^2)).

    Returns:
        Dict with per-task accuracy scores.
    """
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)

    if tasks is None:
        tasks = list(LONGBENCH_TASKS.keys())

    logger.info(f"Evaluating LongBench V1: {len(tasks)} tasks")
    logger.info(f"  temperature={temperature}, top_p={top_p}, seed={seed}")

    torch.manual_seed(seed)

    results = {
        "benchmark": "longbench_v1",
        "tasks": {},
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
    }

    for task_name in tasks:
        task_info = LONGBENCH_TASKS[task_name]
        logger.info(f"  Task: {task_name} ({task_info['category']})")

        try:
            # Load task data from HuggingFace
            from datasets import load_dataset

            # THUDM/LongBench uses a loading script (deprecated in datasets 3+).
            # Xnhyacinth/LongBench is a parquet mirror with the same task names.
            lb_names = [
                ("Xnhyacinth/LongBench", {}),
                ("THUDM/LongBench", {"trust_remote_code": True}),
            ]
            dataset = None
            for lb_name, lb_kwargs in lb_names:
                try:
                    dataset = load_dataset(lb_name, task_name, split="test", **lb_kwargs)
                    break
                except Exception:
                    continue
            if dataset is None:
                raise RuntimeError(f"Could not load LongBench task {task_name}")

            task_scores = []

            for sample_idx, sample in enumerate(dataset):
                if num_samples is not None and sample_idx >= num_samples:
                    break

                context = sample.get("context", "")
                input_text = sample.get("input", "")
                answers = sample.get("answers", [])

                # Build prompt
                prompt = build_longbench_prompt(task_name, context, input_text)

                input_ids = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=task_info["max_length"],
                ).input_ids.to(device)

                # Reset spectral cache between samples
                if past_key_value is not None:
                    past_key_value.reset()

                # Generate with stochastic decoding
                with torch.no_grad():
                    gen_kwargs = dict(
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=temperature,
                        top_p=top_p,
                    )
                    if past_key_value is not None:
                        gen_kwargs["past_key_value"] = past_key_value
                    output = model.generate(input_ids, **gen_kwargs)

                generated = tokenizer.decode(
                    output[0, input_ids.shape[1]:],
                    skip_special_tokens=True,
                )

                # Score the output
                score = score_longbench_output(task_name, generated, answers)
                task_scores.append(score)

            avg_score = sum(task_scores) / max(len(task_scores), 1)
            results["tasks"][task_name] = {
                "category": task_info["category"],
                "num_samples": len(task_scores),
                "mean_score": avg_score,
                "all_scores": task_scores,
            }

            logger.info(f"    Score: {avg_score:.4f}")

        except Exception as e:
            logger.error(f"    FAILED: {e}")
            results["tasks"][task_name] = {"error": str(e)}

    # Overall summary
    all_scores = [
        t["mean_score"] for t in results["tasks"].values() if "mean_score" in t
    ]
    results["overall_mean"] = sum(all_scores) / max(len(all_scores), 1)

    return results


def build_longbench_prompt(task_name: str, context: str, input_text: str) -> str:
    """Build a prompt for a LongBench task.

    Uses the standard LongBench prompt format from THUDM/LongBench.
    """
    task_info = LONGBENCH_TASKS[task_name]
    category = task_info["category"]

    if category in ("single_doc_qa", "multi_doc_qa"):
        prompt = f"Read the following text and answer the question.\n\n{context}\n\nQuestion: {input_text}\n\nAnswer:"
    elif category == "summarization":
        prompt = f"Summarize the following text.\n\n{context}\n\nSummary:"
    elif category == "few_shot":
        prompt = f"{context}\n\n{input_text}"
    elif category == "code_completion":
        prompt = f"Complete the following code.\n\n{context}"
    else:
        prompt = f"{context}\n\n{input_text}"

    return prompt


def score_longbench_output(
    task_name: str,
    prediction: str,
    answers: list[str],
) -> float:
    """Score a LongBench prediction against reference answers.

    Uses the metric appropriate for the task category:
      - QA tasks: F1 score
      - Summarization: ROUGE-L
      - Code completion: Edit similarity
      - Few-shot: Exact match or F1

    This is a simplified scorer. For full evaluation, use the official
    LongBench metrics from THUDM/LongBench.
    """
    if not answers:
        return 0.0

    task_info = LONGBENCH_TASKS[task_name]
    category = task_info["category"]

    if category in ("single_doc_qa", "multi_doc_qa", "few_shot"):
        # F1 score
        return compute_f1(prediction, answers[0])
    elif category == "summarization":
        # ROUGE-L (simplified)
        return compute_rouge_l(prediction, answers[0])
    elif category == "code_completion":
        # Edit similarity (simplified)
        return compute_edit_similarity(prediction, answers[0])
    else:
        return compute_f1(prediction, answers[0])


def compute_f1(prediction: str, reference: str) -> float:
    """Compute token-level F1 score between prediction and reference."""
    pred_tokens = set(prediction.lower().split())
    ref_tokens = set(reference.lower().split())

    if not pred_tokens or not ref_tokens:
        return 0.0

    overlap = pred_tokens & ref_tokens
    if not overlap:
        return 0.0

    precision = len(overlap) / len(pred_tokens)
    recall = len(overlap) / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_rouge_l(prediction: str, reference: str) -> float:
    """Compute simplified ROUGE-L score (LCS-based)."""
    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()

    if not pred_tokens or not ref_tokens:
        return 0.0

    # LCS length
    m, n = len(pred_tokens), len(ref_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_tokens[i - 1] == ref_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs = dp[m][n]
    if lcs == 0:
        return 0.0

    precision = lcs / m
    recall = lcs / n
    return 2 * precision * recall / (precision + recall)


def compute_edit_similarity(prediction: str, reference: str) -> float:
    """Compute simplified edit similarity between prediction and reference."""
    if not prediction or not reference:
        return 0.0

    # Simple character-level edit distance
    m, n = len(prediction), len(reference)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if prediction[i - 1] == reference[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    edit_dist = dp[m][n]
    max_len = max(m, n)
    return 1.0 - edit_dist / max_len if max_len > 0 else 0.0
