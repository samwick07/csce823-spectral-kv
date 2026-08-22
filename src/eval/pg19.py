"""PG-19 evaluation: long-context language modeling perplexity.

PG-19 contains full books. Perplexity is computed with a 256-token sliding window.
"""

from __future__ import annotations

import logging

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from .metrics import compute_sliding_window_perplexity
from ..utils.constants import DEFAULT_MODEL_NAME

logger = logging.getLogger(__name__)

WINDOW_SIZE = 256


def evaluate_pg19(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
    num_samples: int = 100,
    window_size: int = WINDOW_SIZE,
    device: str = "cuda",
    seed: int = 0,
    hf_token: str | None = None,
) -> dict:
    """Evaluate model on PG-19 benchmark.

    Args:
        model: The language model (possibly with spectral compression).
        tokenizer: Tokenizer (loaded if None).
        model_name: Model name for loading tokenizer if not provided.
        num_samples: Number of books to evaluate.
        window_size: Sliding window size (default 256 per protocol).
        device: Device to run on.
        seed: Random seed for sample selection.
        hf_token: HuggingFace token.

    Returns:
        Dict with perplexity statistics.
    """
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)

    logger.info(f"Evaluating PG-19: {num_samples} books, window={window_size}")

    # Load PG-19 test split
    # deepmind/pg19 uses a loading script (deprecated in datasets 3+).
    # emozilla/pg19-test is a parquet mirror that works without scripts.
    pg19_names = [
        "emozilla/pg19-test",
        "deepmind/pg19",
    ]
    dataset = None
    for pg_name in pg19_names:
        try:
            dataset = load_dataset(pg_name, split="test")
            logger.info(f"Loaded PG-19 from {pg_name}: {len(dataset)} rows")
            break
        except Exception as e:
            logger.warning(f"Could not load {pg_name}: {e}")
    if dataset is None:
        raise RuntimeError(f"Could not load PG-19. Tried: {pg19_names}")

    # Select samples
    torch.manual_seed(seed)
    indices = torch.randperm(len(dataset))[:num_samples].tolist()

    perplexities = []

    for idx in indices:
        text = dataset[idx]["text"]
        input_ids = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=16384,
        )["input_ids"]

        if input_ids.shape[1] < window_size * 2:
            continue  # Skip very short texts

        ppl = compute_sliding_window_perplexity(
            model=model,
            input_ids=input_ids,
            window_size=window_size,
            device=device,
        )
        perplexities.append(ppl)

    perplexities_tensor = torch.tensor(perplexities)

    results = {
        "benchmark": "pg19",
        "num_samples": len(perplexities),
        "window_size": window_size,
        "mean_perplexity": float(perplexities_tensor.mean()),
        "median_perplexity": float(perplexities_tensor.median()),
        "std_perplexity": float(perplexities_tensor.std()),
        "min_perplexity": float(perplexities_tensor.min()),
        "max_perplexity": float(perplexities_tensor.max()),
        "all_perplexities": perplexities,
    }

    logger.info(
        f"PG-19 results: mean={results['mean_perplexity']:.2f}, "
        f"median={results['median_perplexity']:.2f}"
    )

    return results
