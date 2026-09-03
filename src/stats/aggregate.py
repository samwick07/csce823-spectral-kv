"""Aggregate raw evaluation results across seeds into summary tables.

Reads individual seed result JSONs from results/raw/ and produces
aggregated CSV/JSON in results/aggregated/ ready for statistical analysis.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .experiment_matrix import EXPERIMENT_MATRIX, get_config_by_id

logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results")
RAW_DIR = RESULTS_DIR / "raw"
AGGREGATED_DIR = RESULTS_DIR / "aggregated"

# Number of seeds per config
NUM_SEEDS = 30


def aggregate_results(
    raw_dir: str | Path = RAW_DIR,
    output_dir: str | Path = AGGREGATED_DIR,
    num_seeds: int = NUM_SEEDS,
) -> dict:
    """Aggregate raw per-seed evaluation results into summary tables.

    Expected raw file structure:
        results/raw/{config_id}/seed_{seed}/{benchmark}.json

    Produces:
        results/aggregated/per_seed.csv   — one row per (config, seed, benchmark)
        results/aggregated/summary.csv    — one row per (config, benchmark) with stats
        results/aggregated/summary.json   — same as summary.csv in JSON

    Args:
        raw_dir: Directory containing per-seed raw results.
        output_dir: Directory for aggregated output.
        num_seeds: Expected number of seeds per config.

    Returns:
        Dict with aggregation statistics.
    """
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []

    for config in EXPERIMENT_MATRIX:
        config_dir = raw_dir / config.config_id

        if not config_dir.exists():
            logger.warning(f"No results found for {config.config_id}")
            continue

        for seed_dir in sorted(config_dir.glob("seed_*")):
            seed = int(seed_dir.name.split("_")[1])

            for benchmark_file in sorted(seed_dir.glob("*.json")):
                benchmark_name = benchmark_file.stem
                with open(benchmark_file) as f:
                    data = json.load(f)

                row = {
                    "config_id": config.config_id,
                    "transform": config.transform,
                    "filter": config.filter,
                    "gamma": config.gamma,
                    "variant": config.variant_name,
                    "is_baseline": config.is_baseline,
                    "seed": seed,
                    "benchmark": benchmark_name,
                }

                # Extract primary metric (perplexity for PG-19/Proof-pile, score for LongBench)
                if "mean_perplexity" in data:
                    row["metric"] = "perplexity"
                    row["value"] = data["mean_perplexity"]
                elif "overall_mean" in data:
                    row["metric"] = "accuracy"
                    row["value"] = data["overall_mean"]
                elif "peak_kv_memory_gb" in data:
                    row["metric"] = "peak_memory_gb"
                    row["value"] = data["peak_kv_memory_gb"]
                elif "decoding_latency_ms_per_token" in data:
                    row["metric"] = "latency_ms_per_token"
                    row["value"] = data["decoding_latency_ms_per_token"]
                else:
                    logger.warning(f"Unknown metrics in {benchmark_file}")
                    continue

                all_rows.append(row)

    if not all_rows:
        logger.warning("No raw results found to aggregate")
        return {"configs_found": 0, "total_rows": 0}

    df = pd.DataFrame(all_rows)

    # Save per-seed table
    per_seed_path = output_dir / "per_seed.csv"
    df.to_csv(per_seed_path, index=False)
    logger.info(f"Saved per-seed results: {per_seed_path} ({len(df)} rows)")

    # Aggregate by config x benchmark
    summary_rows = []
    for (config_id, benchmark), group in df.groupby(["config_id", "benchmark"]):
        values = group["value"].values
        config = get_config_by_id(config_id)

        summary_rows.append({
            "config_id": config_id,
            "transform": config.transform if config else None,
            "filter": config.filter if config else None,
            "gamma": config.gamma if config else None,
            "variant": config.variant_name if config else None,
            "is_baseline": config.is_baseline if config else False,
            "benchmark": benchmark,
            "metric": group["metric"].iloc[0],
            "num_seeds": len(values),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "std": float(np.std(values, ddof=1)),
            "sem": float(np.std(values, ddof=1) / np.sqrt(len(values))),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "q25": float(np.percentile(values, 25)),
            "q75": float(np.percentile(values, 75)),
        })

    summary_df = pd.DataFrame(summary_rows)

    summary_csv_path = output_dir / "summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    logger.info(f"Saved summary: {summary_csv_path} ({len(summary_df)} rows)")

    summary_json_path = output_dir / "summary.json"
    with open(summary_json_path, "w") as f:
        json.dump(summary_rows, f, indent=2)

    return {
        "configs_found": df["config_id"].nunique(),
        "total_rows": len(df),
        "seeds_found": int(df["seed"].nunique()),
        "benchmarks": sorted(df["benchmark"].unique().tolist()),
    }
