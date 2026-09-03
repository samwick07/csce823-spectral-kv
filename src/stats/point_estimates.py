"""N=1 point-estimate table for the class-project version of the experiment.

Reads seed-0 (or single-seed) raw result JSONs from results/raw/ and produces
results/point/point_table.csv and point_table.md: one row per config with
per-benchmark values and delta vs. the C00 baseline.

No significance testing: with a single seed there is no distribution to test.
The N=30 publication version (aggregate.py + analyze.py, 7-step pipeline)
lives in the spectral-kv archive repository.

Usage:
    python -m src.stats.point_estimates [--seed 0] [--results-dir results]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from .experiment_matrix import EXPERIMENT_MATRIX

logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results")
POINT_DIR = RESULTS_DIR / "point"

# (csv column, benchmark key, json path within that benchmark's results)
BENCH_FIELDS: list[tuple[str, str, tuple[str, ...]]] = [
    ("pg19_mean_ppl", "pg19", ("mean_perplexity",)),
    ("proofpile_mean_ppl", "proof_pile", ("mean_perplexity",)),
    ("longbench_overall", "longbench", ("overall_mean",)),
    ("eff_decode_latency_ms", "efficiency", ("decoding_latency_ms_per_token",)),
    ("eff_peak_kv_gb", "efficiency", ("peak_kv_memory_gb",)),
    ("eff_overhead_pct", "efficiency", ("compression_overhead_pct",)),
]

# (flat per-benchmark file, benchmark key) -- fallback when all_results.json
# is absent. These files are the same dicts dumped individually by
# run_experiment.py, so they use the benchmark-internal (flat) schema.
BENCH_FILES: list[tuple[str, str]] = [
    ("pg19.json", "pg19"),
    ("proof_pile.json", "proof_pile"),
    ("longbench.json", "longbench"),
    ("efficiency.json", "efficiency"),
]

HEADER_NOTE = (
    "N=1 point estimate (single seed). No significance tests -- see the "
    "spectral-kv archive repository for the N=30 distributional analysis."
)


def _load_json(path: Path) -> dict | None:
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Could not load {path}: {e}")
        return None


def _dig(data: dict, path: tuple[str, ...]):
    cur = data
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _load_config_benchmarks(config_dir: Path) -> dict:
    """Load benchmark result dicts for one config/seed directory.

    Prefers the canonical all_results.json (nested: {benchmark: {...}}),
    falling back to the individual flat benchmark files if it is absent.
    Returns {benchmark_key: result_dict} for whatever is available.
    """
    all_results = _load_json(config_dir / "all_results.json")
    if isinstance(all_results, dict):
        benches = {
            key: all_results[key]
            for key, _ in BENCH_FILES
            if isinstance(all_results.get(key), dict)
        }
        return benches
    benches = {}
    for filename, key in BENCH_FILES:
        data = _load_json(config_dir / filename)
        if isinstance(data, dict):
            benches[key] = data
    return benches


def build_point_table(
    raw_dir: str | Path = RESULTS_DIR / "raw",
    seed: int = 0,
) -> pd.DataFrame:
    """Build the per-config point-estimate DataFrame."""
    raw_dir = Path(raw_dir)
    rows = []
    for config in EXPERIMENT_MATRIX:
        config_dir = raw_dir / config.config_id / f"seed_{seed}"
        if not config_dir.exists():
            logger.warning(f"Missing results for {config.config_id} seed_{seed}: {config_dir}")
            continue
        benches = _load_config_benchmarks(config_dir)
        row = {"config_id": config.config_id, "variant": config.variant_name,
               "gamma": config.gamma, "is_baseline": config.is_baseline}
        for column, bench_key, json_path in BENCH_FIELDS:
            data = benches.get(bench_key)
            row[column] = _dig(data, json_path) if data else None
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Delta vs. baseline C00 (None where baseline or value missing)
    base = df[df["config_id"] == "C00"]
    if not base.empty:
        for column, _, _ in BENCH_FIELDS:
            base_val = base[column].iloc[0]
            delta = df[column] - base_val if pd.notna(base_val) else None
            df[f"delta_{column}"] = delta
    return df


def _fmt_md(df: pd.DataFrame) -> str:
    lines = [
        "# Spectral KV-Cache Compression -- Point Estimates",
        "",
        "> " + HEADER_NOTE,
        "",
        df.to_markdown(index=False, floatfmt=".4g"),
        "",
    ]
    return "\n".join(lines)


def write_point_table(df: pd.DataFrame, output_dir: str | Path = POINT_DIR) -> dict:
    """Write point_table.csv and point_table.md. Returns output paths."""
    if df.empty:
        raise FileNotFoundError("No point-estimate rows found -- run at least one config first.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "point_table.csv"
    md_path = output_dir / "point_table.md"
    df.to_csv(csv_path, index=False)
    md_path.write_text(_fmt_md(df))
    logger.info(f"Point table written: {csv_path}, {md_path}")
    return {"csv": str(csv_path), "md": str(md_path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0, help="Seed directory to read (default 0)")
    parser.add_argument("--results-dir", type=str, default=str(RESULTS_DIR))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    raw_dir = Path(args.results_dir) / "raw"
    df = build_point_table(raw_dir=raw_dir, seed=args.seed)
    paths = write_point_table(df)
    print(f"\nPoint table (N=1, seed {args.seed}): {paths['md']}")
    print(df[["config_id", "variant", "gamma", "pg19_mean_ppl",
              "proofpile_mean_ppl", "longbench_overall",
              "eff_decode_latency_ms", "eff_peak_kv_gb"]].to_string(index=False))


if __name__ == "__main__":
    main()
