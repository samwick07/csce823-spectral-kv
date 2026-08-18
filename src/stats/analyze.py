"""Statistical analysis: 7-step pipeline for the ablation study.

1. Wilcoxon signed-rank: each config vs baseline
2. Friedman test: across all configs at each gamma
3. Nemenyi post-hoc: pairwise comparisons after Friedman
4. Kolmogorov-Smirnov: distribution comparison (config vs baseline)
5. Anderson-Darling: test for normality of each config's distribution
6. ART ANOVA: aligned rank transform for the 2x2 factorial
7. Holm-Bonferroni: multiple comparison correction
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import (
    wilcoxon,
    friedmanchisquare,
    ks_2samp,
    anderson,
)
from scikit_posthocs import posthoc_nemenyi_friedman

from .experiment_matrix import EXPERIMENT_MATRIX, FACTOR_LEVELS, get_config_by_id

logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results")
AGGREGATED_DIR = RESULTS_DIR / "aggregated"

# Holm-Bonferroni alpha
ALPHA = 0.05


def run_full_analysis(
    aggregated_dir: str | Path = AGGREGATED_DIR,
    output_dir: str | Path = RESULTS_DIR,
    alpha: float = ALPHA,
) -> dict:
    """Run the full 7-step statistical analysis pipeline.

    Args:
        aggregated_dir: Directory with per_seed.csv and summary.csv.
        output_dir: Directory to save analysis results.
        alpha: Family-wise error rate for Holm-Bonferroni.

    Returns:
        Dict with all analysis results.
    """
    aggregated_dir = Path(aggregated_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    per_seed_path = aggregated_dir / "per_seed.csv"
    if not per_seed_path.exists():
        raise FileNotFoundError(f"Per-seed data not found: {per_seed_path}. Run aggregate first.")

    df = pd.read_csv(per_seed_path)
    logger.info(f"Loaded {len(df)} per-seed results for analysis")

    results = {}
    num_tests = 0  # For Holm-Bonferroni correction

    # ----------------------------------------------------------------
    # Step 1: Wilcoxon signed-rank test (each config vs baseline)
    # ----------------------------------------------------------------
    logger.info("Step 1: Wilcoxon signed-rank tests vs baseline")
    wilcoxon_results = _wilcoxon_vs_baseline(df)
    results["wilcoxon"] = wilcoxon_results
    num_tests += len(wilcoxon_results)

    # ----------------------------------------------------------------
    # Step 2: Friedman test (across configs at each gamma level)
    # ----------------------------------------------------------------
    logger.info("Step 2: Friedman tests across configs at each gamma")
    friedman_results = _friedman_test(df)
    results["friedman"] = friedman_results

    # ----------------------------------------------------------------
    # Step 3: Nemenyi post-hoc (pairwise after Friedman)
    # ----------------------------------------------------------------
    logger.info("Step 3: Nemenyi post-hoc tests")
    nemenyi_results = _nemenyi_posthoc(df)
    results["nemenyi"] = nemenyi_results
    num_tests += sum(len(v.get("p_values", {})) for v in nemenyi_results.values())

    # ----------------------------------------------------------------
    # Step 4: Kolmogorov-Smirnov test (distribution comparison)
    # ----------------------------------------------------------------
    logger.info("Step 4: Kolmogorov-Smirnov distribution tests")
    ks_results = _ks_test(df)
    results["ks_test"] = ks_results
    num_tests += len(ks_results)

    # ----------------------------------------------------------------
    # Step 5: Anderson-Darling test (normality check)
    # ----------------------------------------------------------------
    logger.info("Step 5: Anderson-Darling normality tests")
    ad_results = _anderson_darling(df)
    results["anderson_darling"] = ad_results

    # ----------------------------------------------------------------
    # Step 6: ART ANOVA (2x2 factorial analysis)
    # ----------------------------------------------------------------
    logger.info("Step 6: ART ANOVA for 2x2 factorial design")
    art_results = _art_anova(df)
    results["art_anova"] = art_results

    # ----------------------------------------------------------------
    # Step 7: Holm-Bonferroni correction
    # ----------------------------------------------------------------
    logger.info(f"Step 7: Holm-Bonferroni correction ({num_tests} tests, alpha={alpha})")
    corrected = _holm_bonferroni(results, num_tests, alpha)
    results["holm_bonferroni"] = corrected

    # Save results
    output_path = output_dir / "statistical_analysis.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Saved analysis to {output_path}")

    return results


def _wilcoxon_vs_baseline(df: pd.DataFrame) -> list[dict]:
    """Wilcoxon signed-rank test: each config vs baseline.

    Requires paired samples (same seeds).
    """
    results = []
    baseline = df[df["config_id"] == "C00"]

    if baseline.empty:
        logger.warning("No baseline (C00) results found for Wilcoxon test")
        return results

    for benchmark in df["benchmark"].unique():
        baseline_bench = baseline[baseline["benchmark"] == benchmark]

        for config_id in sorted(df["config_id"].unique()):
            if config_id == "C00":
                continue

            config_data = df[
                (df["config_id"] == config_id) & (df["benchmark"] == benchmark)
            ]

            # Match on seeds
            common_seeds = set(baseline_bench["seed"]) & set(config_data["seed"])
            if len(common_seeds) < 5:
                continue

            baseline_vals = baseline_bench[
                baseline_bench["seed"].isin(common_seeds)
            ]["value"].values
            config_vals = config_data[
                config_data["seed"].isin(common_seeds)
            ]["value"].values

            # Sort by seed to ensure pairing
            baseline_vals = baseline_bench[
                baseline_bench["seed"].isin(common_seeds)
            ].sort_values("seed")["value"].values
            config_vals = config_data[
                config_data["seed"].isin(common_seeds)
            ].sort_values("seed")["value"].values

            try:
                stat, pval = wilcoxon(config_vals, baseline_vals)
                results.append({
                    "test": "wilcoxon_signed_rank",
                    "config_id": config_id,
                    "benchmark": benchmark,
                    "statistic": float(stat),
                    "p_value": float(pval),
                    "significant_uncorrected": pval < ALPHA,
                    "n_samples": len(common_seeds),
                })
            except ValueError as e:
                logger.warning(f"Wilcoxon failed for {config_id}/{benchmark}: {e}")

    return results


def _friedman_test(df: pd.DataFrame) -> list[dict]:
    """Friedman test: compare all configs at each gamma level."""
    results = []

    for benchmark in df["benchmark"].unique():
        for gamma in [0.50, 0.22, 0.01]:
            gamma_configs = df[
                (df["gamma"] == gamma) & (df["benchmark"] == benchmark)
            ]

            # Need at least 3 configs with paired seeds
            config_ids = sorted(gamma_configs["config_id"].unique())
            if len(config_ids) < 3:
                continue

            # Build matrix: [seeds x configs]
            common_seeds = None
            for cid in config_ids:
                config_seeds = set(
                    gamma_configs[gamma_configs["config_id"] == cid]["seed"]
                )
                common_seeds = (
                    config_seeds if common_seeds is None else common_seeds & config_seeds
                )

            if not common_seeds or len(common_seeds) < 3:
                continue

            common_seeds = sorted(common_seeds)
            groups = []
            for cid in config_ids:
                vals = gamma_configs[
                    (gamma_configs["config_id"] == cid)
                    & (gamma_configs["seed"].isin(common_seeds))
                ].sort_values("seed")["value"].values
                groups.append(vals)

            try:
                stat, pval = friedmanchisquare(*groups)
                results.append({
                    "test": "friedman",
                    "gamma": gamma,
                    "benchmark": benchmark,
                    "config_ids": config_ids,
                    "statistic": float(stat),
                    "p_value": float(pval),
                    "significant": pval < ALPHA,
                    "n_configs": len(config_ids),
                    "n_seeds": len(common_seeds),
                })
            except ValueError as e:
                logger.warning(f"Friedman failed for gamma={gamma}/{benchmark}: {e}")

    return results


def _nemenyi_posthoc(df: pd.DataFrame) -> dict:
    """Nemenyi post-hoc test: pairwise comparisons after Friedman."""
    results = {}

    for benchmark in df["benchmark"].unique():
        for gamma in [0.50, 0.22, 0.01]:
            gamma_configs = df[
                (df["gamma"] == gamma) & (df["benchmark"] == benchmark)
            ]

            config_ids = sorted(gamma_configs["config_id"].unique())
            if len(config_ids) < 3:
                continue

            # Build matrix
            common_seeds = None
            for cid in config_ids:
                config_seeds = set(
                    gamma_configs[gamma_configs["config_id"] == cid]["seed"]
                )
                common_seeds = (
                    config_seeds if common_seeds is None else common_seeds & config_seeds
                )

            if not common_seeds or len(common_seeds) < 3:
                continue

            common_seeds = sorted(common_seeds)
            data_matrix = []
            for cid in config_ids:
                vals = gamma_configs[
                    (gamma_configs["config_id"] == cid)
                    & (gamma_configs["seed"].isin(common_seeds))
                ].sort_values("seed")["value"].values
                data_matrix.append(vals)

            data_matrix = np.array(data_matrix).T  # [seeds x configs]

            try:
                p_values = posthoc_nemenyi_friedman(data_matrix)
                p_values.index = config_ids
                p_values.columns = config_ids
                key = f"{benchmark}_gamma{gamma}"
                results[key] = {
                    "p_values": {
                        str(r): {str(c): float(p_values.loc[r, c]) for c in config_ids}
                        for r in config_ids
                    },
                    "config_ids": config_ids,
                }
            except Exception as e:
                logger.warning(f"Nemenyi failed for gamma={gamma}/{benchmark}: {e}")

    return results


def _ks_test(df: pd.DataFrame) -> list[dict]:
    """Kolmogorov-Smirnov test: compare distributions (config vs baseline)."""
    results = []
    baseline = df[df["config_id"] == "C00"]

    if baseline.empty:
        return results

    for benchmark in df["benchmark"].unique():
        baseline_vals = baseline[baseline["benchmark"] == benchmark]["value"].values
        if len(baseline_vals) < 5:
            continue

        for config_id in sorted(df["config_id"].unique()):
            if config_id == "C00":
                continue

            config_vals = df[
                (df["config_id"] == config_id) & (df["benchmark"] == benchmark)
            ]["value"].values

            if len(config_vals) < 5:
                continue

            stat, pval = ks_2samp(config_vals, baseline_vals)
            results.append({
                "test": "kolmogorov_smirnov",
                "config_id": config_id,
                "benchmark": benchmark,
                "statistic": float(stat),
                "p_value": float(pval),
                "significant_uncorrected": pval < ALPHA,
            })

    return results


def _anderson_darling(df: pd.DataFrame) -> list[dict]:
    """Anderson-Darling test for normality of each config's score distribution."""
    results = []

    for config_id in sorted(df["config_id"].unique()):
        for benchmark in df["benchmark"].unique():
            vals = df[
                (df["config_id"] == config_id) & (df["benchmark"] == benchmark)
            ]["value"].values

            if len(vals) < 8:
                continue

            try:
                result = anderson(vals, dist="norm")
                # Check against 5% significance level
                sig_5 = result.significance_level.tolist().index(5.0)
                is_normal = result.statistic < result.critical_values[sig_5]

                results.append({
                    "test": "anderson_darling",
                    "config_id": config_id,
                    "benchmark": benchmark,
                    "statistic": float(result.statistic),
                    "critical_values": [float(c) for c in result.critical_values],
                    "significance_levels": [float(s) for s in result.significance_level],
                    "is_normal_at_5pct": bool(is_normal),
                })
            except Exception as e:
                logger.warning(f"Anderson-Darling failed for {config_id}/{benchmark}: {e}")

    return results


def _art_anova(df: pd.DataFrame) -> dict:
    """Aligned Rank Transform ANOVA for the 2x2 factorial design.

    Tests main effects and interactions of:
      - Factor A: Transform (DCT vs FFT)
      - Factor B: Filter (Fixed vs Learnable)
      - Interaction: A x B

    Uses a simplified ART implementation via rank transformation.
    For full ART, use the ARTool R package or py-art library.
    """
    results = {"main_effects": {}, "interaction": {}}

    # Filter to non-baseline configs (the 2x2 design space)
    factorial_df = df[
        (df["transform"].isin(["dct", "fft"])) &
        (df["filter"].isin(["fixed", "learnable"]))
    ].copy()

    if factorial_df.empty:
        logger.warning("No factorial design data found for ART ANOVA")
        return results

    for benchmark in factorial_df["benchmark"].unique():
        bench_df = factorial_df[factorial_df["benchmark"] == benchmark]

        # Simplified: use Kruskal-Wallis as a non-parametric alternative
        # for each factor and the interaction
        for factor in ["transform", "filter"]:
            groups = []
            labels = []
            for level in bench_df[factor].unique():
                vals = bench_df[bench_df[factor] == level]["value"].values
                if len(vals) >= 3:
                    groups.append(vals)
                    labels.append(level)

            if len(groups) >= 2:
                stat, pval = stats.kruskal(*groups)
                results["main_effects"][f"{factor}_{benchmark}"] = {
                    "factor": factor,
                    "benchmark": benchmark,
                    "levels": labels,
                    "statistic": float(stat),
                    "p_value": float(pval),
                    "significant": pval < ALPHA,
                }

        # Interaction: create combined factor
        bench_df["interaction"] = bench_df["transform"] + ":" + bench_df["filter"]
        groups = []
        labels = []
        for level in bench_df["interaction"].unique():
            vals = bench_df[bench_df["interaction"] == level]["value"].values
            if len(vals) >= 3:
                groups.append(vals)
                labels.append(level)

        if len(groups) >= 2:
            stat, pval = stats.kruskal(*groups)
            results["interaction"][f"transform_x_filter_{benchmark}"] = {
                "factor": "transform:filter",
                "benchmark": benchmark,
                "levels": labels,
                "statistic": float(stat),
                "p_value": float(pval),
                "significant": pval < ALPHA,
            }

    return results


def _holm_bonferroni(results: dict, num_tests: int, alpha: float) -> dict:
    """Apply Holm-Bonferroni correction to all p-values.

    Collects all p-values from Wilcoxon, KS, Friedman, Nemenyi tests,
    sorts them, and applies the step-down Holm correction.
    """
    # Collect all p-values with their test identifiers
    pvals = []

    for entry in results.get("wilcoxon", []):
        pvals.append({
            "test": "wilcoxon",
            "config_id": entry["config_id"],
            "benchmark": entry["benchmark"],
            "p_value": entry["p_value"],
        })

    for entry in results.get("ks_test", []):
        pvals.append({
            "test": "ks_test",
            "config_id": entry["config_id"],
            "benchmark": entry["benchmark"],
            "p_value": entry["p_value"],
        })

    # Sort by p-value (ascending)
    pvals.sort(key=lambda x: x["p_value"])

    # Apply Holm-Bonferroni: reject H0 for p_i if p_i <= alpha / (m - i + 1)
    m = len(pvals)
    for i, entry in enumerate(pvals):
        rank = i + 1
        adjusted_alpha = alpha / (m - rank + 1)
        entry["holm_rank"] = rank
        entry["adjusted_alpha"] = float(adjusted_alpha)
        entry["significant_corrected"] = entry["p_value"] <= adjusted_alpha

    return {
        "num_tests": m,
        "alpha": alpha,
        "corrected_results": pvals,
    }
