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

    Implements the ART procedure from Wobbrock et al. (2011), "The Aligned
    Rank Transform for Nonparametric Factorial Analyses Using Only ANOVA
    Procedures" (CHI '11). The procedure:

    1. Compute aligned observations for each effect by removing the estimated
       effects of all *other* factors and interactions.
    2. Rank the aligned observations (average ranks for ties).
    3. Run standard one-way ANOVA (F-test) on the ranks.

    This correctly detects interaction effects -- the core research question
    of whether the benefit of learnable filtering depends on transform type --
    which the previous per-factor Kruskal-Wallis approach could not.

    Alignment formulas for a 2-factor design (A = transform, B = filter):
      Grand mean:              y_bar
      A marginal mean:         y_Ai   = mean over all B for level A_i
      B marginal mean:         y_Bj   = mean over all A for level B_j
      Cell mean:               y_AiBj = mean for (A_i, B_j)

      Aligned for A:    Y'_A  = Y - y_AiBj + y_Ai
      Aligned for B:    Y'_B  = Y - y_AiBj + y_Bj
      Aligned for A×B:  Y'_AB = Y - y_Ai - y_Bj + y_bar

    The analysis is run per benchmark (collapsing across gamma levels) and
    also per (benchmark, gamma) combination for finer-grained insight.
    Kruskal-Wallis per-factor results are retained as a sanity-check fallback.
    """
    results = {
        "main_effects": {},
        "interaction": {},
        "per_gamma": {},
        "kruskal_wallis_fallback": {},
        "method": "aligned_rank_transform (Wobbrock et al. 2011)",
    }

    # Filter to non-baseline configs (the 2x2 design space)
    factorial_df = df[
        (df["transform"].isin(["dct", "fft"])) &
        (df["filter"].isin(["fixed", "learnable"]))
    ].copy()

    if factorial_df.empty:
        logger.warning("No factorial design data found for ART ANOVA")
        return results

    # ------------------------------------------------------------------
    # Helper: run ART for a single subset of data
    # ------------------------------------------------------------------
    def _run_art(sub_df: pd.DataFrame, label: str) -> dict:
        """Run the 3-effect ART (A, B, A×B) on a data subset.

        Returns a dict with keys 'A', 'B', 'AB', each mapping to a result
        dict or None if insufficient data.
        """
        out = {}

        if len(sub_df) < 4:
            return out

        # Compute means
        grand_mean = sub_df["value"].mean()
        transform_means = sub_df.groupby("transform")["value"].mean()
        filter_means = sub_df.groupby("filter")["value"].mean()
        cell_means = sub_df.groupby(["transform", "filter"])["value"].mean()

        # Aligned observations
        def align_A(row):
            cell = cell_means.get((row["transform"], row["filter"]))
            marg = transform_means.get(row["transform"])
            if cell is None or marg is None or pd.isna(cell) or pd.isna(marg):
                return float("nan")
            return row["value"] - cell + marg

        def align_B(row):
            cell = cell_means.get((row["transform"], row["filter"]))
            marg = filter_means.get(row["filter"])
            if cell is None or marg is None or pd.isna(cell) or pd.isna(marg):
                return float("nan")
            return row["value"] - cell + marg

        def align_AB(row):
            marg_A = transform_means.get(row["transform"])
            marg_B = filter_means.get(row["filter"])
            if marg_A is None or marg_B is None or pd.isna(marg_A) or pd.isna(marg_B):
                return float("nan")
            return row["value"] - marg_A - marg_B + grand_mean

        sub_df = sub_df.copy()
        sub_df["_aligned_A"] = sub_df.apply(align_A, axis=1)
        sub_df["_aligned_B"] = sub_df.apply(align_B, axis=1)
        sub_df["_aligned_AB"] = sub_df.apply(align_AB, axis=1)

        # Drop rows with NaN aligned values (incomplete cells)
        sub_df = sub_df.dropna(subset=["_aligned_A", "_aligned_B", "_aligned_AB"])

        if len(sub_df) < 4:
            return out

        # Rank the aligned observations (average ranks for ties)
        ranked_A = stats.rankdata(sub_df["_aligned_A"].values)
        ranked_B = stats.rankdata(sub_df["_aligned_B"].values)
        ranked_AB = stats.rankdata(sub_df["_aligned_AB"].values)

        # --- Effect A (transform) ---
        groups_A = [
            ranked_A[sub_df["transform"].values == t]
            for t in sorted(sub_df["transform"].unique())
        ]
        if len(groups_A) >= 2 and all(len(g) >= 2 for g in groups_A):
            f_stat, p_val = stats.f_oneway(*groups_A)
            grand_rank = ranked_A.mean()
            ss_between = sum(len(g) * (g.mean() - grand_rank) ** 2 for g in groups_A)
            ss_total = ((ranked_A - grand_rank) ** 2).sum()
            ss_error = ss_total - ss_between
            denom = ss_between + ss_error
            eta_sq = float(ss_between / denom) if denom > 0 else 0.0
            out["A"] = {
                "factor": "transform",
                "effect_label": label,
                "method": "ART ANOVA (F-test on aligned ranks)",
                "f_statistic": float(f_stat),
                "p_value": float(p_val),
                "significant": p_val < ALPHA,
                "partial_eta_squared": eta_sq,
                "df_between": len(groups_A) - 1,
                "df_within": len(sub_df) - len(groups_A),
                "n_observations": len(sub_df),
            }

        # --- Effect B (filter) ---
        groups_B = [
            ranked_B[sub_df["filter"].values == f]
            for f in sorted(sub_df["filter"].unique())
        ]
        if len(groups_B) >= 2 and all(len(g) >= 2 for g in groups_B):
            f_stat, p_val = stats.f_oneway(*groups_B)
            grand_rank = ranked_B.mean()
            ss_between = sum(len(g) * (g.mean() - grand_rank) ** 2 for g in groups_B)
            ss_total = ((ranked_B - grand_rank) ** 2).sum()
            ss_error = ss_total - ss_between
            denom = ss_between + ss_error
            eta_sq = float(ss_between / denom) if denom > 0 else 0.0
            out["B"] = {
                "factor": "filter",
                "effect_label": label,
                "method": "ART ANOVA (F-test on aligned ranks)",
                "f_statistic": float(f_stat),
                "p_value": float(p_val),
                "significant": p_val < ALPHA,
                "partial_eta_squared": eta_sq,
                "df_between": len(groups_B) - 1,
                "df_within": len(sub_df) - len(groups_B),
                "n_observations": len(sub_df),
            }

        # --- Effect A×B (interaction) ---
        interaction_labels = sub_df["transform"].values + ":" + sub_df["filter"].values
        unique_interactions = sorted(set(interaction_labels))
        groups_AB = [
            ranked_AB[interaction_labels == g]
            for g in unique_interactions
        ]
        if len(groups_AB) >= 2 and all(len(g) >= 2 for g in groups_AB):
            f_stat, p_val = stats.f_oneway(*groups_AB)
            grand_rank = ranked_AB.mean()
            ss_between = sum(len(g) * (g.mean() - grand_rank) ** 2 for g in groups_AB)
            ss_total = ((ranked_AB - grand_rank) ** 2).sum()
            ss_error = ss_total - ss_between
            denom = ss_between + ss_error
            eta_sq = float(ss_between / denom) if denom > 0 else 0.0
            out["AB"] = {
                "factor": "transform:filter",
                "effect_label": label,
                "method": "ART ANOVA (F-test on aligned ranks)",
                "f_statistic": float(f_stat),
                "p_value": float(p_val),
                "significant": p_val < ALPHA,
                "partial_eta_squared": eta_sq,
                "df_between": len(groups_AB) - 1,
                "df_within": len(sub_df) - len(groups_AB),
                "n_observations": len(sub_df),
                "levels": unique_interactions,
            }

        return out

    # ------------------------------------------------------------------
    # Main analysis: per benchmark, collapsing across gamma
    # ------------------------------------------------------------------
    for benchmark in factorial_df["benchmark"].unique():
        bench_df = factorial_df[factorial_df["benchmark"] == benchmark]
        label = f"{benchmark}"
        art = _run_art(bench_df, label)

        if "A" in art:
            results["main_effects"][f"transform_{benchmark}"] = art["A"]
        if "B" in art:
            results["main_effects"][f"filter_{benchmark}"] = art["B"]
        if "AB" in art:
            results["interaction"][f"transform_x_filter_{benchmark}"] = art["AB"]

    # ------------------------------------------------------------------
    # Per-gamma analysis: does the interaction vary with compression?
    # ------------------------------------------------------------------
    for benchmark in factorial_df["benchmark"].unique():
        for gamma in [0.50, 0.22, 0.01]:
            subset = factorial_df[
                (factorial_df["benchmark"] == benchmark) &
                (factorial_df["gamma"] == gamma)
            ]
            if len(subset) < 4:
                continue
            label = f"{benchmark}_gamma{gamma}"
            art = _run_art(subset, label)
            key = label
            entry = {}
            if "A" in art:
                entry["transform"] = art["A"]
            if "B" in art:
                entry["filter"] = art["B"]
            if "AB" in art:
                entry["interaction"] = art["AB"]
            if entry:
                results["per_gamma"][key] = entry

    # ------------------------------------------------------------------
    # Kruskal-Wallis fallback (sanity check, cannot detect interactions)
    # ------------------------------------------------------------------
    for benchmark in factorial_df["benchmark"].unique():
        bench_df = factorial_df[factorial_df["benchmark"] == benchmark]
        for factor in ["transform", "filter"]:
            groups = []
            labels = []
            for level in sorted(bench_df[factor].unique()):
                vals = bench_df[bench_df[factor] == level]["value"].values
                if len(vals) >= 3:
                    groups.append(vals)
                    labels.append(level)
            if len(groups) >= 2:
                stat, pval = stats.kruskal(*groups)
                results["kruskal_wallis_fallback"][f"{factor}_{benchmark}"] = {
                    "factor": factor,
                    "benchmark": benchmark,
                    "levels": labels,
                    "statistic": float(stat),
                    "p_value": float(pval),
                    "significant": pval < ALPHA,
                    "note": "Kruskal-Wallis per-factor; cannot detect interactions",
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
