"""Experiment matrix: all 14 configurations (4 variants x 3 ratios + baseline).

The 2x2 factorial design:
  Factor A: Transform type    {DCT, Complex FFT}
  Factor B: Filter type       {Fixed low-pass, Learnable mask}
  Three compression ratios:   gamma in {0.50, 0.22, 0.01}
  Plus one uncompressed baseline (gamma=1.0)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ExperimentConfig:
    """Single experiment configuration in the ablation matrix."""

    config_id: str
    transform: str          # "dct" or "fft" or "none"
    filter: str             # "fixed" or "learnable" or "none"
    gamma: float            # Compression ratio (1.0 = no compression)
    description: str
    is_baseline: bool = False

    @property
    def variant_name(self) -> str:
        if self.is_baseline:
            return "baseline"
        return f"{self.transform}_{self.filter}"

    @property
    def compression_factor(self) -> float:
        """How much the KV cache is compressed (1/gamma)."""
        return 1.0 / self.gamma if self.gamma > 0 else float("inf")


# Full experiment matrix: 13 compressed configs + 1 baseline
EXPERIMENT_MATRIX: list[ExperimentConfig] = [
    # Baseline (no compression)
    ExperimentConfig(
        config_id="C00",
        transform="none",
        filter="none",
        gamma=1.0,
        description="Uncompressed baseline (no spectral compression)",
        is_baseline=True,
    ),

    # DCT + Fixed Low-Pass (FreqKV baseline)
    ExperimentConfig("C01", "dct", "fixed", 0.50, "DCT + Fixed LP, gamma=0.50 (2x compression)"),
    ExperimentConfig("C02", "dct", "fixed", 0.22, "DCT + Fixed LP, gamma=0.22 (4.5x compression)"),
    ExperimentConfig("C03", "dct", "fixed", 0.01, "DCT + Fixed LP, gamma=0.01 (100x compression)"),

    # DCT + Learnable Filter
    ExperimentConfig("C04", "dct", "learnable", 0.50, "DCT + Learnable, gamma=0.50 (2x compression)"),
    ExperimentConfig("C05", "dct", "learnable", 0.22, "DCT + Learnable, gamma=0.22 (4.5x compression)"),
    ExperimentConfig("C06", "dct", "learnable", 0.01, "DCT + Learnable, gamma=0.01 (100x compression)"),

    # FFT + Fixed Low-Pass
    ExperimentConfig("C07", "fft", "fixed", 0.50, "FFT + Fixed LP, gamma=0.50 (2x compression)"),
    ExperimentConfig("C08", "fft", "fixed", 0.22, "FFT + Fixed LP, gamma=0.22 (4.5x compression)"),
    ExperimentConfig("C09", "fft", "fixed", 0.01, "FFT + Fixed LP, gamma=0.01 (100x compression)"),

    # FFT + Learnable Filter (full proposed method)
    ExperimentConfig("C10", "fft", "learnable", 0.50, "FFT + Learnable, gamma=0.50 (2x compression)"),
    ExperimentConfig("C11", "fft", "learnable", 0.22, "FFT + Learnable, gamma=0.22 (4.5x compression)"),
    ExperimentConfig("C12", "fft", "learnable", 0.01, "FFT + Learnable, gamma=0.01 (100x compression)"),
]


def get_config_by_id(config_id: str) -> Optional[ExperimentConfig]:
    """Look up an experiment configuration by its ID (e.g., 'C01')."""
    for cfg in EXPERIMENT_MATRIX:
        if cfg.config_id == config_id:
            return cfg
    return None


def get_configs_by_variant(transform: str, filter_type: str) -> list[ExperimentConfig]:
    """Get all configurations for a given variant."""
    return [
        cfg for cfg in EXPERIMENT_MATRIX
        if cfg.transform == transform and cfg.filter == filter_type
    ]


def get_configs_by_gamma(gamma: float) -> list[ExperimentConfig]:
    """Get all configurations at a given compression ratio."""
    return [cfg for cfg in EXPERIMENT_MATRIX if cfg.gamma == gamma]


# Factor level labels for ANOVA / reporting
FACTOR_LEVELS = {
    "transform": {"dct": "DCT (real)", "fft": "Complex FFT (phase-preserving)"},
    "filter": {"fixed": "Fixed low-pass", "learnable": "Learnable mask"},
    "gamma": {0.50: "0.50 (2x)", 0.22: "0.22 (4.5x)", 0.01: "0.01 (100x)"},
}
