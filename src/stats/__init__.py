"""Statistical analysis framework for the 2x2 factorial ablation study."""

from .aggregate import aggregate_results
from .analyze import run_full_analysis
from .experiment_matrix import EXPERIMENT_MATRIX, get_config_by_id

__all__ = [
    "aggregate_results",
    "run_full_analysis",
    "EXPERIMENT_MATRIX",
    "get_config_by_id",
]
