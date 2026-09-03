"""Evaluation suite for spectral KV-cache compression experiments."""

from .pg19 import evaluate_pg19
from .proof_pile import evaluate_proof_pile
from .longbench import evaluate_longbench
from .metrics import compute_perplexity, compute_efficiency_metrics
from .efficiency import measure_efficiency

__all__ = [
    "evaluate_pg19",
    "evaluate_proof_pile",
    "evaluate_longbench",
    "compute_perplexity",
    "compute_efficiency_metrics",
    "measure_efficiency",
]
