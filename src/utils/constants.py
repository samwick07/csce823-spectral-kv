"""Project-wide constants: model identifiers, dataset names, etc.

Centralized so model/version changes propagate from one place.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# HuggingFace model identifiers
# ---------------------------------------------------------------------------

# Primary experiment model: Llama-3.1-8B-Instruct
# 32 query heads, 8 KV heads (GQA), 128K native context, 32 layers.
# Used in all 13 experiment configs (C00-C12) and all evaluation scripts.
DEFAULT_MODEL_NAME: str = "meta-llama/Llama-3.1-8B-Instruct"

# Smoke-test model: Llama-3.2-1B-Instruct
# Small enough for rapid integration testing on a single GPU.
# Same architecture family (GQA, LlamaAttention) as the 8B model.
SMOKE_TEST_MODEL_NAME: str = "meta-llama/Llama-3.2-1B-Instruct"
