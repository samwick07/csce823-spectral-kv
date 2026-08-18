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

# Smoke-test model: same as the primary model.
# On H200 (141 GB) the 8B model loads in seconds; running smoke tests on
# the actual experiment model eliminates any cross-model variable.
# Override per-run with: --model <hf-id>
SMOKE_TEST_MODEL_NAME: str = DEFAULT_MODEL_NAME
