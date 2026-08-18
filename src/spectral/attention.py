"""Spectral attention: LlamaAttention with KV-cache compression.

This module applies spectral KV-cache compression to Llama-3 attention layers
by subclassing LlamaAttention and overriding the forward pass.

The compression follows the FreqKV paradigm (arXiv:2505.00570, ICLR 2026):
  1. Compute Q, K, V projections as normal.
  2. Apply rotary position embeddings to Q, K.
  3. [SPECTRAL] Transform K, V to spectral domain (DCT or FFT).
  4. [SPECTRAL] Apply filter (fixed low-pass or learnable mask).
  5. [SPECTRAL] Truncate to gamma fraction of coefficients.
  6. [SPECTRAL] Reconstruct K, V from compressed spectral representation.
  7. Repeat KV heads for GQA (8 KV heads -> 32 query heads).
  8. Compute standard attention: softmax(Q @ K^T / sqrt(d)) @ V.
  9. Output projection.

The spectral-domain cache achieves O(gamma * N) storage instead of O(N).
During training, the model learns to operate with the lossy reconstructed K/V.
The learnable filter (if present) is updated via backpropagation.

For GQA: Llama-3.1-8B uses 32 query heads and 8 KV heads. Compression is
applied only to the 8 KV heads. Query heads are never compressed.
"""

from __future__ import annotations

import logging
import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .cache import CompressionConfig, SpectralKVCache
from .transform import SpectralTransform, DCTTransform, FFTTransform
from .filter import SpectralFilter, FixedLowPassFilter, LearnableSpectralFilter

logger = logging.getLogger(__name__)


def apply_spectral_compression(
    model: nn.Module,
    config: CompressionConfig,
) -> nn.Module:
    """Apply spectral KV-cache compression to a Llama model.

    Wraps each attention layer's forward method to insert spectral compression
    between K/V projection and attention computation. The spectral caches
    (containing learnable filter parameters) are registered as submodules so
    their parameters are included in the model's parameter list for training.

    Args:
        model: A LlamaForCausalLM model.
        config: Compression configuration (transform type, filter type, gamma).

    Returns:
        The model (modified in-place) with spectral compression applied.
    """
    if config.is_baseline:
        logger.info("Baseline config: no spectral compression applied")
        return model

    logger.info(
        f"Applying spectral compression: transform={config.transform_type}, "
        f"filter={config.filter_type}, gamma={config.gamma}"
    )

    # Find all attention layers: LlamaForCausalLM -> model.model.layers[i].self_attn
    layers = model.model.layers
    num_layers = len(layers)
    logger.info(f"Found {num_layers} transformer layers")

    for i, layer in enumerate(layers):
        attn = layer.self_attn

        # Read architecture constants from the attention module
        num_kv_heads = attn.num_key_value_heads
        head_dim = attn.head_dim

        # Create spectral cache for this layer
        spectral_cache = SpectralKVCache(config, num_kv_heads, head_dim)

        # Register the cache as a submodule so its parameters are tracked
        attn.add_module("spectral_cache", spectral_cache)

        # Store original forward and wrap it
        original_forward = attn.forward

        def make_wrapped_forward(
            attn_module: nn.Module,
            cache: SpectralKVCache,
            orig_fwd,
            layer_idx: int,
        ):
            """Create a closure that has direct access to the attn module."""

            def wrapped_forward(
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                past_key_value=None,
                output_attentions: bool = False,
                use_cache: bool = False,
                **kwargs,
            ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple]]:
                return _spectral_forward(
                    attn_module=attn_module,
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    spectral_cache=cache,
                    layer_idx=layer_idx,
                    **kwargs,
                )

            return wrapped_forward

        attn.forward = make_wrapped_forward(attn, spectral_cache, original_forward, i)

    logger.info(f"Spectral compression applied to {num_layers} layers")
    return model


def _spectral_forward(
    attn_module: nn.Module,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    past_key_value=None,
    output_attentions: bool = False,
    use_cache: bool = False,
    spectral_cache: SpectralKVCache = None,
    layer_idx: int = 0,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple]]:
    """Reimplemented LlamaAttention.forward with spectral KV compression.

    This follows the standard Llama attention computation, inserting spectral
    compression after K/V projections and rotary embeddings.

    Handles both older transformers (rotary_emb on the attention module) and
    newer versions (position_embeddings passed as kwargs).
    """
    from transformers.models.llama.modeling_llama import (
        apply_rotary_pos_emb,
        repeat_kv,
    )

    bsz, q_len, _ = hidden_states.shape

    # 1. Project hidden_states to Q, K, V
    query_states = attn_module.q_proj(hidden_states)
    key_states = attn_module.k_proj(hidden_states)
    value_states = attn_module.v_proj(hidden_states)

    num_q_heads = attn_module.num_heads
    num_kv_heads = attn_module.num_key_value_heads
    head_dim = attn_module.head_dim

    # Reshape to [B, num_heads, S, head_dim]
    query_states = query_states.view(bsz, q_len, num_q_heads, head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)

    # 2. Apply rotary position embeddings
    # Handle different transformers versions:
    #   - Newer (>=4.46): position_embeddings=(cos, sin) passed as kwarg
    #   - Older (<4.46): rotary_emb module on the attention layer
    position_embeddings = kwargs.get("position_embeddings", None)
    if position_embeddings is not None:
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    elif hasattr(attn_module, "rotary_emb") and attn_module.rotary_emb is not None:
        cos, sin = attn_module.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    else:
        logger.warning_once(
            f"Layer {layer_idx}: No rotary embeddings found. "
            f"Pass position_embeddings=(cos, sin) or ensure rotary_emb is set."
        )

    # 3. [SPECTRAL] Compress K, V into spectral domain
    #    Transform -> Filter -> Truncate -> Store compressed coefficients
    spectral_cache.compress(key_states, value_states)

    # 4. [SPECTRAL] Reconstruct K, V from compressed spectral representation
    #    This is the lossy reconstruction that the model learns to work with
    key_states, value_states = spectral_cache.reconstruct(target_seq_len=q_len)

    # 5. Repeat KV heads for GQA (8 KV heads -> 32 query heads)
    key_states = repeat_kv(key_states, num_q_heads // num_kv_heads)
    value_states = repeat_kv(value_states, num_q_heads // num_kv_heads)

    # 6. Compute attention: Q @ K^T / sqrt(d)
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)

    # Apply attention mask (causal mask from HF is [B, 1, 1, S])
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    # Softmax in float32 for numerical stability
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

    # Apply attention to values
    attn_output = torch.matmul(attn_weights, value_states)

    # 7. Reshape and output projection
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, num_q_heads * head_dim)
    attn_output = attn_module.o_proj(attn_output)

    # Return in HF's expected format: (output, attn_weights, past_key_value)
    if output_attentions:
        return attn_output, attn_weights, past_key_value
    return attn_output, None, past_key_value


def get_spectral_caches(model: nn.Module) -> list[SpectralKVCache]:
    """Extract all spectral caches from a model (for inspection or logging).

    Args:
        model: A LlamaForCausalLM with spectral compression applied.

    Returns:
        List of SpectralKVCache objects, one per layer.
    """
    caches = []
    for layer in model.model.layers:
        attn = layer.self_attn
        if hasattr(attn, "spectral_cache"):
            caches.append(attn.spectral_cache)
    return caches


def get_compression_stats(model: nn.Module) -> list[dict]:
    """Get compression statistics from all layers.

    Args:
        model: A LlamaForCausalLM with spectral compression applied.

    Returns:
        List of dicts with per-layer compression stats.
    """
    stats = []
    for i, cache in enumerate(get_spectral_caches(model)):
        stats.append({
            "layer": i,
            "compression_ratio": cache.get_compression_ratio(),
            "cache_size_bytes": cache.get_cache_size_bytes(),
            "is_spectral": cache._is_spectral,
            **cache.compression_stats,
        })
    return stats


def get_learnable_filter_params(model: nn.Module) -> list[nn.Parameter]:
    """Get all learnable filter parameters from the model.

    Used for LoRA modules_to_save configuration and for inspecting
    the learned frequency masks.

    Args:
        model: A LlamaForCausalLM with spectral compression applied.

    Returns:
        List of learnable filter parameter tensors.
    """
    params = []
    for cache in get_spectral_caches(model):
        params.extend(cache.get_learnable_parameters())
    return params


def reset_all_caches(model: nn.Module) -> None:
    """Reset all spectral KV caches in the model.

    Call between sequences during evaluation to clear cached states.
    """
    for cache in get_spectral_caches(model):
        cache.reset()


# Backward compatibility with the old API
class CompressedAttention:
    """Deprecated: use apply_spectral_compression() instead.

    Kept for backward compatibility with older code that references
    CompressedAttention. This class is a no-op wrapper.
    """

    def __init__(self, config: CompressionConfig, num_kv_heads: int = 8, head_dim: int = 128):
        self.config = config
        self.spectral_cache = SpectralKVCache(config, num_kv_heads, head_dim)

    def apply_to_model(self, model: nn.Module) -> nn.Module:
        return apply_spectral_compression(model, self.config)
