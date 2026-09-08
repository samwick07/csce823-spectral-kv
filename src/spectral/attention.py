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

Generation uses K=1 incremental caching via SpectralDynamicCache:
  - Prefill: compress full prompt K/V into spectral domain.
  - Decode: append one token, reconstruct old + new, recompress.
  This makes generation O(N log N) per step instead of O(N^2).
"""

from __future__ import annotations

import logging
import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .cache import CompressionConfig, SpectralKVCache, SpectralDynamicCache
from .transform import SpectralTransform, DCTTransform, FFTTransform
from .filter import SpectralFilter, FixedLowPassFilter, LearnableSpectralFilter

logger = logging.getLogger(__name__)

# Try to import FlashAttention-2 (Fix 2)
try:
    from flash_attn import flash_attn_func
    _HAS_FLASH_ATTN = True
except ImportError:
    _HAS_FLASH_ATTN = False
    logger.debug("flash_attn not available. Using manual attention.")

# Try to import SDPA as a fallback for FA2
try:
    from torch.nn.functional import scaled_dot_product_attention as _sdpa
    _HAS_SDPA = True
except ImportError:
    _HAS_SDPA = False


def _compute_attention(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    use_flash: bool = True,
) -> torch.Tensor:
    """Compute attention with FlashAttention-2 if available, else manual.

    Fix 2: After spectral reconstruction, K/V are standard tensors.
    Use FA2 for fused Q@K^T + softmax + @V to get 2-4x speedup.
    Falls back to SDPA, then to manual attention.

    Args:
        query_states: [B, num_q_heads, S_q, head_dim]
        key_states: [B, num_q_heads, S_kv, head_dim] (already repeated for GQA)
        value_states: [B, num_q_heads, S_kv, head_dim] (already repeated for GQA)
        attention_mask: [B, 1, 1, S_kv] or None
        use_flash: If True and FA2 is available, use it.
    """
    bsz, num_heads, q_len, head_dim = query_states.shape
    kv_len = key_states.shape[-2]

    # Path 1: FlashAttention-2 (fastest)
    # FA2 expects [B, S, H, D] layout (not [B, H, S, D])
    if use_flash and _HAS_FLASH_ATTN and q_len > 1:
        try:
            q_fa2 = query_states.transpose(1, 2)  # [B, S_q, H, D]
            k_fa2 = key_states.transpose(1, 2)    # [B, S_kv, H, D]
            v_fa2 = value_states.transpose(1, 2)  # [B, S_kv, H, D]
            output = flash_attn_func(q_fa2, k_fa2, v_fa2, causal=True)
            return output.transpose(1, 2)  # back to [B, H, S_q, D]
        except Exception as e:
            logger.debug(f"flash_attn_func failed ({e}), falling back")

    # Path 2: PyTorch SDPA (almost as fast, no external dep)
    if use_flash and _HAS_SDPA:
        try:
            # SDPA handles causal mask internally if we pass is_causal=True
            # For decode (q_len=1), is_causal is a no-op
            is_causal = (q_len > 1) and (q_len == kv_len)
            output = _sdpa(
                query_states, key_states, value_states,
                attn_mask=attention_mask if not is_causal else None,
                is_causal=is_causal,
            )
            return output
        except Exception as e:
            logger.debug(f"SDPA failed ({e}), falling back to manual")

    # Path 3: Manual attention (original implementation, always works)
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)

    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    # Softmax in float32 for numerical stability
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    return attn_output


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
    layers = _get_model_layers(model)
    num_layers = len(layers)
    logger.info(f"Found {num_layers} transformer layers")

    for i, layer in enumerate(layers):
        attn = layer.self_attn

        # Read architecture constants from the attention module.
        # In transformers <4.50 these live on attn; in >=4.50 they moved
        # to the model config. Try both for compatibility.
        num_kv_heads = getattr(attn, "num_key_value_heads", None)
        if num_kv_heads is None:
            num_kv_heads = model.config.num_key_value_heads
        head_dim = getattr(attn, "head_dim", None)
        if head_dim is None:
            head_dim = model.config.head_dim if hasattr(model.config, "head_dim") else \
                       model.config.hidden_size // model.config.num_attention_heads

        # Create spectral cache for this layer
        spectral_cache = SpectralKVCache(config, num_kv_heads, head_dim)

        # Register the cache as a submodule so its parameters are tracked
        # and moved to the correct device along with the model.
        attn.add_module("spectral_cache", spectral_cache)
        # Move spectral cache parameters (learnable filter logits, etc.)
        # to the same device as the attention layer.
        spectral_cache = spectral_cache.to(next(attn.parameters()).device)

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
                past_key_values=None,
                output_attentions: bool = False,
                use_cache: bool = False,
                **kwargs,
            ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple]]:
                return _spectral_forward(
                    attn_module=attn_module,
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
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
    past_key_values=None,
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

    Generation flow (with SpectralDynamicCache):
      - Prefill (q_len > 1): compress full K/V, reconstruct, attend.
      - Decode (q_len == 1): append new K/V via cache, get full K/V, attend.
    This reduces generation from O(N^2) to O(N log N) per step.
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

    # Read architecture constants with fallbacks for transformers >=4.50
    num_q_heads = getattr(attn_module, "num_heads", None)
    if num_q_heads is None:
        num_q_heads = attn_module.config.num_attention_heads
    num_kv_heads = getattr(attn_module, "num_key_value_heads", None)
    if num_kv_heads is None:
        num_kv_heads = attn_module.config.num_key_value_heads
    head_dim = getattr(attn_module, "head_dim", None)
    if head_dim is None:
        cfg = attn_module.config
        head_dim = getattr(cfg, "head_dim", None) or (cfg.hidden_size // cfg.num_attention_heads)

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

    # 3. [SPECTRAL] Compress or append K/V
    #
    # Generation path: if past_key_values is a SpectralDynamicCache,
    # use the Cache protocol for incremental updates (K=1).
    # Training path: always compress full sequence (no cache reuse).
    is_decode_step = (
        isinstance(past_key_values, SpectralDynamicCache)
        and q_len == 1
        and past_key_values.get_seq_length(layer_idx) > 0
    )

    if is_decode_step:
        # Decode: append new token's K/V, get full reconstructed K/V back
        key_states, value_states = past_key_values.update(
            key_states, value_states, layer_idx
        )
    else:
        # Prefill or training: compress full sequence
        spectral_cache.compress(key_states, value_states)

        # Reconstruct from compressed spectral representation
        key_states, value_states = spectral_cache.reconstruct(target_seq_len=q_len)

        # If using a SpectralDynamicCache, update it so future decode
        # steps know the cache is populated
        if isinstance(past_key_values, SpectralDynamicCache):
            # The compress() call above already populated spectral_cache;
            # SpectralDynamicCache just needs to know seq_length is nonzero.
            pass

    # 4. Repeat KV heads for GQA (8 KV heads -> 32 query heads)
    key_states = repeat_kv(key_states, num_q_heads // num_kv_heads)
    value_states = repeat_kv(value_states, num_q_heads // num_kv_heads)

    # 5. Compute attention (Fix 2: use FlashAttention-2 / SDPA if available)
    attn_output = _compute_attention(
        query_states, key_states, value_states,
        attention_mask=attention_mask,
        use_flash=not output_attentions,
    )

    # If output_attentions is requested, recompute with manual attention
    # (FA2/SDPA don't return attention weights)
    if output_attentions:
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
    else:
        attn_weights = None

    # 6. Reshape and output projection
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, num_q_heads * head_dim)
    attn_output = attn_module.o_proj(attn_output)

    # Return in HF's expected format.
    # transformers <4.50: (attn_output, attn_weights, past_key_values)
    # transformers >=4.50: (attn_output, attn_weights) — cache updated in-place
    return attn_output, attn_weights


def _get_model_layers(model: nn.Module) -> nn.ModuleList:
    """Get the transformer layer list from a model, handling PEFT wrapping.

    LlamaForCausalLM: model.model.layers
    PeftModel: model.base_model.model.model.layers
    LlamaModel: model.layers
    """
    # Try direct access first
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    # PEFT wrapping: PeftModel.base_model is LoraModel,
    # whose .model is the original LlamaForCausalLM
    if hasattr(model, "base_model"):
        inner = model.base_model
        if hasattr(inner, "model") and hasattr(inner.model, "model"):
            return inner.model.model.layers
        if hasattr(inner, "model") and hasattr(inner.model, "layers"):
            return inner.model.layers
    # Already a LlamaModel?
    if hasattr(model, "layers"):
        return model.layers
    raise AttributeError(
        f"Could not find .layers in model of type {type(model).__name__}. "
        f"Available attrs: {[a for a in dir(model) if not a.startswith('_')]}"
    )


def get_spectral_caches(model: nn.Module) -> list[SpectralKVCache]:
    """Extract all spectral caches from a model (for inspection or logging).

    Args:
        model: A LlamaForCausalLM (or PeftModel wrapping one) with spectral
               compression applied.

    Returns:
        List of SpectralKVCache objects, one per layer.
    """
    caches = []
    for layer in _get_model_layers(model):
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


def create_spectral_dynamic_cache(model: nn.Module) -> SpectralDynamicCache:
    """Create a SpectralDynamicCache wrapping the model's per-layer caches.

    Pass this to model.generate(past_key_values=...) so that HF's generation
    loop passes only the new token at each decode step instead of the full
    sequence. This is what activates the K=1 incremental caching path.

    Args:
        model: A LlamaForCausalLM with spectral compression applied.

    Returns:
        A SpectralDynamicCache ready to pass to generate().
    """
    return SpectralDynamicCache(get_spectral_caches(model))


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
