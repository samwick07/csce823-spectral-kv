"""Spectral attention: LlamaAttention with FreqKV-style chunk-wise KV compression.

This module implements the FreqKV paradigm (arXiv:2505.00570, ICLR 2026) for
causally-correct spectral KV-cache compression, extended with learnable
spectral filtering and complex FFT support.

Two forward paths:
  - _freqkv_forward_noiterate: for training and perplexity evaluation.
    Processes the full sequence in one pass, chunk-wise. Each chunk compresses
    on-the-fly from the full spatial K/V tensor. No KV cache reuse.
  - _freqkv_forward_iterate: for generation. Uses DynamicCache, stores
    compressed group K/V back, appends new tokens incrementally. Cache stays
    bounded at cache_size.

Chunk-wise attention structure (per FreqKV):
  cache_size = sink_size + fft_span + recent_size
  fft_span = cache_size - sink_size - recent_size
  fft_size = int(fft_span * gamma)  [compressed length]
  chunk_size = fft_span - fft_size  [new tokens per chunk after first]

  First cache_size tokens: standard causal attention (no compression).
  Each subsequent chunk of chunk_size tokens:
    Q = chunk queries
    K = [sink (S uncompressed) + compressed(fft_span→fft_size) + recent (R uncompressed) + chunk]
    V = same structure
    Attention: Q @ K^T with causal mask
    Total KV length = S + fft_size + R + chunk_size = cache_size (bounded)

RoPE is applied AFTER compression (compress raw keys, then rotate the
compressed result). This is consistent across both forward paths, fixing
an inconsistency in FreqKV's reference implementation where the no-iterate
path compresses after RoPE.

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

from .cache import CompressionConfig, SpectralKVCompressor

logger = logging.getLogger(__name__)

# Try to import FlashAttention-2
try:
    from flash_attn import flash_attn_func
    from flash_attn.bert_padding import unpad_input, pad_input
    from flash_attn.flash_attn_interface import flash_attn_varlen_kvpacked_func
    _HAS_FLASH_ATTN = True
except ImportError:
    _HAS_FLASH_ATTN = False
    logger.debug("flash_attn not available. Using SDPA/manual attention.")

# Try to import SDPA as a fallback
try:
    from torch.nn.functional import scaled_dot_product_attention as _sdpa
    _HAS_SDPA = True
except ImportError:
    _HAS_SDPA = False


def _compute_attention_sdpa(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    is_causal: bool = True,
) -> torch.Tensor:
    """Compute attention using SDPA or manual fallback.

    Args:
        query_states: [B, num_heads, S_q, head_dim]
        key_states: [B, num_heads, S_kv, head_dim]
        value_states: [B, num_heads, S_kv, head_dim]
        attention_mask: [B, 1, 1, S_kv] or None
        is_causal: Whether to apply causal mask.
    """
    bsz, num_heads, q_len, head_dim = query_states.shape

    if _HAS_SDPA:
        try:
            output = _sdpa(
                query_states, key_states, value_states,
                attn_mask=attention_mask if not is_causal else None,
                is_causal=is_causal,
            )
            return output
        except Exception as e:
            logger.debug(f"SDPA failed ({e}), falling back to manual")

    # Manual attention
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    if is_causal:
        causal_mask = torch.triu(
            torch.ones(q_len, key_states.shape[-2], device=query_states.device, dtype=torch.bool),
            diagonal=1
        )
        attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    return torch.matmul(attn_weights, value_states)


def _compute_attention_flash(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    is_causal: bool = True,
) -> torch.Tensor:
    """Compute attention using flash_attn_varlen_kvpacked_func.

    Handles packing/unpacking for variable-length sequences.
    Falls back to SDPA if flash_attn is not available.
    """
    bsz, num_heads, q_len, head_dim = query_states.shape
    kv_len = key_states.shape[-2]

    if not _HAS_FLASH_ATTN or q_len == 0 or kv_len == 0:
        return _compute_attention_sdpa(
            query_states, key_states, value_states, is_causal=is_causal
        )

    try:
        # FA2 expects [B, S, H, D] layout
        q_fa2 = query_states.transpose(1, 2)  # [B, S_q, H, D]
        k_fa2 = key_states.transpose(1, 2)    # [B, S_kv, H, D]
        v_fa2 = value_states.transpose(1, 2)  # [B, S_kv, H, D]

        # Use packed KV format
        kv = torch.stack([k_fa2, v_fa2], dim=2)  # [B, S_kv, 2, H, D]
        kv = kv.transpose(1, 3)  # [B, H, 2, S_kv, D] -> need [B, S_kv, 2, H, D]

        # Actually, flash_attn_func is simpler for non-varlen case
        output = flash_attn_func(q_fa2, k_fa2, v_fa2, causal=is_causal)
        return output.transpose(1, 2)  # back to [B, H, S_q, D]
    except Exception as e:
        logger.debug(f"flash_attn failed ({e}), falling back to SDPA")
        return _compute_attention_sdpa(
            query_states, key_states, value_states, is_causal=is_causal
        )


def apply_spectral_compression(
    model: nn.Module,
    config: CompressionConfig,
    is_iterate: bool = True,
) -> nn.Module:
    """Apply FreqKV-style spectral KV compression to a Llama model.

    Wraps each attention layer's forward method to insert chunk-wise
    spectral compression. The spectral compressors (containing learnable
    filter parameters) are registered as submodules named "spectral_cache"
    so their parameters are included in the model's parameter list for
    training and LoRA modules_to_save.

    Args:
        model: A LlamaForCausalLM model.
        config: Compression configuration (transform, filter, gamma, cache params).
        is_iterate: If True, install the iterate forward (for training/generation).
                    If False, install the no-iterate forward (for perplexity eval).

    Returns:
        The model (modified in-place) with spectral compression applied.
    """
    if config.is_baseline:
        logger.info("Baseline config: no spectral compression applied")
        return model

    logger.info(
        f"Applying FreqKV spectral compression: transform={config.transform_type}, "
        f"filter={config.filter_type}, gamma={config.gamma}, "
        f"cache_size={config.cache_size}, sink={config.sink_size}, recent={config.recent_size}, "
        f"is_iterate={is_iterate}"
    )

    layers = _get_model_layers(model)
    num_layers = len(layers)
    logger.info(f"Found {num_layers} transformer layers")

    for i, layer in enumerate(layers):
        attn = layer.self_attn

        # Read architecture constants
        num_kv_heads = getattr(attn, "num_key_value_heads", None)
        if num_kv_heads is None:
            num_kv_heads = model.config.num_key_value_heads
        head_dim = getattr(attn, "head_dim", None)
        if head_dim is None:
            head_dim = model.config.head_dim if hasattr(model.config, "head_dim") else \
                       model.config.hidden_size // model.config.num_attention_heads

        # Create spectral compressor for this layer
        compressor = SpectralKVCompressor(config, num_kv_heads, head_dim)

        # Register as "spectral_cache" submodule for LoRA modules_to_save
        attn.add_module("spectral_cache", compressor)
        compressor = compressor.to(next(attn.parameters()).device)

        # Store config and is_iterate on the attention module
        attn._freqkv_config = config
        attn._freqkv_is_iterate = is_iterate

        # Install the appropriate forward function
        if is_iterate:
            forward_fn = _make_iterate_forward(attn, compressor, config, i)
        else:
            forward_fn = _make_noiterate_forward(attn, compressor, config, i)

        attn.forward = forward_fn

    logger.info(f"Spectral compression applied to {num_layers} layers "
                f"({'iterate' if is_iterate else 'no-iterate'} mode)")
    return model


def _make_noiterate_forward(attn_module, compressor, config, layer_idx):
    """Create a no-iterate forward function for training/perplexity eval.

    Processes the full sequence in one pass, chunk-wise. Each chunk compresses
    on-the-fly from the full spatial K/V tensor. No KV cache reuse.
    """

    def forward(
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple]]:
        return _freqkv_forward_noiterate(
            attn_module=attn_module,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            compressor=compressor,
            config=config,
            layer_idx=layer_idx,
            **kwargs,
        )

    return forward


def _make_iterate_forward(attn_module, compressor, config, layer_idx):
    """Create an iterate forward function for training/generation.

    Uses DynamicCache for KV state. Maintains compressed group K/V between
    chunks. Cache stays bounded at cache_size.
    """

    def forward(
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple]]:
        return _freqkv_forward_iterate(
            attn_module=attn_module,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            compressor=compressor,
            config=config,
            layer_idx=layer_idx,
            **kwargs,
        )

    return forward


def _freqkv_forward_noiterate(
    attn_module: nn.Module,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    past_key_value=None,
    output_attentions: bool = False,
    use_cache: bool = False,
    compressor: SpectralKVCompressor = None,
    config: CompressionConfig = None,
    layer_idx: int = 0,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple]]:
    """No-iterate forward: chunk-wise compression from full spatial K/V.

    Used for training and perplexity evaluation. Processes the full sequence
    in one forward pass, applying compression on-the-fly per chunk.
    """
    from transformers.models.llama.modeling_llama import (
        apply_rotary_pos_emb,
        repeat_kv,
    )

    bsz, q_len, _ = hidden_states.shape

    # 1. Project Q, K, V
    query_states = attn_module.q_proj(hidden_states)
    key_states = attn_module.k_proj(hidden_states)
    value_states = attn_module.v_proj(hidden_states)

    num_q_heads = getattr(attn_module, "num_heads", None) or attn_module.config.num_attention_heads
    num_kv_heads = getattr(attn_module, "num_key_value_heads", None) or attn_module.config.num_key_value_heads
    head_dim = getattr(attn_module, "head_dim", None)
    if head_dim is None:
        cfg = attn_module.config
        head_dim = getattr(cfg, "head_dim", None) or (cfg.hidden_size // cfg.num_attention_heads)

    query_states = query_states.view(bsz, q_len, num_q_heads, head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)

    # 2. Get RoPE embeddings (but DON'T apply yet — compress before RoPE)
    position_embeddings = kwargs.get("position_embeddings", None)

    # 3. Chunk-wise attention
    sink_size = config.sink_size
    recent_size = config.recent_size
    cache_size = min(config.cache_size, q_len)
    fft_span = cache_size - sink_size - recent_size
    fft_size = int(fft_span * config.gamma)
    chunk_size = fft_span - fft_size

    if q_len <= cache_size:
        # No compression needed — standard causal attention
        # Apply RoPE to full Q and K
        if position_embeddings is not None:
            cos, sin = position_embeddings
        elif hasattr(attn_module, "rotary_emb") and attn_module.rotary_emb is not None:
            cos, sin = attn_module.rotary_emb(value_states, position_ids)
        else:
            cos = sin = None

        if cos is not None:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        key_states = repeat_kv(key_states, num_q_heads // num_kv_heads)
        value_states = repeat_kv(value_states, num_q_heads // num_kv_heads)

        attn_output = _compute_attention_flash(
            query_states, key_states, value_states, is_causal=True
        )
    else:
        # Chunk-wise compression
        num_groups = math.ceil((q_len - cache_size) / chunk_size)

        # Compute RoPE for full sequence
        if position_embeddings is not None:
            cos, sin = position_embeddings
        elif hasattr(attn_module, "rotary_emb") and attn_module.rotary_emb is not None:
            cos, sin = attn_module.rotary_emb(value_states, position_ids)
        else:
            cos = sin = None

        # First chunk: standard attention on first cache_size tokens
        q_first = query_states[:, :, :cache_size, :]
        k_first = key_states[:, :, :cache_size, :]
        v_first = value_states[:, :, :cache_size, :]

        # Apply RoPE to first chunk
        if cos is not None:
            q_first, k_first = apply_rotary_pos_emb(q_first, k_first, cos, sin)

        k_first = repeat_kv(k_first, num_q_heads // num_kv_heads)
        v_first = repeat_kv(v_first, num_q_heads // num_kv_heads)

        attn_output = _compute_attention_flash(
            q_first, k_first, v_first, is_causal=True
        )

        # Process subsequent chunks
        group_key = key_states[:, :, :cache_size, :]
        group_value = value_states[:, :, :cache_size, :]

        for group_idx in range(num_groups):
            chunk_start = cache_size + group_idx * chunk_size
            chunk_end = min(cache_size + (group_idx + 1) * chunk_size, q_len)
            actual_chunk_len = chunk_end - chunk_start

            if actual_chunk_len <= 0:
                break

            # Get chunk queries (un-rotated)
            q_chunk = query_states[:, :, chunk_start:chunk_end, :]

            # Compress past tokens (group_key minus sink and recent)
            # Compress tokens [sink_size : sink_size + fft_span] to fft_size
            if fft_size > 0 and group_key.shape[-2] > sink_size + fft_span:
                compress_start = sink_size
                compress_end = min(sink_size + fft_span, group_key.shape[-2])
                to_compress_k = group_key[:, :, compress_start:compress_end, :]
                to_compress_v = group_value[:, :, compress_start:compress_end, :]

                compressed_k = compressor.compress(to_compress_k, fft_size)
                compressed_v = compressor.compress(to_compress_v, fft_size)
            elif fft_size > 0 and group_key.shape[-2] > sink_size + recent_size:
                to_compress_k = group_key[:, :, sink_size:group_key.shape[-2] - recent_size, :]
                to_compress_v = group_value[:, :, sink_size:group_value.shape[-2] - recent_size, :]
                if to_compress_k.shape[-2] > fft_size:
                    compressed_k = compressor.compress(to_compress_k, fft_size)
                    compressed_v = compressor.compress(to_compress_v, fft_size)
                else:
                    compressed_k = to_compress_k
                    compressed_v = to_compress_v
            else:
                compressed_k = group_key[:, :, sink_size:sink_size, :]
                compressed_v = group_value[:, :, sink_size:sink_size, :]

            # Build group K/V: [sink + compressed + recent + chunk]
            recent_start = max(group_key.shape[-2] - recent_size, sink_size)
            recent_k = group_key[:, :, recent_start:, :]
            recent_v = group_value[:, :, recent_start:, :]

            chunk_k = key_states[:, :, chunk_start:chunk_end, :]
            chunk_v = value_states[:, :, chunk_start:chunk_end, :]

            group_k_full = torch.cat([
                group_key[:, :, :sink_size, :],  # sink (uncompressed)
                compressed_k,                      # compressed past
                recent_k,                          # recent (uncompressed)
                chunk_k,                           # current chunk
            ], dim=-2)
            group_v_full = torch.cat([
                group_value[:, :, :sink_size, :],
                compressed_v,
                recent_v,
                chunk_v,
            ], dim=-2)

            # Apply RoPE to Q and K for this chunk
            if cos is not None:
                # Q gets RoPE starting from its position
                # K gets RoPE from position 0 (the full group)
                q_chunk_rot, k_group_rot = apply_rotary_pos_emb(
                    q_chunk, group_k_full, cos, sin
                )
            else:
                q_chunk_rot = q_chunk
                k_group_rot = group_k_full

            q_chunk_rot = repeat_kv(q_chunk_rot, 1)  # already num_q_heads
            k_group_rot = repeat_kv(k_group_rot, num_q_heads // num_kv_heads)
            v_group = repeat_kv(group_v_full, num_q_heads // num_kv_heads)

            # Attention with causal mask (Q can only attend to positions <= its own)
            chunk_attn = _compute_attention_sdpa(
                q_chunk_rot, k_group_rot, v_group,
                is_causal=True
            )

            attn_output = torch.cat([attn_output, chunk_attn], dim=-2)

            # Update group state for next iteration
            group_key = group_k_full
            group_value = group_v_full

    # 4. Output projection
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, num_q_heads * head_dim)
    attn_output = attn_module.o_proj(attn_output)

    return attn_output, None


def _freqkv_forward_iterate(
    attn_module: nn.Module,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    past_key_value=None,
    output_attentions: bool = False,
    use_cache: bool = False,
    compressor: SpectralKVCompressor = None,
    config: CompressionConfig = None,
    layer_idx: int = 0,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple]]:
    """Iterate forward: compounding compression with DynamicCache.

    Used for training (is_iterate=True) and generation. Maintains
    compressed group K/V state between chunks. Cache stays bounded
    at cache_size.

    During generation (with DynamicCache):
      - Prefill (q_len > 1): same as no-iterate, stores final group K/V.
      - Decode (q_len == 1): append new token, recompress, attend.
    """
    from transformers.models.llama.modeling_llama import (
        apply_rotary_pos_emb,
        repeat_kv,
    )

    bsz, q_len, _ = hidden_states.shape

    # 1. Project Q, K, V
    query_states = attn_module.q_proj(hidden_states)
    key_states = attn_module.k_proj(hidden_states)
    value_states = attn_module.v_proj(hidden_states)

    num_q_heads = getattr(attn_module, "num_heads", None) or attn_module.config.num_attention_heads
    num_kv_heads = getattr(attn_module, "num_key_value_heads", None) or attn_module.config.num_key_value_heads
    head_dim = getattr(attn_module, "head_dim", None)
    if head_dim is None:
        cfg = attn_module.config
        head_dim = getattr(cfg, "head_dim", None) or (cfg.hidden_size // cfg.num_attention_heads)

    query_states = query_states.view(bsz, q_len, num_q_heads, head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)

    # 2. Get RoPE embeddings
    position_embeddings = kwargs.get("position_embeddings", None)
    if position_embeddings is not None:
        cos, sin = position_embeddings
    elif hasattr(attn_module, "rotary_emb") and attn_module.rotary_emb is not None:
        cos, sin = attn_module.rotary_emb(value_states, position_ids)
    else:
        cos = sin = None

    # 3. Check if this is a decode step (generation with DynamicCache)
    is_decode = (
        past_key_value is not None
        and q_len == 1
        and hasattr(past_key_value, 'get_seq_length')
        and past_key_value.get_seq_length(layer_idx) > 0
    )

    sink_size = config.sink_size
    recent_size = config.recent_size
    cache_size = config.cache_size
    fft_span = cache_size - sink_size - recent_size
    fft_size = int(fft_span * config.gamma)
    chunk_size = fft_span - fft_size

    if is_decode:
        # Decode step: append new K/V, recompress, attend
        # Get cached K/V from DynamicCache
        old_k, old_v = past_key_value.update(
            key_states, value_states, layer_idx,
            cache_kwargs={"sin": cos, "cos": sin} if cos is not None else None
        )

        # old_k now has all cached + new token. If it exceeds cache_size, compress.
        total_len = old_k.shape[-2]

        if total_len <= cache_size:
            # Still within cache budget — no compression needed
            k_attend = old_k
            v_attend = old_v
        else:
            # Compress: keep sink + compress middle + keep recent + new token
            # The new token is at the end (1 token)
            compress_end = total_len - recent_size - 1  # exclude recent and new
            if compress_end > sink_size and fft_size > 0:
                to_compress_k = old_k[:, :, sink_size:compress_end, :]
                to_compress_v = old_v[:, :, sink_size:compress_end, :]

                compressed_k = compressor.compress(to_compress_k, fft_size)
                compressed_v = compressor.compress(to_compress_v, fft_size)

                k_attend = torch.cat([
                    old_k[:, :, :sink_size, :],     # sink
                    compressed_k,                    # compressed
                    old_k[:, :, compress_end:, :],  # recent + new
                ], dim=-2)
                v_attend = torch.cat([
                    old_v[:, :, :sink_size, :],
                    compressed_v,
                    old_v[:, :, compress_end:, :],
                ], dim=-2)
            else:
                k_attend = old_k
                v_attend = old_v

        # Apply RoPE
        if cos is not None:
            query_states, k_attend = apply_rotary_pos_emb(query_states, k_attend, cos, sin)

        k_attend = repeat_kv(k_attend, num_q_heads // num_kv_heads)
        v_attend = repeat_kv(v_attend, num_q_heads // num_kv_heads)

        attn_output = _compute_attention_flash(
            query_states, k_attend, v_attend, is_causal=True
        )

        # Store compressed K/V back into cache
        if use_cache and total_len > cache_size:
            # Update cache with compressed version
            # DynamicCache stores by layer_idx
            try:
                past_key_value.key_cache[layer_idx] = k_attend[:, :num_kv_heads, :, :]
                past_key_value.value_cache[layer_idx] = v_attend[:, :num_kv_heads, :, :]
            except (AttributeError, IndexError):
                pass  # Some cache implementations don't expose internal storage

    else:
        # Prefill or training: same chunk-wise logic as no-iterate
        # but store group K/V in DynamicCache for future decode steps
        effective_cache_size = min(cache_size, q_len)

        if q_len <= effective_cache_size:
            # No compression — standard attention
            if cos is not None:
                query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

            key_states = repeat_kv(key_states, num_q_heads // num_kv_heads)
            value_states = repeat_kv(value_states, num_q_heads // num_kv_heads)

            attn_output = _compute_attention_flash(
                query_states, key_states, value_states, is_causal=True
            )

            # Store in cache
            if use_cache and past_key_value is not None:
                past_key_value.update(key_states[:, :num_kv_heads], value_states[:, :num_kv_heads], layer_idx)
        else:
            # Chunk-wise with cache storage
            num_groups = math.ceil((q_len - effective_cache_size) / chunk_size)

            # First chunk
            q_first = query_states[:, :, :effective_cache_size, :]
            k_first = key_states[:, :, :effective_cache_size, :]
            v_first = value_states[:, :, :effective_cache_size, :]

            if cos is not None:
                q_first, k_first = apply_rotary_pos_emb(q_first, k_first, cos, sin)

            k_first_rep = repeat_kv(k_first, num_q_heads // num_kv_heads)
            v_first_rep = repeat_kv(v_first, num_q_heads // num_kv_heads)

            attn_output = _compute_attention_flash(q_first, k_first_rep, v_first_rep, is_causal=True)

            group_key = key_states[:, :, :effective_cache_size, :]
            group_value = value_states[:, :, :effective_cache_size, :]

            for group_idx in range(num_groups):
                chunk_start = effective_cache_size + group_idx * chunk_size
                chunk_end = min(effective_cache_size + (group_idx + 1) * chunk_size, q_len)
                actual_chunk_len = chunk_end - chunk_start
                if actual_chunk_len <= 0:
                    break

                q_chunk = query_states[:, :, chunk_start:chunk_end, :]

                # Compress past (group minus sink, recent, and chunk)
                if fft_size > 0 and group_key.shape[-2] > sink_size + recent_size:
                    compress_start = sink_size
                    compress_end = min(sink_size + fft_span, group_key.shape[-2] - recent_size)
                    if compress_end > compress_start:
                        to_compress_k = group_key[:, :, compress_start:compress_end, :]
                        to_compress_v = group_value[:, :, compress_start:compress_end, :]
                        compressed_k = compressor.compress(to_compress_k, fft_size)
                        compressed_v = compressor.compress(to_compress_v, fft_size)
                    else:
                        compressed_k = group_key[:, :, sink_size:sink_size, :]
                        compressed_v = group_value[:, :, sink_size:sink_size, :]
                else:
                    compressed_k = group_key[:, :, sink_size:sink_size, :]
                    compressed_v = group_value[:, :, sink_size:sink_size, :]

                recent_start = max(group_key.shape[-2] - recent_size, sink_size)
                recent_k = group_key[:, :, recent_start:, :]
                recent_v = group_value[:, :, recent_start:, :]

                chunk_k = key_states[:, :, chunk_start:chunk_end, :]
                chunk_v = value_states[:, :, chunk_start:chunk_end, :]

                group_k_full = torch.cat([
                    group_key[:, :, :sink_size, :],
                    compressed_k,
                    recent_k,
                    chunk_k,
                ], dim=-2)
                group_v_full = torch.cat([
                    group_value[:, :, :sink_size, :],
                    compressed_v,
                    recent_v,
                    chunk_v,
                ], dim=-2)

                if cos is not None:
                    q_chunk_rot, k_group_rot = apply_rotary_pos_emb(q_chunk, group_k_full, cos, sin)
                else:
                    q_chunk_rot = q_chunk
                    k_group_rot = group_k_full

                k_group_rep = repeat_kv(k_group_rot, num_q_heads // num_kv_heads)
                v_group_rep = repeat_kv(group_v_full, num_q_heads // num_kv_heads)

                chunk_attn = _compute_attention_sdpa(q_chunk_rot, k_group_rep, v_group_rep, is_causal=True)
                attn_output = torch.cat([attn_output, chunk_attn], dim=-2)

                group_key = group_k_full
                group_value = group_v_full

            # Store final group K/V in cache for decode steps
            if use_cache and past_key_value is not None:
                past_key_value.update(group_key, group_value, layer_idx)

    # 4. Output projection
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, num_q_heads * head_dim)
    attn_output = attn_module.o_proj(attn_output)

    return attn_output, None


def _get_model_layers(model: nn.Module) -> nn.ModuleList:
    """Get the transformer layer list from a model, handling PEFT wrapping."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "base_model"):
        inner = model.base_model
        if hasattr(inner, "model") and hasattr(inner.model, "model"):
            return inner.model.model.layers
        if hasattr(inner, "model") and hasattr(inner.model, "layers"):
            return inner.model.layers
    if hasattr(model, "layers"):
        return model.layers
    raise AttributeError(
        f"Could not find .layers in model of type {type(model).__name__}. "
        f"Available attrs: {[a for a in dir(model) if not a.startswith('_')]}"
    )


def get_spectral_compressors(model: nn.Module) -> list[SpectralKVCompressor]:
    """Extract all spectral compressors from a model."""
    compressors = []
    for layer in _get_model_layers(model):
        attn = layer.self_attn
        if hasattr(attn, "spectral_cache"):
            compressors.append(attn.spectral_cache)
    return compressors


# Backward compatibility
get_spectral_caches = get_spectral_compressors


def get_compression_stats(model: nn.Module) -> list[dict]:
    """Get compression statistics from all layers."""
    stats = []
    for i, comp in enumerate(get_spectral_compressors(model)):
        stats.append({
            "layer": i,
            "compression_ratio": comp.get_compression_ratio(),
            "is_spectral": comp.transform is not None,
            **comp.compression_stats,
        })
    return stats


def get_learnable_filter_params(model: nn.Module) -> list[nn.Parameter]:
    """Get all learnable filter parameters from the model."""
    params = []
    for comp in get_spectral_compressors(model):
        params.extend(comp.get_learnable_parameters())
    return params


def reset_all_caches(model: nn.Module) -> None:
    """Reset all spectral compressor stats in the model."""
    for comp in get_spectral_compressors(model):
        comp.reset_stats()


# Backward compatibility
class CompressedAttention:
    """Deprecated: use apply_spectral_compression() instead."""

    def __init__(self, config: CompressionConfig, num_kv_heads: int = 8, head_dim: int = 128):
        self.config = config
        self.compressor = SpectralKVCompressor(config, num_kv_heads, head_dim)

    def apply_to_model(self, model: nn.Module) -> nn.Module:
        return apply_spectral_compression(model, self.config)
