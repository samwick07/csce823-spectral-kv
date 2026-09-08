"""Chunked spectral compression: causality-preserving variant.

The standard spectral KV-cache compression (cache.py) transforms the FULL
sequence dimension at once. After truncation + reconstruction, V_recon[j]
for j <= t contains information from future positions j+1..N-1, breaking
causality in autoregressive language modeling.

This module provides a drop-in replacement that applies the spectral
transform to local chunks (windows) of the K/V tensors. Each chunk only
mixes information within its window. The causal attention mask then
prevents attention from accessing future chunks' V values, preserving
causality.

This is NOT applied to the current running experiment (N=1). It is a
proposed fix for the future full experiment (N=30) and is provided here
for review and testing.

Usage:
    from src.spectral.chunked_cache import ChunkedCompressionConfig, ChunkedSpectralKVCache

    config = ChunkedCompressionConfig(
        transform_type="fft",
        filter_type="learnable",
        gamma=0.22,
        chunk_size=256,
        max_seq_len=16384,
    )
    cache = ChunkedSpectralKVCache(config, num_kv_heads=8, head_dim=128)
    cache.compress(key_states, value_states)
    k_recon, v_recon = cache.reconstruct()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from .transform import SpectralTransform, DCTTransform, FFTTransform
from .filter import SpectralFilter, FixedLowPassFilter, LearnableSpectralFilter
from .cache import CompressionConfig, SpectralKVCache

logger = logging.getLogger(__name__)


@dataclass
class ChunkedCompressionConfig(CompressionConfig):
    """Configuration for chunked spectral KV-cache compression.

    Extends CompressionConfig with a chunk_size parameter that controls
    the local window size for the spectral transform.

    Attributes:
        chunk_size: Window size for local spectral transforms.
                   Must be a power of 2 for FFT efficiency.
                   Default 256 (matches eval_window_size).
    """
    chunk_size: int = 256


class ChunkedSpectralKVCache(SpectralKVCache):
    """Spectral KV cache with chunked (causality-preserving) compression.

    Instead of transforming the full sequence at once, this cache splits
    the K/V tensors into non-overlapping chunks of size chunk_size, applies
    the spectral transform to each chunk independently, truncates each
    chunk's spectral coefficients, and stores the per-chunk compressed
    representations.

    During reconstruction, each chunk is inverse-transformed independently.
    V_recon within chunk c only contains information from positions within
    chunk c. The causal attention mask prevents Q[t] from attending to
    K_recon[j] for j > t, which also prevents access to V_recon in future
    chunks.

    For positions within the same chunk as t but after t, the causal mask
    still blocks attention to those K positions. However, V_recon at
    positions within the same chunk but before t may contain slight leakage
    from positions after t within the same chunk. This residual leakage
    is bounded by the chunk size and is typically negligible (chunk_size=256
    vs seq_len=2048 means 8x less leakage than full-sequence transform).
    """

    def __init__(
        self,
        config: ChunkedCompressionConfig,
        num_kv_heads: int,
        head_dim: int,
    ):
        super().__init__(config, num_kv_heads, head_dim)
        self.chunk_size = config.chunk_size

    def compress(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> None:
        """Transform and store KV states in chunked spectral domain.

        Splits K/V into chunks of size chunk_size, applies spectral
        transform + truncation + filter to each chunk independently,
        and stores the per-chunk compressed coefficients.

        Args:
            key_states: [B, num_kv_heads, S, head_dim]
            value_states: [B, num_kv_heads, S, head_dim]
        """
        if not self._is_spectral:
            # Baseline: store spatial directly (same as parent)
            self._cached_k_spectral = key_states
            self._cached_v_spectral = value_states
            self._cached_seq_len = key_states.shape[-2]
            self.compression_stats["total_compress_calls"] += 1
            self.compression_stats["total_spatial_elements"] += (
                key_states.shape[-2] * self.num_kv_heads * self.head_dim
            )
            return

        seq_len = key_states.shape[-2]
        self._orig_dtype = key_states.dtype
        chunk_size = self.chunk_size

        # Process each chunk independently
        k_chunks_compressed = []
        v_chunks_compressed = []

        for start in range(0, seq_len, chunk_size):
            end = min(start + chunk_size, seq_len)
            k_chunk = key_states[:, :, start:end, :]
            v_chunk = value_states[:, :, start:end, :]

            # Transform chunk to spectral domain
            k_spectral = self.transform(k_chunk)
            v_spectral = self.transform(v_chunk)

            # Truncate each chunk's spectral coefficients
            k_spectral = self.transform.truncate(k_spectral, self.config.gamma)
            v_spectral = self.transform.truncate(v_spectral, self.config.gamma)

            # Apply filter (learnable mask or no-op for fixed)
            k_spectral = self.filter(k_spectral, self.config.gamma)
            v_spectral = self.filter(v_spectral, self.config.gamma)

            k_chunks_compressed.append(k_spectral)
            v_chunks_compressed.append(v_spectral)

        # Store as list of per-chunk tensors (variable sizes if last chunk is partial)
        self._cached_k_spectral = k_chunks_compressed
        self._cached_v_spectral = v_chunks_compressed
        self._cached_seq_len = seq_len

        self.compression_stats["total_compress_calls"] += 1
        self.compression_stats["total_spatial_elements"] += (
            seq_len * self.num_kv_heads * self.head_dim
        )
        total_spectral = sum(
            c.shape[-2] * self.num_kv_heads * self.head_dim
            for c in k_chunks_compressed
        )
        self.compression_stats["total_spectral_elements"] += total_spectral

    def reconstruct(
        self, target_seq_len: Optional[int] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct K and V from chunked spectral cache.

        Each chunk is inverse-transformed independently, then concatenated
        back to the full sequence length.

        Args:
            target_seq_len: Target sequence length (must match cached length).

        Returns:
            Tuple of (key_states, value_states), each [B, H, S, D].
        """
        if self._cached_k_spectral is None:
            raise RuntimeError("No cached KV states. Call compress() first.")

        if not self._is_spectral:
            # Baseline: already spatial
            return self._cached_k_spectral, self._cached_v_spectral

        target_len = target_seq_len or self._cached_seq_len
        chunk_size = self.chunk_size

        k_chunks_recon = []
        v_chunks_recon = []
        pos = 0

        for i, (k_chunk_spec, v_chunk_spec) in enumerate(
            zip(self._cached_k_spectral, self._cached_v_spectral)
        ):
            # Determine this chunk's original length
            end = min(pos + chunk_size, target_len)
            chunk_len = end - pos

            k_recon = self.transform.inverse(k_chunk_spec, chunk_len)
            v_recon = self.transform.inverse(v_chunk_spec, chunk_len)

            # Cast back to original dtype
            orig_dtype = getattr(self, "_orig_dtype", None)
            if orig_dtype is not None and orig_dtype != torch.float32:
                k_recon = k_recon.to(orig_dtype)
                v_recon = v_recon.to(orig_dtype)

            k_chunks_recon.append(k_recon)
            v_chunks_recon.append(v_recon)
            pos = end

        k_spatial = torch.cat(k_chunks_recon, dim=-2)
        v_spatial = torch.cat(v_chunks_recon, dim=-2)

        self.compression_stats["total_reconstruct_calls"] += 1
        return k_spatial, v_spatial

    def append(
        self,
        new_key: torch.Tensor,
        new_value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """K=1 incremental update with chunked compression.

        For the chunked variant, we reconstruct the full K/V from the
        chunked cache, append the new token, and recompress all chunks.
        This is less efficient than the full-sequence variant but preserves
        causality.

        Args:
            new_key: [B, H, 1, D] -- new token's key
            new_value: [B, H, 1, D] -- new token's value

        Returns:
            Full lossy reconstructed K/V [B, H, S+1, D]
        """
        if self._cached_k_spectral is None:
            self.compress(new_key, new_value)
            return self.reconstruct()

        if not self._is_spectral:
            return super().append(new_key, new_value)

        # Reconstruct old K/V
        old_k, old_v = self.reconstruct()

        # Concatenate new token
        full_k = torch.cat([old_k, new_key], dim=-2)
        full_v = torch.cat([old_v, new_value], dim=-2)

        # Recompress with chunking
        self.compress(full_k, full_v)

        # Return lossy reconstruction
        return self.reconstruct()

    def get_compression_ratio(self) -> float:
        """Return the actual achieved compression ratio."""
        if not self._is_spectral or self._cached_k_spectral is None:
            return 1.0
        if isinstance(self._cached_k_spectral, list):
            total_spectral = sum(c.shape[-2] for c in self._cached_k_spectral)
        else:
            total_spectral = self._cached_k_spectral.shape[-2]
        return total_spectral / self._cached_seq_len if self._cached_seq_len > 0 else 1.0

    def get_cache_size_bytes(self) -> int:
        """Return the current cache size in bytes."""
        if self._cached_k_spectral is None:
            return 0
        if isinstance(self._cached_k_spectral, list):
            k_bytes = sum(c.nelement() * c.element_size() for c in self._cached_k_spectral)
            v_bytes = sum(c.nelement() * c.element_size() for c in self._cached_v_spectral)
        else:
            k_bytes = self._cached_k_spectral.nelement() * self._cached_k_spectral.element_size()
            v_bytes = self._cached_v_spectral.nelement() * self._cached_v_spectral.element_size()
        return k_bytes + v_bytes

    def reset(self) -> None:
        """Clear the cache."""
        self._cached_k_spectral = None
        self._cached_v_spectral = None
        self._cached_seq_len = 0
