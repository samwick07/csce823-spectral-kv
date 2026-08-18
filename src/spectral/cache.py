"""Spectral KV-cache: stores compressed key/value states in the frequency domain.

Following the FreqKV paradigm (arXiv:2505.00570, ICLR 2026):

  During generation, the KV cache grows linearly with sequence length.
  Instead of storing the full spatial KV tensors, we:

    1. Transform K and V into the spectral domain (DCT or FFT).
    2. Apply the spectral filter (fixed low-pass or learnable mask).
    3. Truncate to retain only gamma fraction of coefficients.
    4. Store the compressed spectral coefficients as the cache.

  When attention needs to compute Q @ K^T and softmax(QK^T) @ V:
    5. Reconstruct K and V from the compressed spectral representation
       back to the spatial domain at the original (or reduced) length.
    6. Perform standard attention on the reconstructed tensors.

  This achieves a true cache size reduction from O(N * d) to O(gamma * N * d),
  at the cost of reconstruction error from spectral truncation.

  During training, compression is applied within each forward pass to teach
  the model to operate with the spectral-domain cache. The learnable filter
  (if present) is updated via backpropagation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn

from .transform import SpectralTransform, DCTTransform, FFTTransform
from .filter import SpectralFilter, FixedLowPassFilter, LearnableSpectralFilter

logger = logging.getLogger(__name__)


@dataclass
class CompressionConfig:
    """Configuration for spectral KV-cache compression.

    Attributes:
        transform_type: "dct", "fft", or "none" (baseline).
        filter_type: "fixed", "learnable", or "none".
        gamma: Compression ratio (0 < gamma <= 1). Fraction of spectral
               coefficients retained. 1.0 = no compression.
        max_seq_len: Maximum sequence length for spectral resolution.
        init_sharpness: Initial sharpness of learnable filter's low-pass.
        init_offset: Initial frequency offset of learnable filter's cutoff.
    """
    transform_type: str = "fft"
    filter_type: str = "learnable"
    gamma: float = 0.22
    max_seq_len: int = 16384
    init_sharpness: float = 10.0
    init_offset: float = 0.0

    @property
    def is_baseline(self) -> bool:
        return self.transform_type == "none" or self.gamma >= 1.0

    @property
    def variant_name(self) -> str:
        if self.is_baseline:
            return "baseline"
        return f"{self.transform_type}_{self.filter_type}"


class SpectralKVCache(nn.Module):
    """Spectral-domain KV cache for a single attention layer.

    Stores compressed spectral coefficients for keys and values instead of
    the full spatial tensors. Reconstructs to spatial domain on demand.

    For the baseline (no compression), this is a pass-through -- the cache
    stores spatial tensors directly with no spectral transform.

    The spectral-domain cache works as follows:

      compress(kv_states, seq_len) -> spectral_coefficients:
        spatial [B, H, S, D] -> spectral [B, H, S_spec, D] (complex or real)
        After truncation: [B, H, gamma*S_spec, D]

      reconstruct(seq_len) -> spatial_kv:
        spectral [B, H, gamma*S_spec, D] -> spatial [B, H, S, D]

    During generation, compress is called after computing new K/V states,
    and reconstruct is called before attention computation.
    """

    def __init__(
        self,
        config: CompressionConfig,
        num_kv_heads: int,
        head_dim: int,
    ):
        """
        Args:
            config: Compression configuration.
            num_kv_heads: Number of KV heads (may differ from query heads with GQA).
            head_dim: Dimension of each attention head.
        """
        super().__init__()

        self.config = config
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        if config.is_baseline:
            # Baseline: no spectral processing
            self.transform: Optional[SpectralTransform] = None
            self.filter: Optional[SpectralFilter] = None
        else:
            # Build the spectral transform
            if config.transform_type == "dct":
                self.transform = DCTTransform(dim=-2)
            elif config.transform_type == "fft":
                self.transform = FFTTransform(dim=-2)
            else:
                raise ValueError(f"Unknown transform_type: {config.transform_type}")

            # Build the spectral filter
            max_spectral_len = self.transform.spectral_len(config.max_seq_len)

            if config.filter_type == "fixed":
                self.filter = FixedLowPassFilter()
            elif config.filter_type == "learnable":
                self.filter = LearnableSpectralFilter(
                    num_heads=num_kv_heads,
                    max_spectral_len=max_spectral_len,
                    init_sharpness=config.init_sharpness,
                    init_offset=config.init_offset,
                )
            else:
                raise ValueError(f"Unknown filter_type: {config.filter_type}")

        # Cache state: either spatial (baseline) or spectral (compressed)
        self._cached_k_spectral: Optional[torch.Tensor] = None
        self._cached_v_spectral: Optional[torch.Tensor] = None
        self._cached_seq_len: int = 0
        self._is_spectral: bool = not config.is_baseline

        # Statistics for logging
        self.compression_stats = {
            "total_compress_calls": 0,
            "total_reconstruct_calls": 0,
            "total_spatial_elements": 0,
            "total_spectral_elements": 0,
        }

    def compress(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> None:
        """Transform and store KV states in the spectral domain.

        Called after the attention layer computes new K/V projections.
        For each forward pass during training, this compresses the full
        sequence's KV states. During generation, this appends and recompresses.

        Args:
            key_states: [B, num_kv_heads, S, head_dim]
            value_states: [B, num_kv_heads, S, head_dim]
        """
        seq_len = key_states.shape[-2]

        if self._is_spectral:
            # Transform to spectral domain
            k_spectral = self.transform(key_states)    # [B, H, S_spec, D]
            v_spectral = self.transform(value_states)  # [B, H, S_spec, D]

            # Apply the spectral filter (learnable mask or no-op for fixed)
            k_spectral = self.filter(k_spectral, self.config.gamma)
            v_spectral = self.filter(v_spectral, self.config.gamma)

            # Truncate to gamma fraction
            k_spectral = self.transform.truncate(k_spectral, self.config.gamma)
            v_spectral = self.transform.truncate(v_spectral, self.config.gamma)

            self._cached_k_spectral = k_spectral
            self._cached_v_spectral = v_spectral
        else:
            # Baseline: store spatial directly
            self._cached_k_spectral = key_states
            self._cached_v_spectral = value_states

        self._cached_seq_len = seq_len
        self.compression_stats["total_compress_calls"] += 1
        self.compression_stats["total_spatial_elements"] += seq_len * self.num_kv_heads * self.head_dim
        if self._is_spectral:
            spectral_len = self._cached_k_spectral.shape[-2]
            self.compression_stats["total_spectral_elements"] += spectral_len * self.num_kv_heads * self.head_dim

    def reconstruct(self, target_seq_len: Optional[int] = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct K and V from the spectral cache.

        Called before attention computation (Q @ K^T, softmax(QK^T) @ V).

        Args:
            target_seq_len: Target sequence length for reconstruction.
                            If None, uses the original cached sequence length.

        Returns:
            Tuple of (key_states, value_states), each [B, num_kv_heads, S, head_dim].
        """
        if self._cached_k_spectral is None:
            raise RuntimeError("No cached KV states. Call compress() first.")

        target_len = target_seq_len or self._cached_seq_len

        if self._is_spectral:
            k_spatial = self.transform.inverse(self._cached_k_spectral, target_len)
            v_spatial = self.transform.inverse(self._cached_v_spectral, target_len)
        else:
            # Baseline: already spatial
            k_spatial = self._cached_k_spectral
            v_spatial = self._cached_v_spectral

        self.compression_stats["total_reconstruct_calls"] += 1
        return k_spatial, v_spatial

    def get_compression_ratio(self) -> float:
        """Return the actual achieved compression ratio (cached / original)."""
        if not self._is_spectral or self._cached_k_spectral is None:
            return 1.0
        spectral_len = self._cached_k_spectral.shape[-2]
        return spectral_len / self._cached_seq_len if self._cached_seq_len > 0 else 1.0

    def get_cache_size_bytes(self) -> int:
        """Return the current cache size in bytes."""
        if self._cached_k_spectral is None:
            return 0
        k_bytes = self._cached_k_spectral.nelement() * self._cached_k_spectral.element_size()
        v_bytes = self._cached_v_spectral.nelement() * self._cached_v_spectral.element_size()
        return k_bytes + v_bytes

    def reset(self) -> None:
        """Clear the cache."""
        self._cached_k_spectral = None
        self._cached_v_spectral = None
        self._cached_seq_len = 0

    def get_learnable_parameters(self) -> list[nn.Parameter]:
        """Return learnable parameters from the filter (for LoRA modules_to_save)."""
        if self.filter is not None and hasattr(self.filter, 'filter_logits'):
            return [self.filter.filter_logits]
        return []
