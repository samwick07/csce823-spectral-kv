"""Compressed attention layer that applies spectral KV-cache compression.

This module patches Llama-3's attention mechanism to compress the KV cache
in the spectral domain before performing attention computation.

The 2x2 factorial design is realized by combining:
  - Transform: DCTTransform or FFTTransform
  - Filter: FixedLowPassFilter or LearnableSpectralFilter
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from .transform import SpectralTransform, DCTTransform, FFTTransform
from .filter import SpectralFilter, FixedLowPassFilter, LearnableSpectralFilter

logger = logging.getLogger(__name__)


@dataclass
class CompressionConfig:
    """Configuration for KV-cache spectral compression.

    The transform_type and filter_type together define the ablation variant:
      - DCT + Fixed     = FreqKV baseline
      - DCT + Learnable = Isolates filter effect
      - FFT + Fixed     = Isolates phase preservation effect
      - FFT + Learnable = Full proposed method
    """

    transform_type: str = "dct"  # "dct" or "fft"
    filter_type: str = "fixed"   # "fixed" or "learnable"
    gamma: float = 0.22          # Fraction of spectral coefficients to retain
    max_seq_len: int = 16384     # Max sequence length (for learnable filter init)
    init_sharpness: float = 10.0 # Initial low-pass sharpness for learnable filter
    init_offset: float = 0.0     # Initial cutoff offset for learnable filter

    @property
    def variant_name(self) -> str:
        return f"{self.transform_type}_{self.filter_type}"

    @property
    def is_baseline(self) -> bool:
        """True if this is the uncompressed baseline (gamma=1.0)."""
        return self.gamma >= 1.0


class CompressedAttention(nn.Module):
    """Wrapper that applies spectral compression to KV cache during attention.

    This is not a full attention implementation — it wraps the compression
    logic that is applied to key/value tensors before the model's native
    attention computation.

    Usage:
        compressor = CompressedAttention(config, num_heads=32, head_dim=128)
        # During attention computation:
        k_compressed, v_compressed = compressor.compress_kv(k, v)
        # Then pass compressed K,V to the model's attention
    """

    def __init__(
        self,
        config: CompressionConfig,
        num_heads: int,
        head_dim: int,
    ):
        super().__init__()
        self.config = config
        self.num_heads = num_heads
        self.head_dim = head_dim

        if config.is_baseline:
            self.transform: Optional[SpectralTransform] = None
            self.filter: Optional[SpectralFilter] = None
            logger.info("Initialized baseline (no compression)")
            return

        # Select spectral transform
        if config.transform_type == "dct":
            self.transform = DCTTransform(dim=-2)
        elif config.transform_type == "fft":
            self.transform = FFTTransform(dim=-2)
        else:
            raise ValueError(f"Unknown transform_type: {config.transform_type}")

        # Select spectral filter
        if config.filter_type == "fixed":
            self.filter = FixedLowPassFilter()
        elif config.filter_type == "learnable":
            self.filter = LearnableSpectralFilter(
                num_heads=num_heads,
                max_seq_len=config.max_seq_len,
                head_dim=head_dim,
                init_sharpness=config.init_sharpness,
                init_offset=config.init_offset,
            )
        else:
            raise ValueError(f"Unknown filter_type: {config.filter_type}")

        logger.info(
            f"Initialized compression: {config.variant_name}, gamma={config.gamma}"
        )

    def compress_kv(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compress key and value tensors in the spectral domain.

        Args:
            keys: Key tensor [batch, num_heads, seq_len, head_dim]
            values: Value tensor [batch, num_heads, seq_len, head_dim]

        Returns:
            Tuple of (compressed_keys, compressed_values) — reconstructed to
            the original spatial domain but with high-frequency components removed.
        """
        if self.config.is_baseline:
            return keys, values

        seq_len = keys.shape[-2]
        gamma = self.config.gamma

        # Transform to spectral domain
        k_spectral = self.transform.forward(keys)
        v_spectral = self.transform.forward(values)

        # Apply learnable filter (if using learnable; fixed is a no-op)
        k_spectral = self.filter.forward(k_spectral, gamma)
        v_spectral = self.filter.forward(v_spectral, gamma)

        # Truncate to retain gamma fraction of coefficients
        k_spectral = self.transform.truncate(k_spectral, gamma)
        v_spectral = self.transform.truncate(v_spectral, gamma)

        # Reconstruct to spatial domain
        # The reconstructed tensors have length = gamma * original_seq_len
        # The attention mechanism will operate on this compressed representation
        k_compressed = self.transform.inverse(k_spectral, target_len=seq_len)
        v_compressed = self.transform.inverse(v_spectral, target_len=seq_len)

        return k_compressed, v_compressed

    def get_compressed_cache_size(self, seq_len: int) -> int:
        """Return the number of spectral coefficients retained for a given seq_len.

        This is the effective cache size after compression.
        """
        if self.config.is_baseline:
            return seq_len
        return max(1, int(seq_len * self.config.gamma))

    def get_filter_mask(self) -> Optional[torch.Tensor]:
        """Return the current filter mask (for learnable filters only)."""
        if isinstance(self.filter, LearnableSpectralFilter):
            return self.filter.get_mask()
        return None


def apply_compression_to_model(
    model: nn.Module,
    config: CompressionConfig,
    num_heads: int,
    head_dim: int,
) -> nn.Module:
    """Apply spectral KV-cache compression to a transformer model.

    This function creates a CompressedAttention wrapper and registers it
    as a module attribute on the model. The actual patching of attention
    layers requires model-specific hooks — see the training pipeline for
    integration with HuggingFace's LlamaModel.

    Args:
        model: The transformer model (e.g., LlamaForCausalLM).
        config: Compression configuration.
        num_heads: Number of attention heads.
        head_dim: Dimension per attention head.

    Returns:
        The model with compression wrapper attached.
    """
    compressor = CompressedAttention(config, num_heads, head_dim)
    model.spectral_compressor = compressor

    if not config.is_baseline and config.filter_type == "learnable":
        # Register learnable filter parameters with the model so they're
        # included in the optimizer and checkpointed
        for name, param in compressor.filter.named_parameters():
            model.register_parameter(f"spectral_filter_{name}", param)
            logger.info(f"Registered learnable spectral parameter: spectral_filter_{name}")

    return model
