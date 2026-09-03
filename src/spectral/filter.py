"""Spectral filters for selecting which frequency components to retain.

Two filter types:
  - FixedLowPassFilter: Retains a fixed fraction of lowest-frequency components.
    This is the FreqKV baseline approach.
  - LearnableSpectralFilter: A per-layer, per-head learnable soft mask that
    adaptively selects frequency bands. Initialized to approximate a low-pass
    filter for stable training.

The 2x2 factorial design crosses these filter types with the transform types
(DCT vs FFT) from transform.py:
  - DCT + Fixed     = FreqKV baseline (parameter-free)
  - DCT + Learnable = Adaptive DCT (novel contribution)
  - FFT + Fixed     = Complex FFT with fixed LP (novel: phase-preserving)
  - FFT + Learnable = Full proposed method (complex FFT + adaptive mask)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class SpectralFilter(ABC, nn.Module):
    """Base class for spectral frequency filters."""

    @abstractmethod
    def forward(self, x_spectral: torch.Tensor, gamma: float) -> torch.Tensor:
        """Apply the filter to spectral coefficients.

        Args:
            x_spectral: Spectral representation [batch, num_heads, freq_len, head_dim].
            gamma: Target compression ratio (0 < gamma <= 1).

        Returns:
            Filtered and truncated spectral representation.
        """
        ...


class FixedLowPassFilter(SpectralFilter):
    """Fixed low-pass filter: retains the lowest-frequency components.

    This is the FreqKV baseline. No learnable parameters.
    Truncation is handled by the spectral transform's truncate() method
    when SpectralKVCache.compress() is called.

    This class exists for interface consistency in the 2x2 factorial design.
    """

    def forward(self, x_spectral: torch.Tensor, gamma: float) -> torch.Tensor:
        """Pass through -- truncation is done by the transform.

        This filter is a no-op; the spectral transform's truncate() method
        performs the actual low-pass selection.
        """
        return x_spectral


class LearnableSpectralFilter(SpectralFilter):
    """Learnable per-layer, per-head spectral filter.

    Parameterized as a sigmoid soft mask over spectral coefficients.
    Initialized to approximate a low-pass filter for stable training:
    the mask starts as a smooth decay from 1.0 (low freq) to 0.0 (high freq).

    The filter has separate parameters for each head in each layer, allowing
    different heads to retain different frequency bands -- important for
    symbolic and mathematical content that may require high-frequency components.

    The gamma parameter is used as a soft budget: after applying the learned
    mask, the effective retained fraction is regularized toward gamma via
    the mask's initialization and training dynamics.
    """

    def __init__(
        self,
        num_heads: int,
        max_spectral_len: int,
        init_sharpness: float = 10.0,
        init_offset: float = 0.0,
    ):
        """
        Args:
            num_heads: Number of attention heads in this layer.
            max_spectral_len: Maximum number of spectral coefficients
                              (= seq_len for DCT, seq_len//2+1 for FFT).
            init_sharpness: Controls how sharply the initial mask transitions
                           from 1 to 0. Higher = sharper low-pass cutoff.
            init_offset: Shifts the initial cutoff. 0.0 = cutoff at Nyquist/2.
        """
        super().__init__()

        # The learnable parameters: one logit per frequency bin per head
        # Shape: [num_heads, max_spectral_len]
        freq_norm = torch.linspace(0, 1, max_spectral_len)  # [0, 1] across frequencies
        cutoff = 0.5 + init_offset  # Default cutoff at half the spectrum

        # Shape: [num_heads, max_spectral_len] -- same init for all heads, but they'll diverge
        init_logits = init_sharpness * (cutoff - freq_norm).unsqueeze(0).expand(num_heads, -1)

        self.filter_logits = nn.Parameter(init_logits.clone())
        self.max_spectral_len = max_spectral_len

    def forward(self, x_spectral: torch.Tensor, gamma: float) -> torch.Tensor:
        """Apply the learnable soft mask to spectral coefficients.

        Args:
            x_spectral: Spectral representation [batch, num_heads, freq_len, head_dim].
            gamma: Target compression ratio (used for regularization loss;
                   the mask is not hard-thresholded by gamma).

        Returns:
            Filtered spectral representation (same shape as input).
        """
        freq_len = x_spectral.shape[-2]  # Current frequency dimension
        num_heads = x_spectral.shape[-3]

        # Get the mask for the current frequency length
        # Slice or interpolate the learned mask to match the current spectral resolution
        if freq_len <= self.max_spectral_len:
            mask_logits = self.filter_logits[:, :freq_len]  # [num_heads, freq_len]
        else:
            # Interpolate if sequence is longer than max (shouldn't happen normally)
            mask_logits = torch.nn.functional.interpolate(
                self.filter_logits.unsqueeze(0),  # [1, num_heads, max_spectral_len]
                size=freq_len,
                mode='linear',
                align_corners=False,
            ).squeeze(0)  # [num_heads, freq_len]

        # Apply sigmoid to get soft mask [0, 1]
        mask = torch.sigmoid(mask_logits)  # [num_heads, freq_len]

        # Reshape for broadcasting: [1, num_heads, freq_len, 1]
        mask = mask.unsqueeze(0).unsqueeze(-1)

        # Apply mask
        x_filtered = x_spectral * mask

        return x_filtered

    def get_mask(self) -> torch.Tensor:
        """Return the current filter mask (sigmoid of logits) for analysis.

        Returns:
            Tensor of shape [num_heads, max_spectral_len] with values in [0, 1].
        """
        return torch.sigmoid(self.filter_logits)

    def get_retained_fraction(self) -> torch.Tensor:
        """Return the effective fraction of frequencies retained per head.

        Returns:
            Tensor of shape [num_heads] with values in [0, 1].
        """
        return self.get_mask().mean(dim=-1)
