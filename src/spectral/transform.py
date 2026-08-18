"""Spectral transforms for KV-cache compression.

Two transforms are provided:
  - DCTTransform: Discrete Cosine Transform (FreqKV baseline). Real-valued,
    discards phase information.
  - FFTTransform: Complex-valued Fast Fourier Transform. Preserves phase.

Both operate along the sequence dimension of key/value tensors.

The FreqKV approach (arXiv:2505.00570, ICLR 2026):
  1. Transform the KV cache into the spectral domain.
  2. Truncate to retain only the lowest-frequency gamma fraction.
  3. Store the compressed spectral representation as the cache.
  4. Reconstruct to the spatial domain only when attention needs to
     compute Q @ K^T.

This gives a true O(gamma * N) cache size instead of O(N).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class SpectralTransform(ABC, nn.Module):
    """Base class for spectral transforms applied to KV-cache tensors."""

    def __init__(self, dim: int = -2):
        """
        Args:
            dim: Dimension along which to apply the transform (sequence dimension).
                 Default is -2, assuming tensor shape [batch, num_heads, seq_len, head_dim].
        """
        super().__init__()
        self.dim = dim

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transform a tensor into the spectral domain.

        Args:
            x: Input tensor of shape [batch, num_heads, seq_len, head_dim].

        Returns:
            Spectral representation of the tensor.
        """
        ...

    @abstractmethod
    def inverse(self, x_spectral: torch.Tensor, target_len: int) -> torch.Tensor:
        """Inverse transform from spectral domain back to spatial domain.

        Args:
            x_spectral: Spectral representation (possibly truncated).
            target_len: Target sequence length for reconstruction.

        Returns:
            Reconstructed tensor of shape [batch, num_heads, target_len, head_dim].
        """
        ...

    @abstractmethod
    def truncate(self, x_spectral: torch.Tensor, gamma: float) -> torch.Tensor:
        """Truncate spectral coefficients to retain fraction gamma.

        Args:
            x_spectral: Full spectral representation.
            gamma: Fraction of coefficients to retain (0 < gamma <= 1).

        Returns:
            Truncated spectral representation.
        """
        ...

    @abstractmethod
    def spectral_len(self, seq_len: int) -> int:
        """Return the length of the spectral representation for a given seq_len."""
        ...

    @abstractmethod
    def pad_to_len(self, x_spectral: torch.Tensor, target_spectral_len: int) -> torch.Tensor:
        """Zero-pad spectral coefficients to a target length (for batch operations)."""
        ...


class DCTTransform(SpectralTransform):
    """Discrete Cosine Transform (DCT-II) for KV-cache compression.

    This is the FreqKV baseline transform. The DCT is real-valued and
    discards phase information. Implemented via torch.fft for efficiency
    using the FFT-based DCT algorithm.

    The DCT produces N real coefficients for a length-N input. After
    truncation to gamma*N coefficients, the inverse DCT must zero-pad
    back to N coefficients before reconstructing.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply DCT-II along the sequence dimension.

        Uses the standard FFT-based DCT algorithm:
        DCT(x) = Re(FFT(x_reordered)) * phase_factors
        """
        N = x.shape[self.dim]

        # Rearrange for FFT-based DCT: [x0, x2, x4, ..., x5, x3, x1]
        # This is the standard "interleave" trick for DCT via FFT
        v = torch.cat([x[..., 0::2], x[..., 1::2].flip(self.dim)], dim=self.dim)

        # Apply complex FFT
        V = torch.fft.fft(v, dim=self.dim)

        # Multiply by phase factors to get DCT
        k = torch.arange(N, device=x.device, dtype=x.dtype)
        phase = torch.exp(-1j * torch.pi * k / (2 * N))

        # Reshape phase for broadcasting
        shape = [1] * x.ndim
        shape[self.dim] = N
        phase = phase.reshape(shape)

        dct = (V * phase).real

        return dct

    def inverse(self, x_spectral: torch.Tensor, target_len: int) -> torch.Tensor:
        """Apply inverse DCT (DCT-III) to reconstruct the spatial tensor.

        If x_spectral has been truncated to fewer coefficients than target_len,
        we zero-pad back to target_len before applying the inverse DCT.

        Args:
            x_spectral: Spectral representation (possibly truncated).
            target_len: Original sequence length for reconstruction.
        """
        current_len = x_spectral.shape[self.dim]

        # Zero-pad spectral coefficients back to target_len if truncated
        if current_len < target_len:
            pad_shape = list(x_spectral.shape)
            pad_shape[self.dim] = target_len - current_len
            padding = torch.zeros(pad_shape, device=x_spectral.device, dtype=x_spectral.dtype)
            x_spectral = torch.cat([x_spectral, padding], dim=self.dim)
        elif current_len > target_len:
            # Truncate to target_len (shouldn't happen, but be safe)
            x_spectral = x_spectral.narrow(self.dim, 0, target_len)

        N = target_len

        # DCT-III via FFT
        k = torch.arange(N, device=x_spectral.device, dtype=x_spectral.dtype)
        phase = torch.exp(1j * torch.pi * k / (2 * N))

        shape = [1] * x_spectral.ndim
        shape[self.dim] = N
        phase = phase.reshape(shape)

        V = torch.fft.ifft(x_spectral * phase, dim=self.dim)

        # Undo the interleaving
        half = (target_len + 1) // 2
        x = torch.empty(
            *x_spectral.shape[:-1], target_len,
            device=x_spectral.device, dtype=V.real.dtype,
        )
        x[..., 0::2] = V[..., :half].real
        x[..., 1::2] = V[..., half:].real.flip(self.dim)

        return x

    def truncate(self, x_spectral: torch.Tensor, gamma: float) -> torch.Tensor:
        """Retain the lowest-frequency gamma fraction of DCT coefficients.

        The DCT produces N coefficients for a length-N input. We keep the
        first gamma*N coefficients (lowest frequencies).
        """
        N = x_spectral.shape[self.dim]
        keep = max(1, int(N * gamma))
        return x_spectral.narrow(self.dim, 0, keep)

    def spectral_len(self, seq_len: int) -> int:
        """DCT produces N real coefficients for a length-N input."""
        return seq_len

    def pad_to_len(self, x_spectral: torch.Tensor, target_spectral_len: int) -> torch.Tensor:
        """Zero-pad DCT coefficients to a target length."""
        current = x_spectral.shape[self.dim]
        if current >= target_spectral_len:
            return x_spectral.narrow(self.dim, 0, target_spectral_len)
        pad_shape = list(x_spectral.shape)
        pad_shape[self.dim] = target_spectral_len - current
        padding = torch.zeros(pad_shape, device=x_spectral.device, dtype=x_spectral.dtype)
        return torch.cat([x_spectral, padding], dim=self.dim)


class FFTTransform(SpectralTransform):
    """Complex-valued Fast Fourier Transform for KV-cache compression.

    Unlike the DCT, the complex FFT preserves phase information.
    The full proposed method uses this transform.

    Uses rfft/irfft for real-valued input, which returns N//2+1 complex
    coefficients for a length-N input.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply complex FFT along the sequence dimension.

        The input is real-valued, so we use rfft which returns only the
        non-redundant half of the spectrum. For reconstruction, we use irfft.
        """
        return torch.fft.rfft(x, dim=self.dim)

    def inverse(self, x_spectral: torch.Tensor, target_len: int) -> torch.Tensor:
        """Apply inverse FFT to reconstruct the spatial tensor.

        Args:
            x_spectral: Truncated spectral representation from rfft.
            target_len: Original sequence length for irfft reconstruction.
        """
        return torch.fft.irfft(x_spectral, n=target_len, dim=self.dim)

    def truncate(self, x_spectral: torch.Tensor, gamma: float) -> torch.Tensor:
        """Retain the lowest-frequency gamma fraction of FFT coefficients.

        For rfft output of length N//2+1, we keep the first gamma fraction.
        """
        N = x_spectral.shape[self.dim]
        keep = max(1, int(N * gamma))
        return x_spectral.narrow(self.dim, 0, keep)

    def spectral_len(self, seq_len: int) -> int:
        """rfft returns N//2+1 complex coefficients for a length-N input."""
        return seq_len // 2 + 1

    def pad_to_len(self, x_spectral: torch.Tensor, target_spectral_len: int) -> torch.Tensor:
        """Zero-pad FFT coefficients to a target length."""
        current = x_spectral.shape[self.dim]
        if current >= target_spectral_len:
            return x_spectral.narrow(self.dim, 0, target_spectral_len)
        pad_shape = list(x_spectral.shape)
        pad_shape[self.dim] = target_spectral_len - current
        padding = torch.zeros(pad_shape, device=x_spectral.device, dtype=x_spectral.dtype)
        return torch.cat([x_spectral, padding], dim=self.dim)
