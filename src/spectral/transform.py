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
    discards phase information. Implemented via a manual FFT-based DCT algorithm.

    The DCT produces N real coefficients for a length-N input. After
    truncation to gamma*N coefficients, the inverse DCT must zero-pad
    back to N coefficients before reconstructing.

    Fix 4: DCT index/phase tensors are precomputed and cached as
    non-persistent buffers keyed by (N, device) to avoid recomputing
    them on every forward/inverse call.
    """

    def __init__(self, dim: int = -2):
        super().__init__(dim=dim)
        # Cache for precomputed tensors: {(N, device): (perm, phase, mirror_idx)}
        # These are NOT nn buffers (they vary by N); they are memoized per-call.
        self._fwd_cache: dict = {}
        self._inv_cache: dict = {}

    def _get_fwd_constants(self, N: int, device: torch.device, dtype: torch.dtype):
        """Get or precompute forward DCT constants for length N.

        Returns:
            (perm, phase) where:
              perm: interleaving permutation indices [N]
              phase: DCT-II phase factors [N] (complex)
        """
        key = (N, str(device))
        if key not in self._fwd_cache:
            even_idx = torch.arange(0, N, 2, device=device)
            odd_idx = torch.arange(1, N, 2, device=device)
            perm = torch.cat([even_idx, odd_idx.flip(0)])
            k = torch.arange(N, device=device, dtype=dtype)
            phase = torch.exp(-1j * torch.pi * k / (2 * N))
            self._fwd_cache[key] = (perm, phase)
        return self._fwd_cache[key]

    def _get_inv_constants(self, N: int, device: torch.device, dtype: torch.dtype):
        """Get or precompute inverse DCT constants for length N.

        Returns:
            (mirror_idx, phase, even_idx, odd_idx) where:
              mirror_idx: mirror permutation for Hermitian reconstruction [N]
              phase: inverse DCT phase factors [N] (complex)
              even_idx: even indices for de-interleaving [N//2 or (N+1)//2]
              odd_idx: odd indices for de-interleaving [N//2]
        """
        key = (N, str(device))
        if key not in self._inv_cache:
            mirror_idx = torch.cat([
                torch.zeros(1, dtype=torch.long, device=device),
                torch.arange(N - 1, 0, -1, device=device),
            ])
            k = torch.arange(N, device=device, dtype=dtype)
            phase = torch.exp(1j * torch.pi * k / (2 * N))
            even_idx = torch.arange(0, N, 2, device=device)
            odd_idx = torch.arange(1, N, 2, device=device)
            self._inv_cache[key] = (mirror_idx, phase, even_idx, odd_idx)
        return self._inv_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply DCT-II along the sequence dimension.

        Uses the standard interleaving trick to compute DCT-II via FFT.
        Returns N real coefficients for a length-N input.

        The interleaving permutation v = [x_0, x_2, ..., x_{N-1},
        x_{N-2}, ..., x_1] is applied via index_select along self.dim
        (NOT ``x[..., 0::2]`` which would slice the last dimension).
        """
        N = x.shape[self.dim]
        dim = self.dim if self.dim >= 0 else x.ndim + self.dim

        # Use precomputed constants (Fix 4)
        perm, phase = self._get_fwd_constants(N, x.device, x.dtype)

        v = x.index_select(dim, perm)

        # Apply complex FFT
        V = torch.fft.fft(v, dim=dim)

        # Multiply by phase factors to get DCT-II coefficients
        shape = [1] * x.ndim
        shape[dim] = N
        phase = phase.reshape(shape)

        dct = (V * phase).real
        return dct

    def inverse(self, x_spectral: torch.Tensor, target_len: int) -> torch.Tensor:
        """Apply inverse DCT (DCT-III) to reconstruct the spatial tensor.

        If x_spectral has been truncated to fewer coefficients than
        target_len, we zero-pad back to target_len before applying the
        inverse.

        The forward DCT-II computes X_k = Re(W_k) where
        W_k = exp(-j*pi*k/(2N)) * FFT(v)_k and v is the interleaved
        permutation of x.  Since v is real, FFT(v) has Hermitian
        symmetry, which constrains the imaginary part of W_k:

            Im(W_k) = -X_{N-k}   for k = 1 .. N-1
            Im(W_0) = 0

        So W_k = X_k - j*X_{N-k} (with X_mirror[0]=0), and we recover
        v = IFFT(W * exp(j*pi*k/(2N))) which is guaranteed real by the
        Hermitian symmetry of the reconstructed spectrum.  No X_0
        scaling or 2x factor is needed — this is an exact
        reconstruction, not the normalized DCT-III formula.

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
            x_spectral = x_spectral.narrow(self.dim, 0, target_len)

        N = target_len
        dim = self.dim if self.dim >= 0 else x_spectral.ndim + self.dim

        # Use precomputed constants (Fix 4)
        mirror_idx, phase, even_idx, odd_idx = self._get_inv_constants(
            N, x_spectral.device, x_spectral.dtype
        )

        # Build the mirror index: X_mirror[k] = X_{N-k} for k=1..N-1, 0 for k=0
        X_mirror = x_spectral.index_select(dim, mirror_idx)
        # Zero out position 0 (Im(W_0) = 0)
        zero_sel = [slice(None)] * x_spectral.ndim
        zero_sel[dim] = 0
        X_mirror[tuple(zero_sel)] = 0

        # Reconstruct complex spectrum: W = X - j * X_mirror
        W = torch.complex(x_spectral, -X_mirror)

        # Apply phase: V = W * exp(j*pi*k/(2N))
        shape = [1] * x_spectral.ndim
        shape[dim] = N
        phase = phase.reshape(shape)
        V = W * phase

        # IFFT to get interleaved permutation (guaranteed real by construction)
        v = torch.fft.ifft(V, dim=dim)

        # Undo interleaving: v = [x_0, x_2, ..., x_{N-1}, x_{N-2}, ..., x_1]
        half = (N + 1) // 2
        x = torch.empty_like(x_spectral)
        even_sel = [slice(None)] * x_spectral.ndim
        even_sel[dim] = even_idx
        odd_sel = [slice(None)] * x_spectral.ndim
        odd_sel[dim] = odd_idx
        v_first = v.narrow(dim, 0, half)
        v_second = v.narrow(dim, half, N - half)
        x[tuple(even_sel)] = v_first.real
        x[tuple(odd_sel)] = v_second.real.flip(dim)

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
