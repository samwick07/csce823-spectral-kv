"""Spectral transforms for KV-cache compression.

Implements the frequency-domain compression from FreqKV (arXiv:2505.00570, ICLR 2026),
extended with a complex FFT variant and learnable spectral filtering.

Two transforms are provided:
  - DCTTransform: Discrete Cosine Transform (DCT-II) with ortho normalization.
    Real-valued, discards phase information. This is the FreqKV baseline.
  - FFTTransform: Complex-valued Fast Fourier Transform (rfft/irfft).
    Preserves phase. Novel contribution beyond FreqKV.

Both operate along the sequence dimension of key/value tensors.
The compression flow is:
  1. Transform the KV tensor into the spectral domain.
  2. Apply the spectral filter (learnable mask or no-op for fixed).
  3. Truncate to retain only compress_len coefficients.
  4. Inverse transform back to the spatial domain (at reduced length).
  5. Scale by sqrt(compress_len / original_len) to compensate for the
     amplitude change when going from N to L samples in the ortho-normalized
     DCT/IDCT pair.

This gives a compressed tensor of length compress_len (NOT original_len with
lossy reconstruction). The compressed "tokens" represent the low-frequency
structure of the original tokens at reduced resolution.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class SpectralTransform(ABC, nn.Module):
    """Base class for spectral transforms applied to KV-cache tensors.

    Provides the common compress() flow. Subclasses implement forward(),
    inverse(), truncate(), and spectral_len().
    """

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

    def compress(
        self,
        x: torch.Tensor,
        compress_len: int,
        filter_fn=None,
        gamma: float = 0.5,
    ) -> torch.Tensor:
        """Compress x from length N to length compress_len in the spectral domain.

        This is the core compression method, following FreqKV's dct_compress:
          1. Transform to spectral domain (forward).
          2. Apply learnable filter (if provided) on the FULL spectrum.
          3. Truncate to compress_len coefficients.
          4. Inverse transform to spatial domain at length compress_len.
          5. Scale by sqrt(compress_len / original_len).

        The filter is applied BEFORE truncation (Q13 decision): the filter sees
        the full spectrum and can learn to retain high-frequency components that
        truncation would discard. Truncation then enforces the compression budget.

        Args:
            x: Input tensor [B, H, N, D] where N is the sequence length.
            compress_len: Target compressed length L (L < N).
            filter_fn: Optional callable(x_spectral, gamma) -> filtered spectral.
                       If None, no filtering (fixed low-pass via truncation only).
            gamma: Target compression ratio (used by filter for regularization).

        Returns:
            Compressed tensor [B, H, L, D] where L = compress_len.
        """
        seq_len = x.shape[self.dim]

        # No compression needed
        if compress_len >= seq_len:
            return x

        # No compression for empty
        if compress_len == 0:
            shape = list(x.shape)
            shape[self.dim] = 0
            return torch.empty(shape, device=x.device, dtype=x.dtype)

        # 1. Transform to spectral domain
        x_spectral = self.forward(x)

        # 2. Apply learnable filter on FULL spectrum (before truncation)
        if filter_fn is not None:
            x_spectral = filter_fn(x_spectral, gamma)

        # 3. Truncate to compress_len coefficients
        x_spectral = self.truncate(x_spectral, compress_len / max(seq_len, 1))
        # Ensure exactly compress_len
        current_len = x_spectral.shape[self.dim]
        if current_len > compress_len:
            x_spectral = x_spectral.narrow(
                self.dim if self.dim >= 0 else x_spectral.ndim + self.dim,
                0, compress_len
            )

        # 4. Inverse transform to spatial domain at compressed length
        x_compressed = self.inverse(x_spectral, compress_len)

        # 5. Scale by sqrt(L / N) to compensate for amplitude change
        scale = math.sqrt(compress_len / seq_len)
        x_compressed = x_compressed * scale

        return x_compressed


class DCTTransform(SpectralTransform):
    """Discrete Cosine Transform (DCT-II) with ortho normalization.

    This is the FreqKV baseline transform. Uses ortho-normalized DCT/IDCT
    via the standard FFT-based interleaving trick.

    The DCT produces N real coefficients for a length-N input. After
    truncation to L coefficients, the inverse DCT produces a length-L
    output that is the best low-pass approximation of the input.

    Constants (permutation indices, phase factors) are precomputed and
    cached per (N, device) to avoid recomputation.
    """

    def __init__(self, dim: int = -2):
        super().__init__(dim=dim)
        # Cache for precomputed tensors: {(N, device, dtype): (constants)}
        self._fwd_cache: dict = {}
        self._inv_cache: dict = {}

    def _get_fwd_constants(self, N: int, device: torch.device, dtype: torch.dtype):
        """Get or precompute forward DCT constants for length N.

        Returns:
            (perm, phase, norm_factors) where:
              perm: interleaving permutation indices [N]
              phase: DCT-II phase factors [N] (complex)
              norm_factors: ortho normalization [N]
        """
        key = (N, str(device))
        if key not in self._fwd_cache:
            even_idx = torch.arange(0, N, 2, device=device)
            odd_idx = torch.arange(1, N, 2, device=device)
            perm = torch.cat([even_idx, odd_idx.flip(0)])

            k = torch.arange(N, device=device, dtype=torch.float32)
            phase = torch.exp(-1j * torch.pi * k / (2 * N))

            # Ortho normalization factors
            norm = torch.ones(N, device=device, dtype=torch.float32)
            norm[0] = 1.0 / (math.sqrt(N) * 2)
            norm[1:] = 1.0 / (math.sqrt(N / 2) * 2)

            self._fwd_cache[key] = (perm, phase, norm)
        return self._fwd_cache[key]

    def _get_inv_constants(self, N: int, device: torch.device, dtype: torch.dtype):
        """Get or precompute inverse DCT constants for length N.

        Returns:
            (mirror_idx, phase, norm_factors, even_idx, odd_idx)
        """
        key = (N, str(device))
        if key not in self._inv_cache:
            mirror_idx = torch.cat([
                torch.zeros(1, dtype=torch.long, device=device),
                torch.arange(N - 1, 0, -1, device=device),
            ])

            k = torch.arange(N, device=device, dtype=torch.float32)
            phase = torch.exp(1j * torch.pi * k / (2 * N))

            # Ortho denormalization factors
            norm = torch.ones(N, device=device, dtype=torch.float32)
            norm[0] = math.sqrt(N) * 2
            norm[1:] = math.sqrt(N / 2) * 2

            even_idx = torch.arange(0, N, 2, device=device)
            odd_idx = torch.arange(1, N, 2, device=device)

            self._inv_cache[key] = (mirror_idx, phase, norm, even_idx, odd_idx)
        return self._inv_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply DCT-II with ortho normalization along the sequence dimension.

        Uses the standard interleaving trick to compute DCT-II via FFT.
        Returns N real coefficients for a length-N input.

        The interleaving permutation v = [x_0, x_2, ..., x_{N-1},
        x_{N-2}, ..., x_1] is applied via index_select along self.dim.
        """
        N = x.shape[self.dim]
        dim = self.dim if self.dim >= 0 else x.ndim + self.dim

        # torch.fft doesn't support BFloat16 — cast to float32, restore after
        orig_dtype = x.dtype
        if orig_dtype != torch.float32:
            x = x.to(torch.float32)

        perm, phase, norm = self._get_fwd_constants(N, x.device, x.dtype)

        v = x.index_select(dim, perm)

        # Apply complex FFT
        V = torch.fft.fft(v, dim=dim)

        # Multiply by phase factors to get DCT-II coefficients
        shape = [1] * x.ndim
        shape[dim] = N
        phase = phase.reshape(shape)
        norm = norm.reshape(shape)

        dct = (V * phase).real * norm * 2

        # Restore original dtype
        if orig_dtype != torch.float32:
            dct = dct.to(orig_dtype)
        return dct

    def inverse(self, x_spectral: torch.Tensor, target_len: int) -> torch.Tensor:
        """Apply inverse DCT (DCT-III) with ortho normalization.

        If x_spectral has been truncated to fewer coefficients than
        target_len, we zero-pad back to target_len before applying the
        inverse. The output has length target_len.

        Uses Hermitian symmetry reconstruction from real DCT coefficients:
            W_k = X_k - j*X_{N-k}  (with X_mirror[0]=0)
        Then v = IFFT(W * exp(j*pi*k/(2N))) which is guaranteed real.
        """
        current_len = x_spectral.shape[self.dim]

        # torch.fft doesn't support BFloat16 — cast to float32
        orig_dtype = x_spectral.dtype
        if orig_dtype != torch.float32:
            x_spectral = x_spectral.to(torch.float32)

        # Zero-pad spectral coefficients back to target_len if truncated
        if current_len < target_len:
            pad_shape = list(x_spectral.shape)
            dim = self.dim if self.dim >= 0 else x_spectral.ndim + self.dim
            pad_shape[dim] = target_len - current_len
            padding = torch.zeros(pad_shape, device=x_spectral.device, dtype=x_spectral.dtype)
            x_spectral = torch.cat([x_spectral, padding], dim=self.dim)
        elif current_len > target_len:
            x_spectral = x_spectral.narrow(self.dim, 0, target_len)

        N = target_len
        dim = self.dim if self.dim >= 0 else x_spectral.ndim + self.dim

        mirror_idx, phase, norm, even_idx, odd_idx = self._get_inv_constants(
            N, x_spectral.device, x_spectral.dtype
        )

        # Apply denormalization
        shape = [1] * x_spectral.ndim
        shape[dim] = N
        norm = norm.reshape(shape)
        X_v = x_spectral / 2 * norm

        # Build the mirror index: X_mirror[k] = X_{N-k} for k=1..N-1, 0 for k=0
        X_mirror = X_v.index_select(dim, mirror_idx)
        # Zero out position 0 (Im(W_0) = 0)
        zero_sel = [slice(None)] * x_spectral.ndim
        zero_sel[dim] = 0
        X_mirror[tuple(zero_sel)] = 0

        # Reconstruct complex spectrum: W = X - j * X_mirror
        W = torch.complex(X_v, -X_mirror)

        # Apply phase: V = W * exp(j*pi*k/(2N))
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

        # Restore original dtype
        if orig_dtype != torch.float32:
            x = x.to(orig_dtype)
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
    This is the project's novel contribution beyond FreqKV.

    Uses rfft/irfft for real-valued input, which returns N//2+1 complex
    coefficients for a length-N input. After truncation, irfft reconstructs
    to the target length.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply complex FFT along the sequence dimension.

        The input is real-valued, so we use rfft which returns only the
        non-redundant half of the spectrum. For reconstruction, we use irfft.
        """
        # torch.fft doesn't support BFloat16 — cast to float32
        orig_dtype = x.dtype
        if orig_dtype != torch.float32:
            x = x.to(torch.float32)
        return torch.fft.rfft(x, dim=self.dim)

    def inverse(self, x_spectral: torch.Tensor, target_len: int) -> torch.Tensor:
        """Apply inverse FFT to reconstruct the spatial tensor.

        Args:
            x_spectral: Truncated spectral representation from rfft.
            target_len: Target sequence length for irfft reconstruction.
        """
        # torch.fft doesn't support BFloat16 — cast to float32
        orig_dtype = x_spectral.dtype
        if orig_dtype == torch.bfloat16:
            x_spectral = x_spectral.to(torch.float32)
        result = torch.fft.irfft(x_spectral, n=target_len, dim=self.dim)
        # irfft returns float32 or float64; restore BFloat16 if needed
        if orig_dtype == torch.bfloat16:
            result = result.to(orig_dtype)
        return result

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
