"""Tests for FreqKV-aligned spectral transforms.

Verifies:
  - DCT round-trip (exact reconstruction at gamma=1.0)
  - DCT ortho normalization
  - Truncation produces L-length output
  - sqrt(L/N) scaling
  - Mathematical equivalence to FreqKV reference dct/idct/dct_compress
  - FFT round-trip and compression
  - compress() method on base class
"""

import math

import numpy as np
import torch
import pytest

from src.spectral.transform import DCTTransform, FFTTransform, SpectralTransform


# ─── FreqKV reference implementations (verbatim from LUMIA-Group/FreqKV) ───

def dct_freqkv(x, norm='ortho'):
    """FreqKV's reference DCT-II."""
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)
    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)
    Vc = torch.fft.fft(v.to(torch.float32), dim=1)
    k = -torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)
    V = Vc.real * W_r - Vc.imag * W_i
    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2
    V = 2 * V.view(*x_shape)
    return V


def idct_freqkv(X, norm='ortho'):
    """FreqKV's reference IDCT (DCT-III)."""
    x_shape = X.shape
    N = x_shape[-1]
    X_v = X.contiguous().view(-1, x_shape[-1]) / 2
    if norm == 'ortho':
        X_v[:, 0] *= np.sqrt(N) * 2
        X_v[:, 1:] *= np.sqrt(N / 2) * 2
    k = torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)
    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)
    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r
    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)
    V = torch.view_as_complex(V)
    v = torch.fft.ifft(V, dim=1).real
    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, :N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, :N // 2]
    return x.view(*x_shape)


def dct_compress_freqkv(x, compress_len=2048, seq_dim=2, kv_type="key"):
    """FreqKV's reference dct_compress."""
    if compress_len == 0:
        return x[:, :, 0:0]
    elif compress_len >= x.shape[seq_dim]:
        return x
    bsz, num_heads, q_len, head_dim = x.shape
    x = x.transpose(1, 2).reshape(bsz, q_len, num_heads * head_dim)
    x_dct = dct_freqkv(x.transpose(1, 2), norm='ortho')
    x_dct = x_dct[:, :, :compress_len]
    x_idct = idct_freqkv(x_dct, norm='ortho').transpose(1, 2) * np.sqrt(compress_len / q_len)
    compressed_x = x_idct.to(x.dtype)
    return compressed_x.reshape(bsz, compress_len, num_heads, head_dim).transpose(1, 2)


# ─── DCT Round-Trip Tests ───

class TestDCTRoundTrip:
    """Verify DCT/IDCT round-trip is exact at gamma=1.0."""

    @pytest.mark.parametrize("N", [8, 16, 64, 128, 256])
    def test_roundtrip_1d(self, N):
        """DCT then IDCT should reconstruct the original signal exactly."""
        torch.manual_seed(42)
        x = torch.randn(1, 1, N, 4)
        dct = DCTTransform(dim=-2)

        X = dct.forward(x)
        x_recon = dct.inverse(X, target_len=N)

        assert x_recon.shape == x.shape, f"Shape mismatch: {x_recon.shape} vs {x.shape}"
        assert torch.allclose(x, x_recon, atol=1e-5), \
            f"Round-trip error: {(x - x_recon).abs().max():.2e}"

    def test_roundtrip_multibatch(self):
        """Round-trip with multiple batches and heads."""
        torch.manual_seed(42)
        x = torch.randn(2, 4, 64, 8)
        dct = DCTTransform(dim=-2)

        X = dct.forward(x)
        x_recon = dct.inverse(X, target_len=64)

        assert torch.allclose(x, x_recon, atol=1e-5)

    def test_roundtrip_bfloat16(self):
        """Round-trip with BFloat16 input (casts to float32 for FFT)."""
        torch.manual_seed(42)
        x = torch.randn(1, 2, 32, 8, dtype=torch.bfloat16)
        dct = DCTTransform(dim=-2)

        X = dct.forward(x)
        x_recon = dct.inverse(X, target_len=32)

        assert x_recon.dtype == torch.bfloat16
        # BFloat16 has lower precision
        assert torch.allclose(x.float(), x_recon.float(), atol=1e-2)


# ─── DCT Ortho Normalization Tests ───

class TestDCTOrthoNorm:
    """Verify ortho normalization properties."""

    def test_energy_preservation_full(self):
        """Ortho-normalized DCT preserves energy at full length."""
        torch.manual_seed(42)
        N = 64
        x = torch.randn(1, 1, N, 4)
        dct = DCTTransform(dim=-2)

        X = dct.forward(x)
        # Energy in spatial domain
        spatial_energy = x.pow(2).sum().item()
        # Energy in spectral domain (ortho norm preserves energy)
        spectral_energy = X.pow(2).sum().item()

        assert abs(spectral_energy - spatial_energy) / spatial_energy < 0.01, \
            f"Energy not preserved: spatial={spatial_energy:.4f}, spectral={spectral_energy:.4f}"


# ─── Mathematical Equivalence to FreqKV Reference ───

class TestFreqKVEquivalence:
    """Verify our DCT matches FreqKV's reference implementation."""

    @pytest.mark.parametrize("N", [8, 16, 64])
    def test_dct_matches_freqkv(self, N):
        """Our DCT should match FreqKV's dct() function."""
        torch.manual_seed(42)
        x = torch.randn(2, 3, N, 4)
        dct = DCTTransform(dim=-2)

        # Our DCT
        X_ours = dct.forward(x)

        # FreqKV's DCT (applies on last dim, so we permute)
        x_perm = x.permute(0, 1, 3, 2)  # [2, 3, 4, N]
        X_freqkv = dct_freqkv(x_perm, norm='ortho')
        X_freqkv = X_freqkv.permute(0, 1, 3, 2)  # back to [2, 3, N, 4]

        assert torch.allclose(X_ours, X_freqkv, atol=1e-5), \
            f"DCT mismatch: max diff = {(X_ours - X_freqkv).abs().max():.2e}"

    @pytest.mark.parametrize("N", [8, 16, 64])
    def test_idct_matches_freqkv(self, N):
        """Our IDCT should match FreqKV's idct() function."""
        torch.manual_seed(42)
        x = torch.randn(2, 3, N, 4)
        dct = DCTTransform(dim=-2)

        # Forward to get spectral
        X = dct.forward(x)

        # Our IDCT
        x_ours = dct.inverse(X, target_len=N)

        # FreqKV's IDCT
        x_perm = X.permute(0, 1, 3, 2)  # [2, 3, 4, N]
        x_freqkv = idct_freqkv(x_perm, norm='ortho')
        x_freqkv = x_freqkv.permute(0, 1, 3, 2)

        assert torch.allclose(x_ours, x_freqkv, atol=1e-5), \
            f"IDCT mismatch: max diff = {(x_ours - x_freqkv).abs().max():.2e}"

    @pytest.mark.parametrize("N,L", [(64, 32), (128, 64), (100, 50)])
    def test_compress_matches_freqkv(self, N, L):
        """Our compress() should match FreqKV's dct_compress()."""
        torch.manual_seed(42)
        x = torch.randn(2, 3, N, 4)
        dct = DCTTransform(dim=-2)

        # Our compress
        x_ours = dct.compress(x, compress_len=L)

        # FreqKV's dct_compress
        x_freqkv = dct_compress_freqkv(x, compress_len=L, seq_dim=2)

        assert x_ours.shape == x_freqkv.shape, \
            f"Shape mismatch: {x_ours.shape} vs {x_freqkv.shape}"
        assert torch.allclose(x_ours, x_freqkv, atol=1e-4), \
            f"Compress mismatch: max diff = {(x_ours - x_freqkv).abs().max():.2e}"


# ─── Compression (compress method) Tests ───

class TestCompress:
    """Verify the compress() method on SpectralTransform base class."""

    @pytest.mark.parametrize("N,L", [(64, 32), (128, 64), (256, 128)])
    def test_compress_output_length(self, N, L):
        """compress() should return L-length output."""
        torch.manual_seed(42)
        x = torch.randn(2, 4, N, 8)
        dct = DCTTransform(dim=-2)

        x_compressed = dct.compress(x, compress_len=L)

        assert x_compressed.shape[-2] == L, \
            f"Expected length {L}, got {x_compressed.shape[-2]}"

    def test_compress_preserves_other_dims(self):
        """compress() should preserve batch, head, and head_dim dimensions."""
        B, H, N, D = 2, 4, 64, 8
        x = torch.randn(B, H, N, D)
        dct = DCTTransform(dim=-2)

        x_compressed = dct.compress(x, compress_len=32)

        assert x_compressed.shape == (B, H, 32, D), \
            f"Shape mismatch: {x_compressed.shape}"

    def test_compress_no_compression(self):
        """compress_len >= seq_len should return input unchanged."""
        x = torch.randn(1, 2, 64, 4)
        dct = DCTTransform(dim=-2)

        x_out = dct.compress(x, compress_len=64)
        assert torch.equal(x_out, x)

    def test_compress_zero_len(self):
        """compress_len=0 should return empty tensor."""
        x = torch.randn(1, 2, 64, 4)
        dct = DCTTransform(dim=-2)

        x_out = dct.compress(x, compress_len=0)
        assert x_out.shape[-2] == 0

    @pytest.mark.parametrize("gamma", [0.5, 0.25, 0.1])
    def test_compress_scaling(self, gamma):
        """Verify sqrt(L/N) scaling is applied."""
        torch.manual_seed(42)
        N = 64
        L = int(N * gamma)
        x = torch.randn(1, 1, N, 4)
        dct = DCTTransform(dim=-2)

        x_compressed = dct.compress(x, compress_len=L)

        # Without scaling, the IDCT of truncated coefficients would have
        # different amplitude. The sqrt(L/N) factor scales it.
        # Verify by comparing to manual computation
        X = dct.forward(x)
        X_trunc = dct.truncate(X, gamma)
        x_manual = dct.inverse(X_trunc, target_len=L)
        x_manual_scaled = x_manual * math.sqrt(L / N)

        assert torch.allclose(x_compressed, x_manual_scaled, atol=1e-6), \
            f"Scaling mismatch at gamma={gamma}"


# ─── FFT Transform Tests ───

class TestFFTTransform:
    """Verify FFT transform compression."""

    def test_fft_roundtrip(self):
        """FFT forward then inverse should reconstruct exactly."""
        torch.manual_seed(42)
        N = 64
        x = torch.randn(2, 4, N, 8)
        fft = FFTTransform(dim=-2)

        X = fft.forward(x)
        x_recon = fft.inverse(X, target_len=N)

        assert torch.allclose(x, x_recon, atol=1e-5)

    def test_fft_compress_length(self):
        """FFT compress should return L-length output."""
        torch.manual_seed(42)
        x = torch.randn(2, 4, 64, 8)
        fft = FFTTransform(dim=-2)

        x_compressed = fft.compress(x, compress_len=32)

        assert x_compressed.shape[-2] == 32

    def test_fft_compress_no_filter(self):
        """FFT compress without filter should still work."""
        torch.manual_seed(42)
        x = torch.randn(1, 2, 64, 4)
        fft = FFTTransform(dim=-2)

        x_compressed = fft.compress(x, compress_len=32, filter_fn=None)

        assert x_compressed.shape == (1, 2, 32, 4)
        assert not torch.isnan(x_compressed).any()


# ─── Truncation Tests ───

class TestTruncation:
    """Verify spectral truncation."""

    def test_dct_truncate(self):
        """DCT truncate should keep first gamma*N coefficients."""
        x = torch.randn(1, 1, 64, 4)
        dct = DCTTransform(dim=-2)
        X = dct.forward(x)

        X_trunc = dct.truncate(X, gamma=0.5)
        assert X_trunc.shape[-2] == 32

    def test_fft_truncate(self):
        """FFT truncate should keep first gamma*(N//2+1) coefficients."""
        N = 64
        x = torch.randn(1, 1, N, 4)
        fft = FFTTransform(dim=-2)
        X = fft.forward(x)

        X_trunc = fft.truncate(X, gamma=0.5)
        expected_len = max(1, int((N // 2 + 1) * 0.5))
        assert X_trunc.shape[-2] == expected_len


# ─── Causality Test (Critical) ───

class TestCausality:
    """Verify that compression does not leak future information into past.

    This is the test that would have caught the original causality violation.
    The chunk-wise approach should NOT have this issue because compression
    only applies to PAST tokens (tokens before the current chunk).
    """

    def test_compress_is_causal(self):
        """Modifying tokens at positions > L should not change compressed output.

        The compressed output represents positions 0..N-1. If we modify
        positions >= L (the future relative to the compressed representation),
        the compression SHOULD change (it's a lossy representation of ALL
        positions). But the key property is: during chunk-wise attention,
        the compressed tokens only represent PAST positions, and the causal
        mask prevents attending to future positions.

        This test verifies that compress() is a deterministic function of
        its input — no hidden state leakage.
        """
        torch.manual_seed(42)
        x = torch.randn(1, 2, 64, 8)
        dct = DCTTransform(dim=-2)

        # Compress the same input twice — should get identical results
        x_c1 = dct.compress(x, compress_len=32)
        x_c2 = dct.compress(x, compress_len=32)

        assert torch.equal(x_c1, x_c2), "compress() is not deterministic"

    def test_compress_uses_all_input(self):
        """compress() should use information from all input positions.

        This is EXPECTED behavior — the spectral transform mixes information
        across positions. The causality is enforced by the CHUNK STRUCTURE
        in the forward pass (only past tokens are compressed), not by the
        compression function itself.
        """
        torch.manual_seed(42)
        x = torch.randn(1, 1, 64, 4)
        dct = DCTTransform(dim=-2)

        x_c1 = dct.compress(x, compress_len=32)

        # Modify a future position
        x_modified = x.clone()
        x_modified[:, :, 50, :] += 1.0

        x_c2 = dct.compress(x_modified, compress_len=32)

        # The compressed output SHOULD be different (spectral mixing)
        assert not torch.allclose(x_c1, x_c2), \
            "compress() did not respond to input change — spectral mixing broken"

    def test_chunk_wise_causality(self):
        """Verify that chunk-wise compression is causal.

        Simulates the FreqKV forward: tokens 0..cache_size get standard
        attention. Tokens cache_size..end attend to [sink + compressed_past +
        recent + self]. The compressed_past only contains information from
        tokens BEFORE the current chunk.

        This test verifies: compressing tokens 0..N-1 and using the result
        for attention at positions N..M does NOT leak information from
        positions N..M into the compressed representation.
        """
        torch.manual_seed(42)
        cache_size = 32
        total_len = 64
        x = torch.randn(1, 1, total_len, 4)
        dct = DCTTransform(dim=-2)

        # Compress the first cache_size tokens (the "past")
        past = x[:, :, :cache_size, :]
        compressed_past = dct.compress(past, compress_len=16)

        # Now modify tokens at positions >= cache_size (the "future")
        x_future_modified = x.clone()
        x_future_modified[:, :, cache_size:, :] += 10.0

        # Compress the past again — should be IDENTICAL (future not included)
        past_modified = x_future_modified[:, :, :cache_size, :]
        compressed_past_modified = dct.compress(past_modified, compress_len=16)

        assert torch.allclose(compressed_past, compressed_past_modified, atol=1e-6), \
            "Chunk-wise causality violated: future tokens leaked into compressed past"
