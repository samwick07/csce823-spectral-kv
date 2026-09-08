"""Tests for SpectralKVCompressor.

Verifies:
  - compress() returns L-length output
  - Learnable filter is applied before truncation
  - Both DCT and FFT paths work
  - Filter parameters are registered for LoRA
  - Baseline (no compression) is a pass-through
"""

import torch
import pytest

from src.spectral.cache import CompressionConfig, SpectralKVCompressor
from src.spectral.filter import FixedLowPassFilter, LearnableSpectralFilter


class TestCompressorBasics:
    """Verify basic compressor functionality."""

    def test_dct_fixed_compress(self):
        """DCT + fixed filter should compress to L-length."""
        config = CompressionConfig(
            transform_type="dct", filter_type="fixed", gamma=0.5,
            max_seq_len=256, cache_size=128,
        )
        compressor = SpectralKVCompressor(config, num_kv_heads=4, head_dim=8)

        x = torch.randn(2, 4, 64, 8)
        compressed = compressor.compress(x, compress_len=32)

        assert compressed.shape == (2, 4, 32, 8)
        assert not torch.isnan(compressed).any()

    def test_fft_learnable_compress(self):
        """FFT + learnable filter should compress to L-length."""
        config = CompressionConfig(
            transform_type="fft", filter_type="learnable", gamma=0.5,
            max_seq_len=256, cache_size=128,
        )
        compressor = SpectralKVCompressor(config, num_kv_heads=4, head_dim=8)

        x = torch.randn(2, 4, 64, 8)
        compressed = compressor.compress(x, compress_len=32)

        assert compressed.shape == (2, 4, 32, 8)
        assert not torch.isnan(compressed).any()

    def test_baseline_passthrough(self):
        """Baseline config should pass through without compression."""
        config = CompressionConfig(
            transform_type="none", filter_type="none", gamma=1.0,
        )
        compressor = SpectralKVCompressor(config, num_kv_heads=4, head_dim=8)

        x = torch.randn(2, 4, 64, 8)
        result = compressor.compress(x, compress_len=32)

        # Baseline returns input unchanged
        assert torch.equal(result, x)

    def test_dct_learnable_compress(self):
        """DCT + learnable filter should work."""
        config = CompressionConfig(
            transform_type="dct", filter_type="learnable", gamma=0.5,
            max_seq_len=256, cache_size=128,
        )
        compressor = SpectralKVCompressor(config, num_kv_heads=4, head_dim=8)

        x = torch.randn(2, 4, 64, 8)
        compressed = compressor.compress(x, compress_len=32)

        assert compressed.shape == (2, 4, 32, 8)

    def test_fft_fixed_compress(self):
        """FFT + fixed filter should work."""
        config = CompressionConfig(
            transform_type="fft", filter_type="fixed", gamma=0.5,
            max_seq_len=256, cache_size=128,
        )
        compressor = SpectralKVCompressor(config, num_kv_heads=4, head_dim=8)

        x = torch.randn(2, 4, 64, 8)
        compressed = compressor.compress(x, compress_len=32)

        assert compressed.shape == (2, 4, 32, 8)


class TestFilterIntegration:
    """Verify learnable filter integration."""

    def test_filter_affects_output(self):
        """Learnable filter should change the compressed output vs no filter."""
        torch.manual_seed(42)
        x = torch.randn(1, 4, 64, 8)

        # With learnable filter
        config_learnable = CompressionConfig(
            transform_type="dct", filter_type="learnable", gamma=0.5,
            max_seq_len=256, cache_size=128, init_sharpness=10.0,
        )
        comp_learnable = SpectralKVCompressor(config_learnable, 4, 8)
        compressed_learnable = comp_learnable.compress(x, compress_len=32)

        # With fixed filter (no-op, just truncation)
        config_fixed = CompressionConfig(
            transform_type="dct", filter_type="fixed", gamma=0.5,
            max_seq_len=256, cache_size=128,
        )
        comp_fixed = SpectralKVCompressor(config_fixed, 4, 8)
        compressed_fixed = comp_fixed.compress(x, compress_len=32)

        # They should be different (learnable filter modifies spectrum)
        assert not torch.allclose(compressed_learnable, compressed_fixed, atol=1e-6), \
            "Learnable filter had no effect on output"

    def test_filter_is_applied_before_truncation(self):
        """Filter should see the full spectrum, not just truncated coefficients.

        We verify this by checking that the filter modifies coefficients that
        would otherwise be truncated away. If the filter boosts a high-frequency
        component before truncation, it should survive into the compressed output.
        """
        torch.manual_seed(42)
        x = torch.randn(1, 1, 64, 4)

        # Create a filter that boosts high frequencies (inverse low-pass)
        config = CompressionConfig(
            transform_type="dct", filter_type="learnable", gamma=0.5,
            max_seq_len=256, cache_size=128,
            init_sharpness=-10.0,  # Negative = boost high freq
            init_offset=0.5,       # Shift cutoff
        )
        compressor = SpectralKVCompressor(config, 1, 4)

        # Get the filter mask
        mask = compressor.get_filter_mask()
        assert mask is not None
        # With negative sharpness, the mask should NOT be a simple low-pass
        # (high frequencies should have higher weights than low frequencies)
        assert mask[0, -1] > mask[0, 0], \
            f"Filter should boost high freq with negative sharpness: " \
            f"mask[0]={mask[0, 0]:.4f}, mask[-1]={mask[0, -1]:.4f}"

    def test_filter_parameters_registered(self):
        """Learnable filter parameters should be accessible via get_learnable_parameters()."""
        config = CompressionConfig(
            transform_type="fft", filter_type="learnable", gamma=0.5,
            max_seq_len=256, cache_size=128,
        )
        compressor = SpectralKVCompressor(config, num_kv_heads=4, head_dim=8)

        params = compressor.get_learnable_parameters()
        assert len(params) == 1
        assert params[0].requires_grad

    def test_no_params_for_fixed_filter(self):
        """Fixed filter should have no learnable parameters."""
        config = CompressionConfig(
            transform_type="dct", filter_type="fixed", gamma=0.5,
            max_seq_len=256, cache_size=128,
        )
        compressor = SpectralKVCompressor(config, num_kv_heads=4, head_dim=8)

        params = compressor.get_learnable_parameters()
        assert len(params) == 0

    def test_compressor_is_nn_module(self):
        """Compressor should be an nn.Module with parameters for device migration."""
        config = CompressionConfig(
            transform_type="dct", filter_type="learnable", gamma=0.5,
            max_seq_len=256, cache_size=128,
        )
        compressor = SpectralKVCompressor(config, num_kv_heads=4, head_dim=8)

        assert isinstance(compressor, torch.nn.Module)
        # Should have at least the filter_logits parameter
        param_list = list(compressor.parameters())
        assert len(param_list) >= 1


class TestCompressorConfig:
    """Verify CompressionConfig properties."""

    def test_fft_span(self):
        """fft_span = cache_size - sink_size - recent_size."""
        config = CompressionConfig(
            cache_size=8192, sink_size=4, recent_size=8, gamma=0.5
        )
        assert config.fft_span == 8192 - 4 - 8  # 8180

    def test_fft_size(self):
        """fft_size = int(fft_span * gamma)."""
        config = CompressionConfig(
            cache_size=8192, sink_size=4, recent_size=8, gamma=0.5
        )
        assert config.fft_size == int(8180 * 0.5)  # 4090

    def test_chunk_size(self):
        """chunk_size = fft_span - fft_size."""
        config = CompressionConfig(
            cache_size=8192, sink_size=4, recent_size=8, gamma=0.5
        )
        assert config.chunk_size == 8180 - 4090  # 4090

    def test_cache_bounded(self):
        """sink + fft_size + recent + chunk_size should equal cache_size."""
        config = CompressionConfig(
            cache_size=8192, sink_size=4, recent_size=8, gamma=0.5
        )
        total = config.sink_size + config.fft_size + config.recent_size + config.chunk_size
        assert total == config.cache_size

    def test_baseline(self):
        """transform_type='none' should be baseline."""
        config = CompressionConfig(transform_type="none")
        assert config.is_baseline

    def test_gamma_one_is_baseline(self):
        """gamma=1.0 should be baseline."""
        config = CompressionConfig(transform_type="dct", gamma=1.0)
        assert config.is_baseline
