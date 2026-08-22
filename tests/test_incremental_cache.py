"""Tests for incremental KV caching (K=1), SpectralDynamicCache, and speedup fixes.

Tests:
  1. SpectralKVCache.append() -- incremental K=1 update preserves shapes
  2. append() maintains compression ratio (zero deviation with K=1)
  3. append() on baseline (no compression) -- simple concatenation
  4. SpectralDynamicCache -- get_seq_length after compress/append
  5. SpectralDynamicCache -- update() routes to compress or append correctly
  6. Truncate-before-filter equivalence (Fix 3)
  7. DCT precomputed constants (Fix 4) -- same results, memoized
  8. _compute_attention -- SDPA fallback works
  9. append() followed by reconstruct() is lossy but shape-correct

Run:
    pytest tests/test_incremental_cache.py -v
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def kv_tensors():
    """KV tensors: [batch=1, num_kv_heads=8, seq_len=128, head_dim=128]."""
    torch.manual_seed(42)
    k = torch.randn(1, 8, 128, 128, dtype=torch.float64)
    v = torch.randn(1, 8, 128, 128, dtype=torch.float64)
    return k, v


@pytest.fixture
def single_token_kv():
    """Single token KV tensors: [batch=1, num_kv_heads=8, seq_len=1, head_dim=128]."""
    torch.manual_seed(99)
    k = torch.randn(1, 8, 1, 128, dtype=torch.float64)
    v = torch.randn(1, 8, 1, 128, dtype=torch.float64)
    return k, v


# ---------------------------------------------------------------------------
# 1. SpectralKVCache.append() -- incremental K=1 update
# ---------------------------------------------------------------------------

class TestAppendIncremental:

    def test_append_preserves_shape(self, kv_tensors, single_token_kv):
        """append() should return K/V with seq_len = original + 1."""
        from src.spectral.cache import CompressionConfig, SpectralKVCache

        k, v = kv_tensors
        new_k, new_v = single_token_kv

        config = CompressionConfig(
            transform_type="fft", filter_type="fixed", gamma=0.22,
            max_seq_len=256,
        )
        cache = SpectralKVCache(config, num_kv_heads=8, head_dim=128)

        # First compress the full sequence (prefill)
        cache.compress(k, v)

        # Then append one token (decode step)
        full_k, full_v = cache.append(new_k, new_v)

        assert full_k.shape == (1, 8, 129, 128)
        assert full_v.shape == (1, 8, 129, 128)

    def test_append_maintains_compression_ratio(self, kv_tensors, single_token_kv):
        """After append(), compression ratio should still be ~gamma (zero deviation with K=1)."""
        from src.spectral.cache import CompressionConfig, SpectralKVCache

        k, v = kv_tensors
        new_k, new_v = single_token_kv

        config = CompressionConfig(
            transform_type="fft", filter_type="fixed", gamma=0.22,
            max_seq_len=256,
        )
        cache = SpectralKVCache(config, num_kv_heads=8, head_dim=128)

        cache.compress(k, v)
        ratio_before = cache.get_compression_ratio()

        cache.append(new_k, new_v)
        ratio_after = cache.get_compression_ratio()

        # With K=1, the ratio should be approximately the same
        # (it changes slightly because gamma is applied to N+1 instead of N)
        assert abs(ratio_after - ratio_before) < 0.02, \
            f"Ratio changed too much: {ratio_before} -> {ratio_after}"

    def test_append_multiple_tokens(self, kv_tensors):
        """Multiple sequential append() calls should grow seq_len by 1 each time."""
        from src.spectral.cache import CompressionConfig, SpectralKVCache

        k, v = kv_tensors
        config = CompressionConfig(
            transform_type="fft", filter_type="fixed", gamma=0.22,
            max_seq_len=256,
        )
        cache = SpectralKVCache(config, num_kv_heads=8, head_dim=128)

        cache.compress(k, v)
        original_len = k.shape[-2]

        for i in range(5):
            new_k = torch.randn(1, 8, 1, 128, dtype=torch.float64)
            new_v = torch.randn(1, 8, 1, 128, dtype=torch.float64)
            full_k, full_v = cache.append(new_k, new_v)
            expected_len = original_len + i + 1
            assert full_k.shape[-2] == expected_len, \
                f"Step {i}: expected seq_len={expected_len}, got {full_k.shape[-2]}"

    def test_append_baseline_concatenation(self, kv_tensors, single_token_kv):
        """Baseline (no compression) append() should be exact concatenation."""
        from src.spectral.cache import CompressionConfig, SpectralKVCache

        k, v = kv_tensors
        new_k, new_v = single_token_kv

        config = CompressionConfig(
            transform_type="none", filter_type="none", gamma=1.0,
        )
        cache = SpectralKVCache(config, num_kv_heads=8, head_dim=128)

        cache.compress(k, v)
        full_k, full_v = cache.append(new_k, new_v)

        # Baseline should be exact (no lossy reconstruction)
        expected_k = torch.cat([k, new_k], dim=-2)
        expected_v = torch.cat([v, new_v], dim=-2)
        assert torch.equal(full_k, expected_k)
        assert torch.equal(full_v, expected_v)

    def test_append_dct_transform(self, kv_tensors, single_token_kv):
        """append() should work with DCT transform."""
        from src.spectral.cache import CompressionConfig, SpectralKVCache

        k, v = kv_tensors
        new_k, new_v = single_token_kv

        config = CompressionConfig(
            transform_type="dct", filter_type="fixed", gamma=0.50,
            max_seq_len=256,
        )
        cache = SpectralKVCache(config, num_kv_heads=8, head_dim=128)

        cache.compress(k, v)
        full_k, full_v = cache.append(new_k, new_v)

        assert full_k.shape == (1, 8, 129, 128)
        assert full_v.shape == (1, 8, 129, 128)
        assert not torch.isnan(full_k).any()
        assert not torch.isnan(full_v).any()

    def test_append_first_token(self, single_token_kv):
        """append() on empty cache should work like compress()."""
        from src.spectral.cache import CompressionConfig, SpectralKVCache

        new_k, new_v = single_token_kv
        config = CompressionConfig(
            transform_type="fft", filter_type="fixed", gamma=0.22,
            max_seq_len=256,
        )
        cache = SpectralKVCache(config, num_kv_heads=8, head_dim=128)

        # No prior compress() call -- append should handle it
        full_k, full_v = cache.append(new_k, new_v)

        assert full_k.shape == (1, 8, 1, 128)
        assert full_v.shape == (1, 8, 1, 128)
        assert cache._cached_seq_len == 1

    def test_append_stats_tracked(self, kv_tensors, single_token_kv):
        """append() should increment total_append_calls stat."""
        from src.spectral.cache import CompressionConfig, SpectralKVCache

        k, v = kv_tensors
        new_k, new_v = single_token_kv

        config = CompressionConfig(
            transform_type="fft", filter_type="fixed", gamma=0.22,
            max_seq_len=256,
        )
        cache = SpectralKVCache(config, num_kv_heads=8, head_dim=128)

        cache.compress(k, v)
        assert cache.compression_stats["total_append_calls"] == 0

        cache.append(new_k, new_v)
        assert cache.compression_stats["total_append_calls"] == 1

        cache.append(new_k, new_v)
        assert cache.compression_stats["total_append_calls"] == 2


# ---------------------------------------------------------------------------
# 2. SpectralDynamicCache -- HF Cache protocol wrapper
# ---------------------------------------------------------------------------

class TestSpectralDynamicCache:

    def test_seq_length_after_compress(self, kv_tensors):
        """get_seq_length() should return cached seq_len after compress."""
        from src.spectral.cache import CompressionConfig, SpectralKVCache, SpectralDynamicCache

        k, v = kv_tensors
        config = CompressionConfig(
            transform_type="fft", filter_type="fixed", gamma=0.22,
            max_seq_len=256,
        )
        spectral_cache = SpectralKVCache(config, num_kv_heads=8, head_dim=128)
        dynamic_cache = SpectralDynamicCache([spectral_cache])

        assert dynamic_cache.get_seq_length() == 0

        spectral_cache.compress(k, v)
        assert dynamic_cache.get_seq_length() == 128

    def test_update_prefill_path(self, kv_tensors):
        """update() with q_len > 1 should route to compress()."""
        from src.spectral.cache import CompressionConfig, SpectralKVCache, SpectralDynamicCache

        k, v = kv_tensors
        config = CompressionConfig(
            transform_type="fft", filter_type="fixed", gamma=0.22,
            max_seq_len=256,
        )
        spectral_cache = SpectralKVCache(config, num_kv_heads=8, head_dim=128)
        dynamic_cache = SpectralDynamicCache([spectral_cache])

        # q_len = 128 > 1, so this is a prefill
        ret_k, ret_v = dynamic_cache.update(k, v, layer_idx=0)

        assert ret_k.shape == k.shape
        assert ret_v.shape == v.shape
        assert dynamic_cache.get_seq_length() == 128

    def test_update_decode_path(self, kv_tensors, single_token_kv):
        """update() with q_len == 1 after prefill should route to append()."""
        from src.spectral.cache import CompressionConfig, SpectralKVCache, SpectralDynamicCache

        k, v = kv_tensors
        new_k, new_v = single_token_kv

        config = CompressionConfig(
            transform_type="fft", filter_type="fixed", gamma=0.22,
            max_seq_len=256,
        )
        spectral_cache = SpectralKVCache(config, num_kv_heads=8, head_dim=128)
        dynamic_cache = SpectralDynamicCache([spectral_cache])

        # Prefill
        dynamic_cache.update(k, v, layer_idx=0)
        assert dynamic_cache.get_seq_length() == 128

        # Decode: q_len=1, cache has data -> should append
        ret_k, ret_v = dynamic_cache.update(new_k, new_v, layer_idx=0)

        assert ret_k.shape == (1, 8, 129, 128)
        assert ret_v.shape == (1, 8, 129, 128)
        assert dynamic_cache.get_seq_length() == 129

    def test_reset_clears_all(self, kv_tensors):
        """reset() should clear all per-layer caches."""
        from src.spectral.cache import CompressionConfig, SpectralKVCache, SpectralDynamicCache

        k, v = kv_tensors
        config = CompressionConfig(
            transform_type="fft", filter_type="fixed", gamma=0.22,
            max_seq_len=256,
        )
        spectral_cache = SpectralKVCache(config, num_kv_heads=8, head_dim=128)
        dynamic_cache = SpectralDynamicCache([spectral_cache])

        dynamic_cache.update(k, v, layer_idx=0)
        assert dynamic_cache.get_seq_length() == 128

        dynamic_cache.reset()
        assert dynamic_cache.get_seq_length() == 0

    def test_len_protocol(self, kv_tensors):
        """__len__ should return seq length (HF compatibility)."""
        from src.spectral.cache import CompressionConfig, SpectralKVCache, SpectralDynamicCache

        k, v = kv_tensors
        config = CompressionConfig(
            transform_type="fft", filter_type="fixed", gamma=0.22,
            max_seq_len=256,
        )
        spectral_cache = SpectralKVCache(config, num_kv_heads=8, head_dim=128)
        dynamic_cache = SpectralDynamicCache([spectral_cache])

        spectral_cache.compress(k, v)
        assert len(dynamic_cache) == 128


# ---------------------------------------------------------------------------
# 3. Truncate-before-filter equivalence (Fix 3)
# ---------------------------------------------------------------------------

class TestTruncateBeforeFilter:

    def test_truncate_then_filter_equals_filter_then_truncate(self, kv_tensors):
        """Fix 3: truncating before filtering should give identical results.

        The filter is element-wise (sigmoid mask), so truncating first
        and then filtering the remaining coefficients gives the same
        result as filtering all and then truncating.
        """
        from src.spectral.transform import FFTTransform
        from src.spectral.filter import LearnableSpectralFilter

        k, _ = kv_tensors
        fft = FFTTransform(dim=-2)
        spectral = fft.forward(k)

        filt = LearnableSpectralFilter(
            num_heads=8,
            max_spectral_len=fft.spectral_len(128),
            init_sharpness=10.0,
        )
        gamma = 0.22

        # Path A: filter then truncate (old order)
        filtered_full = filt(spectral, gamma)
        truncated_A = fft.truncate(filtered_full, gamma)

        # Path B: truncate then filter (new order, Fix 3)
        truncated_spectral = fft.truncate(spectral, gamma)
        filtered_B = filt(truncated_spectral, gamma)

        assert torch.allclose(truncated_A, filtered_B, atol=1e-12), \
            "Truncate-before-filter should be equivalent to filter-then-truncate"

    def test_truncate_before_filter_saves_computation(self, kv_tensors):
        """Fix 3: truncated tensor should have fewer elements for filter to process."""
        from src.spectral.transform import FFTTransform
        from src.spectral.filter import LearnableSpectralFilter

        k, _ = kv_tensors
        fft = FFTTransform(dim=-2)
        spectral = fft.forward(k)
        gamma = 0.01  # Extreme compression

        # Full spectral length
        full_len = spectral.shape[-2]
        # Truncated length
        truncated = fft.truncate(spectral, gamma)
        truncated_len = truncated.shape[-2]

        assert truncated_len < full_len, \
            f"Truncated ({truncated_len}) should be < full ({full_len})"
        assert truncated_len == max(1, int(full_len * gamma))


# ---------------------------------------------------------------------------
# 4. DCT precomputed constants (Fix 4)
# ---------------------------------------------------------------------------

class TestDCTPrecomputedConstants:

    def test_dct_forward_same_results(self, kv_tensors):
        """Fix 4: precomputed constants should give same results as recomputed."""
        from src.spectral.transform import DCTTransform

        k, _ = kv_tensors
        dct = DCTTransform(dim=-2)

        # First call (populates cache)
        spectral_1 = dct.forward(k)

        # Second call (uses cache)
        spectral_2 = dct.forward(k)

        assert torch.allclose(spectral_1, spectral_2, atol=0), \
            "Cached DCT constants should give identical results"

    def test_dct_constants_are_memoized(self):
        """Fix 4: forward constants should be cached after first call."""
        from src.spectral.transform import DCTTransform

        dct = DCTTransform(dim=-2)
        x = torch.randn(1, 4, 64, 8, dtype=torch.float64)

        # Before any call, cache should be empty
        assert len(dct._fwd_cache) == 0

        dct.forward(x)

        # After first call, cache should have an entry for N=64
        assert len(dct._fwd_cache) == 1
        assert (64, str(x.device)) in dct._fwd_cache

    def test_dct_inverse_uses_cache(self, kv_tensors):
        """Fix 4: inverse DCT should also use cached constants."""
        from src.spectral.transform import DCTTransform

        k, _ = kv_tensors
        dct = DCTTransform(dim=-2)

        spectral = dct.forward(k)

        # Before inverse, inv cache should be empty
        assert len(dct._inv_cache) == 0

        reconstructed = dct.inverse(spectral, target_len=k.shape[-2])

        # After inverse, cache should have an entry
        assert len(dct._inv_cache) == 1

        # Verify round-trip still works
        error = (k - reconstructed).abs().max().item()
        assert error < 1e-5, f"DCT round-trip error too high with cached constants: {error}"


# ---------------------------------------------------------------------------
# 5. _compute_attention -- SDPA/FA2 fallback
# ---------------------------------------------------------------------------

class TestComputeAttention:

    def test_sdpa_fallback_shape(self):
        """_compute_attention should return correct shape via SDPA."""
        from src.spectral.attention import _compute_attention

        bsz, num_heads, q_len, head_dim = 1, 8, 64, 128
        kv_len = 64

        q = torch.randn(bsz, num_heads, q_len, head_dim)
        k = torch.randn(bsz, num_heads, kv_len, head_dim)
        v = torch.randn(bsz, num_heads, kv_len, head_dim)

        output = _compute_attention(q, k, v, attention_mask=None, use_flash=True)

        assert output.shape == (bsz, num_heads, q_len, head_dim)
        assert not torch.isnan(output).any()

    def test_manual_fallback_shape(self):
        """_compute_attention with use_flash=False should use manual attention."""
        from src.spectral.attention import _compute_attention

        bsz, num_heads, q_len, head_dim = 1, 8, 64, 128
        kv_len = 64

        q = torch.randn(bsz, num_heads, q_len, head_dim)
        k = torch.randn(bsz, num_heads, kv_len, head_dim)
        v = torch.randn(bsz, num_heads, kv_len, head_dim)

        output = _compute_attention(q, k, v, attention_mask=None, use_flash=False)

        assert output.shape == (bsz, num_heads, q_len, head_dim)
        assert not torch.isnan(output).any()

    def test_decode_step_shape(self):
        """_compute_attention should handle q_len=1 (decode step)."""
        from src.spectral.attention import _compute_attention

        bsz, num_heads, head_dim = 1, 8, 128
        q_len = 1  # decode step
        kv_len = 128  # full sequence

        q = torch.randn(bsz, num_heads, q_len, head_dim)
        k = torch.randn(bsz, num_heads, kv_len, head_dim)
        v = torch.randn(bsz, num_heads, kv_len, head_dim)

        output = _compute_attention(q, k, v, attention_mask=None, use_flash=True)

        assert output.shape == (bsz, num_heads, q_len, head_dim)
        assert not torch.isnan(output).any()
