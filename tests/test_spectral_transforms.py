"""Unit tests for spectral transforms, filters, and KV cache.

Tests:
  1. DCT round-trip reconstruction error (forward -> inverse ≈ identity)
  2. FFT round-trip reconstruction error
  3. Truncation preserves expected fraction (gamma·N coefficients)
  4. Learnable filter gradient flow (backward pass updates filter_logits)
  5. FixedLowPassFilter output matches expected low-pass behavior
  6. SpectralKVCache compress -> reconstruct shape correctness
  7. GQA repeat_kv on reconstructed tensors
  8. ART ANOVA correctness on synthetic factorial data

Run:
    pytest tests/test_spectral_transforms.py -v
    pytest tests/test_spectral_transforms.py -v -k "dct"
"""

from __future__ import annotations

import math
import numpy as np
import pytest
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_tensor():
    """Small KV tensor: [batch=2, num_heads=4, seq_len=64, head_dim=8]."""
    torch.manual_seed(42)
    return torch.randn(2, 4, 64, 8, dtype=torch.float64)


@pytest.fixture
def medium_tensor():
    """Medium KV tensor: [batch=1, num_heads=8, seq_len=256, head_dim=128]."""
    torch.manual_seed(123)
    return torch.randn(1, 8, 256, 128, dtype=torch.float64)


# ---------------------------------------------------------------------------
# 1. DCT round-trip reconstruction
# ---------------------------------------------------------------------------

class TestDCTTransform:

    def test_dct_round_trip_identity(self, small_tensor):
        """DCT forward -> inverse should approximately reconstruct the input."""
        from src.spectral.transform import DCTTransform

        dct = DCTTransform(dim=-2)
        spectral = dct.forward(small_tensor)
        reconstructed = dct.inverse(spectral, target_len=small_tensor.shape[-2])

        error = (small_tensor - reconstructed).abs().max().item()
        assert error < 1e-10, f"DCT round-trip error too high: {error}"

    def test_dct_round_trip_medium(self, medium_tensor):
        """DCT round-trip with larger head_dim."""
        from src.spectral.transform import DCTTransform

        dct = DCTTransform(dim=-2)
        spectral = dct.forward(medium_tensor)
        reconstructed = dct.inverse(spectral, target_len=medium_tensor.shape[-2])

        error = (medium_tensor - reconstructed).abs().max().item()
        assert error < 1e-8, f"DCT round-trip error too high: {error}"

    def test_dct_spectral_len(self):
        """DCT spectral_len(N) should equal N."""
        from src.spectral.transform import DCTTransform

        dct = DCTTransform()
        assert dct.spectral_len(64) == 64
        assert dct.spectral_len(128) == 128
        assert dct.spectral_len(1) == 1

    def test_dct_output_shape(self, small_tensor):
        """DCT output should have same shape as input."""
        from src.spectral.transform import DCTTransform

        dct = DCTTransform(dim=-2)
        spectral = dct.forward(small_tensor)
        assert spectral.shape == small_tensor.shape

    def test_dct_output_is_real(self, small_tensor):
        """DCT output should be real-valued (not complex)."""
        from src.spectral.transform import DCTTransform

        dct = DCTTransform(dim=-2)
        spectral = dct.forward(small_tensor)
        assert not spectral.is_complex(), "DCT output should be real"

    def test_dct_pad_to_len(self, small_tensor):
        """DCT pad_to_len should zero-pad to target length."""
        from src.spectral.transform import DCTTransform

        dct = DCTTransform(dim=-2)
        spectral = dct.forward(small_tensor)
        truncated = dct.truncate(spectral, gamma=0.25)  # 16 coefficients
        padded = dct.pad_to_len(truncated, target_spectral_len=64)

        assert padded.shape[-2] == 64
        # First 16 should match the truncated values
        assert torch.allclose(padded[..., :16, :], truncated, atol=1e-12)
        # Rest should be zero
        assert torch.allclose(padded[..., 16:, :], torch.zeros_like(padded[..., 16:, :]))


# ---------------------------------------------------------------------------
# 2. FFT round-trip reconstruction
# ---------------------------------------------------------------------------

class TestFFTTransform:

    def test_fft_round_trip_identity(self, small_tensor):
        """FFT forward -> inverse should approximately reconstruct the input."""
        from src.spectral.transform import FFTTransform

        fft = FFTTransform(dim=-2)
        spectral = fft.forward(small_tensor)
        reconstructed = fft.inverse(spectral, target_len=small_tensor.shape[-2])

        error = (small_tensor - reconstructed).abs().max().item()
        assert error < 1e-10, f"FFT round-trip error too high: {error}"

    def test_fft_round_trip_medium(self, medium_tensor):
        """FFT round-trip with larger head_dim."""
        from src.spectral.transform import FFTTransform

        fft = FFTTransform(dim=-2)
        spectral = fft.forward(medium_tensor)
        reconstructed = fft.inverse(spectral, target_len=medium_tensor.shape[-2])

        error = (medium_tensor - reconstructed).abs().max().item()
        assert error < 1e-8, f"FFT round-trip error too high: {error}"

    def test_fft_spectral_len(self):
        """FFT spectral_len(N) should equal N//2+1."""
        from src.spectral.transform import FFTTransform

        fft = FFTTransform()
        assert fft.spectral_len(64) == 33  # 64//2 + 1
        assert fft.spectral_len(128) == 65
        assert fft.spectral_len(1) == 1

    def test_fft_output_shape(self, small_tensor):
        """FFT output should have reduced sequence dimension (N//2+1)."""
        from src.spectral.transform import FFTTransform

        fft = FFTTransform(dim=-2)
        spectral = fft.forward(small_tensor)

        expected_len = small_tensor.shape[-2] // 2 + 1
        assert spectral.shape[-2] == expected_len
        assert spectral.shape[:-2] == small_tensor.shape[:-2]
        assert spectral.shape[-1] == small_tensor.shape[-1]

    def test_fft_output_is_complex(self, small_tensor):
        """FFT (rfft) output should be complex-valued."""
        from src.spectral.transform import FFTTransform

        fft = FFTTransform(dim=-2)
        spectral = fft.forward(small_tensor)
        assert spectral.is_complex(), "rFFT output should be complex"

    def test_fft_pad_to_len(self, small_tensor):
        """FFT pad_to_len should zero-pad complex coefficients."""
        from src.spectral.transform import FFTTransform

        fft = FFTTransform(dim=-2)
        spectral = fft.forward(small_tensor)
        truncated = fft.truncate(spectral, gamma=0.25)  # ~8 coefficients
        target = spectral.shape[-2]  # Full length = 33
        padded = fft.pad_to_len(truncated, target_spectral_len=target)

        assert padded.shape[-2] == target
        assert padded.is_complex()


# ---------------------------------------------------------------------------
# 3. Truncation tests
# ---------------------------------------------------------------------------

class TestTruncation:

    def test_dct_truncation_fraction(self, small_tensor):
        """DCT truncation should retain gamma*N coefficients."""
        from src.spectral.transform import DCTTransform

        dct = DCTTransform(dim=-2)
        spectral = dct.forward(small_tensor)
        N = small_tensor.shape[-2]

        for gamma in [0.50, 0.22, 0.01]:
            truncated = dct.truncate(spectral, gamma=gamma)
            expected = max(1, int(N * gamma))
            assert truncated.shape[-2] == expected, \
                f"DCT truncate gamma={gamma}: expected {expected}, got {truncated.shape[-2]}"

    def test_fft_truncation_fraction(self, small_tensor):
        """FFT truncation should retain gamma*(N//2+1) coefficients."""
        from src.spectral.transform import FFTTransform

        fft = FFTTransform(dim=-2)
        spectral = fft.forward(small_tensor)
        N_spec = spectral.shape[-2]  # N//2+1

        for gamma in [0.50, 0.22, 0.01]:
            truncated = fft.truncate(spectral, gamma=gamma)
            expected = max(1, int(N_spec * gamma))
            assert truncated.shape[-2] == expected, \
                f"FFT truncate gamma={gamma}: expected {expected}, got {truncated.shape[-2]}"

    def test_truncation_preserves_low_frequencies(self, small_tensor):
        """Truncation should keep the lowest-frequency coefficients."""
        from src.spectral.transform import DCTTransform

        dct = DCTTransform(dim=-2)
        spectral = dct.forward(small_tensor)
        truncated = dct.truncate(spectral, gamma=0.25)

        # The first 25% of coefficients should match
        keep = max(1, int(small_tensor.shape[-2] * 0.25))
        assert torch.allclose(truncated, spectral[..., :keep, :], atol=1e-12)

    def test_truncation_gamma_one(self, small_tensor):
        """gamma=1.0 should retain all coefficients."""
        from src.spectral.transform import DCTTransform, FFTTransform

        for transform_cls in [DCTTransform, FFTTransform]:
            t = transform_cls(dim=-2)
            spectral = t.forward(small_tensor)
            truncated = t.truncate(spectral, gamma=1.0)
            assert truncated.shape[-2] == spectral.shape[-2], \
                f"{transform_cls.__name__} gamma=1.0 should not truncate"


# ---------------------------------------------------------------------------
# 4. Learnable filter gradient flow
# ---------------------------------------------------------------------------

class TestLearnableFilter:

    def test_filter_has_parameters(self):
        """LearnableSpectralFilter should have trainable filter_logits."""
        from src.spectral.filter import LearnableSpectralFilter

        filt = LearnableSpectralFilter(num_heads=4, max_spectral_len=64)
        assert hasattr(filt, "filter_logits")
        assert filt.filter_logits.requires_grad
        assert filt.filter_logits.shape == (4, 64)

    def test_filter_forward_shape(self, small_tensor):
        """Filter forward should preserve shape."""
        from src.spectral.transform import FFTTransform
        from src.spectral.filter import LearnableSpectralFilter

        fft = FFTTransform(dim=-2)
        spectral = fft.forward(small_tensor)  # complex

        filt = LearnableSpectralFilter(
            num_heads=small_tensor.shape[-3],
            max_spectral_len=fft.spectral_len(small_tensor.shape[-2]),
        )
        filtered = filt(spectral, gamma=0.22)

        assert filtered.shape == spectral.shape

    def test_filter_gradient_flow(self, small_tensor):
        """Backward pass should update filter_logits."""
        from src.spectral.transform import FFTTransform
        from src.spectral.filter import LearnableSpectralFilter

        fft = FFTTransform(dim=-2)
        spectral = fft.forward(small_tensor)

        filt = LearnableSpectralFilter(
            num_heads=small_tensor.shape[-3],
            max_spectral_len=fft.spectral_len(small_tensor.shape[-2]),
        )

        # Forward
        filtered = filt(spectral, gamma=0.22)
        loss = filtered.abs().sum()
        loss.backward()

        assert filt.filter_logits.grad is not None
        assert filt.filter_logits.grad.abs().sum() > 0, "No gradient on filter_logits"

    def test_filter_mask_range(self):
        """Sigmoid mask should be in [0, 1]."""
        from src.spectral.filter import LearnableSpectralFilter

        filt = LearnableSpectralFilter(num_heads=4, max_spectral_len=64)
        mask = filt.get_mask()

        assert mask.shape == (4, 64)
        assert mask.min() >= 0.0
        assert mask.max() <= 1.0

    def test_filter_initialization_lowpass(self):
        """Filter should initialize as approximate low-pass (high at low freq)."""
        from src.spectral.filter import LearnableSpectralFilter

        filt = LearnableSpectralFilter(
            num_heads=1, max_spectral_len=64,
            init_sharpness=10.0, init_offset=0.0,
        )
        mask = filt.get_mask()[0]  # [64]

        # Low-frequency end should be near 1, high-frequency near 0
        low_freq_mean = mask[:16].mean().item()
        high_freq_mean = mask[48:].mean().item()

        assert low_freq_mean > 0.8, f"Low freq mask should be ~1, got {low_freq_mean}"
        assert high_freq_mean < 0.2, f"High freq mask should be ~0, got {high_freq_mean}"


# ---------------------------------------------------------------------------
# 5. FixedLowPassFilter behavior
# ---------------------------------------------------------------------------

class TestFixedLowPassFilter:

    def test_fixed_filter_is_noop(self, small_tensor):
        """FixedLowPassFilter should be a pass-through (no-op)."""
        from src.spectral.filter import FixedLowPassFilter

        filt = FixedLowPassFilter()
        output = filt(small_tensor, gamma=0.22)

        # Should be identical (same object, no modification)
        assert torch.equal(output, small_tensor)

    def test_fixed_filter_no_parameters(self):
        """FixedLowPassFilter should have no trainable parameters."""
        from src.spectral.filter import FixedLowPassFilter

        filt = FixedLowPassFilter()
        params = list(filt.parameters())
        assert len(params) == 0, "FixedLowPassFilter should have no parameters"

    def test_fixed_filter_works_with_complex(self, small_tensor):
        """FixedLowPassFilter should work with complex spectral data."""
        from src.spectral.transform import FFTTransform
        from src.spectral.filter import FixedLowPassFilter

        fft = FFTTransform(dim=-2)
        spectral = fft.forward(small_tensor)
        filt = FixedLowPassFilter()

        output = filt(spectral, gamma=0.22)
        assert torch.equal(output, spectral)


# ---------------------------------------------------------------------------
# 6. SpectralKVCache compress/reconstruct
# ---------------------------------------------------------------------------

class TestSpectralKVCache:

    @pytest.fixture
    def kv_tensors(self):
        """KV tensors: [batch=1, num_kv_heads=8, seq_len=128, head_dim=128]."""
        torch.manual_seed(42)
        k = torch.randn(1, 8, 128, 128, dtype=torch.float64)
        v = torch.randn(1, 8, 128, 128, dtype=torch.float64)
        return k, v

    def test_compress_reconstruct_shape(self, kv_tensors):
        """compress -> reconstruct should return tensors with correct shape."""
        from src.spectral.cache import CompressionConfig, SpectralKVCache

        k, v = kv_tensors
        config = CompressionConfig(
            transform_type="fft", filter_type="learnable", gamma=0.22,
            max_seq_len=128,
        )
        cache = SpectralKVCache(config, num_kv_heads=8, head_dim=128)

        cache.compress(k, v)
        k_recon, v_recon = cache.reconstruct(target_seq_len=128)

        assert k_recon.shape == k.shape
        assert v_recon.shape == v.shape

    def test_compression_ratio(self, kv_tensors):
        """get_compression_ratio should report gamma approximately."""
        from src.spectral.cache import CompressionConfig, SpectralKVCache

        k, v = kv_tensors
        config = CompressionConfig(
            transform_type="fft", filter_type="fixed", gamma=0.22,
            max_seq_len=128,
        )
        cache = SpectralKVCache(config, num_kv_heads=8, head_dim=128)
        cache.compress(k, v)

        ratio = cache.get_compression_ratio()
        # FFT: spectral_len = 65, truncated to int(65*0.22) = 14
        # ratio = 14 / 128 ≈ 0.109
        expected = max(1, int(65 * 0.22)) / 128
        assert abs(ratio - expected) < 0.01, f"Expected ratio ~{expected}, got {ratio}"

    def test_reset_clears_cache(self, kv_tensors):
        """reset() should clear cached state."""
        from src.spectral.cache import CompressionConfig, SpectralKVCache

        k, v = kv_tensors
        config = CompressionConfig(
            transform_type="fft", filter_type="fixed", gamma=0.22,
            max_seq_len=128,
        )
        cache = SpectralKVCache(config, num_kv_heads=8, head_dim=128)

        cache.compress(k, v)
        assert cache._cached_k_spectral is not None

        cache.reset()
        assert cache._cached_k_spectral is None
        assert cache._cached_v_spectral is None
        assert cache._cached_seq_len == 0

    def test_baseline_passthrough(self, kv_tensors):
        """Baseline config should store spatial tensors directly."""
        from src.spectral.cache import CompressionConfig, SpectralKVCache

        k, v = kv_tensors
        config = CompressionConfig(
            transform_type="none", filter_type="none", gamma=1.0,
        )
        cache = SpectralKVCache(config, num_kv_heads=8, head_dim=128)
        assert not cache._is_spectral

        cache.compress(k, v)
        k_recon, v_recon = cache.reconstruct()

        assert torch.equal(k_recon, k)
        assert torch.equal(v_recon, v)

    def test_cache_size_bytes(self, kv_tensors):
        """get_cache_size_bytes should report compressed size."""
        from src.spectral.cache import CompressionConfig, SpectralKVCache

        k, v = kv_tensors
        config = CompressionConfig(
            transform_type="fft", filter_type="fixed", gamma=0.22,
            max_seq_len=128,
        )
        cache = SpectralKVCache(config, num_kv_heads=8, head_dim=128)
        cache.compress(k, v)

        size = cache.get_cache_size_bytes()
        assert size > 0

        # Compare to uncompressed size
        uncompressed_size = k.nelement() * k.element_size() + v.nelement() * v.element_size()
        assert size < uncompressed_size, "Compressed cache should be smaller"

    def test_dct_cache_shape(self, kv_tensors):
        """DCT cache should produce correct shapes."""
        from src.spectral.cache import CompressionConfig, SpectralKVCache

        k, v = kv_tensors
        config = CompressionConfig(
            transform_type="dct", filter_type="fixed", gamma=0.50,
            max_seq_len=128,
        )
        cache = SpectralKVCache(config, num_kv_heads=8, head_dim=128)
        cache.compress(k, v)

        # DCT: spectral_len = 128, truncated to int(128*0.50) = 64
        assert cache._cached_k_spectral.shape[-2] == 64

        k_recon, v_recon = cache.reconstruct(target_seq_len=128)
        assert k_recon.shape == k.shape


# ---------------------------------------------------------------------------
# 7. GQA repeat_kv on reconstructed tensors
# ---------------------------------------------------------------------------

class TestGQARepeatKV:

    def test_repeat_kv_shape(self, kv_tensors):
        """repeat_kv should expand 8 KV heads to 32 query heads."""
        from transformers.models.llama.modeling_llama import repeat_kv

        k, v = kv_tensors  # [1, 8, 128, 128]
        num_query_heads = 32
        num_kv_heads = 8
        repeat_factor = num_query_heads // num_kv_heads  # 4

        k_repeated = repeat_kv(k, repeat_factor)
        v_repeated = repeat_kv(v, repeat_factor)

        assert k_repeated.shape == (1, num_query_heads, 128, 128)
        assert v_repeated.shape == (1, num_query_heads, 128, 128)

    def test_repeat_kv_on_reconstructed(self, kv_tensors):
        """repeat_kv should work on reconstructed tensors from spectral cache."""
        from src.spectral.cache import CompressionConfig, SpectralKVCache

        try:
            from transformers.models.llama.modeling_llama import repeat_kv
        except ImportError:
            pytest.skip("transformers not installed")

        k, v = kv_tensors
        config = CompressionConfig(
            transform_type="fft", filter_type="fixed", gamma=0.22,
            max_seq_len=128,
        )
        cache = SpectralKVCache(config, num_kv_heads=8, head_dim=128)
        cache.compress(k, v)
        k_recon, v_recon = cache.reconstruct(target_seq_len=128)

        # repeat_kv should work on reconstructed tensors
        k_repeated = repeat_kv(k_recon, 4)
        v_repeated = repeat_kv(v_recon, 4)

        assert k_repeated.shape == (1, 32, 128, 128)
        assert v_repeated.shape == (1, 32, 128, 128)
        assert not torch.isnan(k_repeated).any()
        assert not torch.isnan(v_repeated).any()


# ---------------------------------------------------------------------------
# 8. ART ANOVA correctness on synthetic data
# ---------------------------------------------------------------------------

class TestARTAnova:

    def _make_synthetic_factorial_df(self, interaction_effect=False):
        """Create a synthetic DataFrame with known factorial structure.

        If interaction_effect=True, the learnable filter helps FFT more than DCT.
        If False, filter type has the same effect regardless of transform.
        """
        import pandas as pd

        np.random.seed(42)
        rows = []
        transforms = ["dct", "fft"]
        filters = ["fixed", "learnable"]
        gammas = [0.50, 0.22, 0.01]
        seeds = list(range(30))

        for t in transforms:
            for f in filters:
                for g in gammas:
                    for s in seeds:
                        # Base effect: FFT slightly better than DCT
                        base = 10.0
                        transform_effect = -0.5 if t == "fft" else 0.0
                        # Learnable slightly better than fixed
                        filter_effect = -0.3 if f == "learnable" else 0.0
                        # Gamma effect: lower gamma = higher PPL
                        gamma_effect = {0.50: 0.5, 0.22: 2.0, 0.01: 8.0}[g]

                        # Interaction: learnable+FFT synergistic
                        interaction = 0.0
                        if interaction_effect and t == "fft" and f == "learnable":
                            interaction = -1.5

                        noise = np.random.normal(0, 0.5)
                        value = base + transform_effect + filter_effect + gamma_effect + interaction + noise

                        rows.append({
                            "config_id": f"C{t}_{f}_{g}",
                            "transform": t,
                            "filter": f,
                            "gamma": g,
                            "seed": s,
                            "benchmark": "pg19",
                            "value": value,
                        })

        return pd.DataFrame(rows)

    def test_art_detects_main_effect_transform(self):
        """ART should detect the transform main effect."""
        import pandas as pd
        from src.stats.analyze import _art_anova

        df = self._make_synthetic_factorial_df(interaction_effect=False)
        results = _art_anova(df)

        assert "transform_pg19" in results["main_effects"]
        effect = results["main_effects"]["transform_pg19"]
        assert effect["p_value"] < 0.05, \
            f"ART should detect transform effect, p={effect['p_value']}"

    def test_art_detects_main_effect_filter(self):
        """ART should detect the filter main effect."""
        from src.stats.analyze import _art_anova

        df = self._make_synthetic_factorial_df(interaction_effect=False)
        results = _art_anova(df)

        assert "filter_pg19" in results["main_effects"]
        effect = results["main_effects"]["filter_pg19"]
        assert effect["p_value"] < 0.05, \
            f"ART should detect filter effect, p={effect['p_value']}"

    def test_art_detects_interaction(self):
        """ART should detect the transform×filter interaction when present."""
        from src.stats.analyze import _art_anova

        df = self._make_synthetic_factorial_df(interaction_effect=True)
        results = _art_anova(df)

        assert "transform_x_filter_pg19" in results["interaction"]
        effect = results["interaction"]["transform_x_filter_pg19"]
        assert effect["p_value"] < 0.05, \
            f"ART should detect interaction effect, p={effect['p_value']}"

    def test_art_no_false_interaction(self):
        """ART should NOT detect interaction when none exists."""
        from src.stats.analyze import _art_anova

        df = self._make_synthetic_factorial_df(interaction_effect=False)
        results = _art_anova(df)

        if "transform_x_filter_pg19" in results["interaction"]:
            effect = results["interaction"]["transform_x_filter_pg19"]
            # With no true interaction, p-value should be non-significant
            # (allow some tolerance since this is statistical)
            assert effect["p_value"] > 0.01, \
                f"ART false-positive on interaction, p={effect['p_value']}"

    def test_art_per_gamma_analysis(self):
        """ART per-gamma analysis should produce results for each gamma."""
        from src.stats.analyze import _art_anova

        df = self._make_synthetic_factorial_df(interaction_effect=True)
        results = _art_anova(df)

        assert "per_gamma" in results
        # Should have entries for each gamma level
        assert len(results["per_gamma"]) >= 1

    def test_art_kruskal_wallis_fallback(self):
        """Kruskal-Wallis fallback should be present."""
        from src.stats.analyze import _art_anova

        df = self._make_synthetic_factorial_df(interaction_effect=False)
        results = _art_anova(df)

        assert "kruskal_wallis_fallback" in results
        assert "transform_pg19" in results["kruskal_wallis_fallback"]
        assert "filter_pg19" in results["kruskal_wallis_fallback"]

    def test_art_method_field(self):
        """Results should include the method field."""
        from src.stats.analyze import _art_anova

        df = self._make_synthetic_factorial_df(interaction_effect=False)
        results = _art_anova(df)

        assert "method" in results
        assert "aligned_rank_transform" in results["method"]

    def test_art_partial_eta_squared(self):
        """ART results should include partial_eta_squared effect size."""
        from src.stats.analyze import _art_anova

        df = self._make_synthetic_factorial_df(interaction_effect=True)
        results = _art_anova(df)

        for key, effect in results["main_effects"].items():
            assert "partial_eta_squared" in effect
            assert 0.0 <= effect["partial_eta_squared"] <= 1.0
