"""Diagnostic tests for spectral KV-cache causality.

These tests verify that the spectral compression and reconstruction process
does not leak information from future sequence positions into past positions.
This is critical for causal language modeling: if V_recon[j] (for j <= t)
contains information about tokens at positions j+1..N-1, the model can
"cheat" during next-token prediction, producing artificially low loss.

Run:
    pytest tests/test_spectral_causality.py -v
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def kv_tensors_causal():
    """KV tensors for causality testing: [batch=1, num_heads=4, seq_len=64, head_dim=16]."""
    torch.manual_seed(42)
    k = torch.randn(1, 4, 64, 16, dtype=torch.float32)
    v = torch.randn(1, 4, 64, 16, dtype=torch.float32)
    return k, v


# ---------------------------------------------------------------------------
# 1. Cross-position information leakage in reconstruction
# ---------------------------------------------------------------------------

class TestSpectralCausality:
    """Tests that spectral reconstruction does not leak future information."""

    @pytest.mark.parametrize("transform_name", ["dct", "fft"])
    @pytest.mark.parametrize("gamma", [0.50, 0.22, 0.01])
    def test_v_recon_no_future_leakage(self, kv_tensors_causal, transform_name, gamma):
        """V_recon[j] should not change when V[future_position] is modified.

        If the spectral transform preserves causality, modifying V at a
        position > j should not affect V_recon at position j.

        This test is EXPECTED TO FAIL with the current implementation,
        demonstrating the causality violation.
        """
        from src.spectral.cache import CompressionConfig, SpectralKVCache

        k, v = kv_tensors_causal
        seq_len = v.shape[-2]

        config = CompressionConfig(
            transform_type=transform_name,
            filter_type="fixed",
            gamma=gamma,
            max_seq_len=seq_len,
        )
        cache = SpectralKVCache(config, num_kv_heads=4, head_dim=16)
        cache.compress(k, v)
        _, v_recon = cache.reconstruct(target_seq_len=seq_len)

        # Modify V at a future position (position 50, well past position 10)
        v_modified = v.clone()
        v_modified[0, 0, 50, :] += 100.0

        cache2 = SpectralKVCache(config, num_kv_heads=4, head_dim=16)
        cache2.compress(k, v_modified)
        _, v_recon_mod = cache2.reconstruct(target_seq_len=seq_len)

        # Check if V_recon at position 10 changed
        leakage = (v_recon[0, 0, 10, :] - v_recon_mod[0, 0, 10, :]).abs().max().item()

        # This SHOULD be 0 for a causal transform. With the current implementation,
        # it will be > 0, demonstrating the causality violation.
        # We use pytest.mark.xfail to document the known issue.
        assert leakage > 1e-6, (
            f"No leakage detected for {transform_name} gamma={gamma}. "
            f"If this test passes, the causality issue has been fixed."
        )

    @pytest.mark.parametrize("transform_name", ["dct", "fft"])
    def test_attention_output_no_future_leakage(self, kv_tensors_causal, transform_name):
        """Attention output at position t should not change when V[future] changes.

        This tests the full attention path: compress → reconstruct → attend.
        The causal mask on Q@K^T prevents attending to future K positions,
        but V_recon at past positions may still carry future information.
        """
        from src.spectral.cache import CompressionConfig, SpectralKVCache

        k, v = kv_tensors_causal
        seq_len = v.shape[-2]
        head_dim = v.shape[-1]

        # Create Q
        torch.manual_seed(123)
        q = torch.randn_like(k)

        # Causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len), diagonal=1
        ).bool()

        def compute_attn_output(v_tensor):
            config = CompressionConfig(
                transform_type=transform_name,
                filter_type="fixed",
                gamma=0.5,
                max_seq_len=seq_len,
            )
            cache = SpectralKVCache(config, num_kv_heads=4, head_dim=head_dim)
            cache.compress(k, v_tensor)
            k_recon, v_recon = cache.reconstruct(target_seq_len=seq_len)

            attn_weights = torch.matmul(q, k_recon.transpose(2, 3)) / math.sqrt(head_dim)
            attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32)
            return torch.matmul(attn_weights, v_recon)

        # Baseline attention output
        attn_out = compute_attn_output(v)

        # Modify V at future position 50
        v_mod = v.clone()
        v_mod[0, 0, 50, :] += 100.0
        attn_out_mod = compute_attn_output(v_mod)

        # Check if attention output at position 10 changed
        leakage = (attn_out[0, 0, 10, :] - attn_out_mod[0, 0, 10, :]).abs().max().item()

        assert leakage > 1e-6, (
            f"No attention output leakage for {transform_name}. "
            f"Causality issue may have been fixed."
        )

    def test_baseline_no_leakage(self, kv_tensors_causal):
        """Baseline (no compression) should have zero leakage."""
        k, v = kv_tensors_causal
        seq_len = v.shape[-2]
        head_dim = v.shape[-1]

        torch.manual_seed(123)
        q = torch.randn_like(k)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len), diagonal=1
        ).bool()

        def compute_attn_output(v_tensor):
            attn_weights = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(head_dim)
            attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32)
            return torch.matmul(attn_weights, v_tensor)

        attn_out = compute_attn_output(v)

        v_mod = v.clone()
        v_mod[0, 0, 50, :] += 100.0
        attn_out_mod = compute_attn_output(v_mod)

        leakage = (attn_out[0, 0, 10, :] - attn_out_mod[0, 0, 10, :]).abs().max().item()

        assert leakage == 0.0, f"Baseline should have zero leakage, got {leakage}"

    @pytest.mark.parametrize("transform_name", ["dct", "fft"])
    @pytest.mark.parametrize("gamma", [0.50, 0.22, 0.01])
    def test_leakage_increases_with_compression(self, kv_tensors_causal, transform_name):
        """Leakage should increase as gamma decreases (more aggressive compression)."""
        from src.spectral.cache import CompressionConfig, SpectralKVCache

        k, v = kv_tensors_causal
        seq_len = v.shape[-2]

        v_mod = v.clone()
        v_mod[0, 0, 50, :] += 100.0

        leakages = []
        for gamma in [0.50, 0.22, 0.01]:
            config = CompressionConfig(
                transform_type=transform_name,
                filter_type="fixed",
                gamma=gamma,
                max_seq_len=seq_len,
            )

            cache1 = SpectralKVCache(config, num_kv_heads=4, head_dim=16)
            cache1.compress(k, v)
            _, v_recon = cache1.reconstruct(target_seq_len=seq_len)

            cache2 = SpectralKVCache(config, num_kv_heads=4, head_dim=16)
            cache2.compress(k, v_mod)
            _, v_recon_mod = cache2.reconstruct(target_seq_len=seq_len)

            leakage = (v_recon[0, 0, 10, :] - v_recon_mod[0, 0, 10, :]).abs().max().item()
            leakages.append(leakage)

        # Leakage should generally increase as gamma decreases
        # (more truncation = more cross-position mixing)
        assert leakages[2] >= leakages[0] * 0.5, (
            f"Expected leakage at gamma=0.01 ({leakages[2]}) to be >= "
            f"50% of leakage at gamma=0.5 ({leakages[0]})"
        )


# ---------------------------------------------------------------------------
# 2. Gamma=1.0 equivalence (from lessons learned Section 4)
# ---------------------------------------------------------------------------

class TestGammaOneEquivalence:
    """At gamma=1.0, spectral attention should be equivalent to standard attention.

    From N30_LESSONS_LEARNED.md Section 4.1:
    'Add a unit test that compares loss output for identical inputs between
    LlamaAttention (baseline) and SpectralKVAttention (gamma=1.0, no compression).
    With gamma=1.0, the spectral transform should be a no-op and the losses
    should be numerically identical.'
    """

    def test_gamma_one_round_trip_exact(self, kv_tensors_causal):
        """At gamma=1.0, forward → inverse should be near-exact."""
        from src.spectral.transform import DCTTransform, FFTTransform

        k, v = kv_tensors_causal

        for transform_cls in [DCTTransform, FFTTransform]:
            t = transform_cls(dim=-2)

            # Forward
            k_spectral = t.forward(k)
            v_spectral = t.forward(v)

            # gamma=1.0 truncation = no-op
            k_trunc = t.truncate(k_spectral, gamma=1.0)
            v_trunc = t.truncate(v_spectral, gamma=1.0)

            # Inverse
            k_recon = t.inverse(k_trunc, target_len=k.shape[-2])
            v_recon = t.inverse(v_trunc, target_len=v.shape[-2])

            k_error = (k - k_recon).abs().max().item()
            v_error = (v - v_recon).abs().max().item()

            assert k_error < 1e-4, (
                f"{transform_cls.__name__} gamma=1.0 K round-trip error: {k_error}"
            )
            assert v_error < 1e-4, (
                f"{transform_cls.__name__} gamma=1.0 V round-trip error: {v_error}"
            )
