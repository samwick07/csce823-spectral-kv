"""Gate 5 smoke test: tiny model forward pass with FreqKV compression.

This is the test that would have caught the 100x loss gap. The old code
produced loss ~0.02 for compressed configs (vs ~2.09 baseline) because
future information leaked through spectral reconstruction. The new chunk-wise
approach should produce loss in a reasonable range (compressed loss should be
HIGHER than baseline, not lower, due to reconstruction error).
"""

import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest

sys.path.insert(0, "/opt/csce823-spectral-kv")

from src.spectral.cache import CompressionConfig, SpectralKVCompressor
from src.spectral.attention import _compute_attention_sdpa


def _repeat_kv(kv, n_rep):
    if n_rep == 1:
        return kv
    return kv.repeat_interleave(n_rep, dim=1)


class TinyAttention(nn.Module):
    """Minimal attention layer (baseline, no compression)."""
    def __init__(self, dim=64, num_heads=4, num_kv_heads=2):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, dim, bias=False)

    def forward(self, x):
        B, S, D = x.shape
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        k_exp = _repeat_kv(k, self.num_heads // self.num_kv_heads)
        v_exp = _repeat_kv(v, self.num_heads // self.num_kv_heads)
        attn = torch.matmul(q, k_exp.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.triu(torch.ones(S, S, device=x.device, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(x.dtype)
        out = torch.matmul(attn, v_exp)
        out = out.transpose(1, 2).reshape(B, S, -1)
        return self.o_proj(out)


class TinyFreqKVAttention(nn.Module):
    """Tiny attention with FreqKV chunk-wise compression."""
    def __init__(self, dim=64, num_heads=4, num_kv_heads=2, compressor=None, config=None):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, dim, bias=False)
        self.compressor = compressor
        self.config = config

    def forward(self, x):
        B, S, D = x.shape
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)

        config = self.config
        cache_size = min(config.cache_size, S)
        sink_size = config.sink_size
        recent_size = config.recent_size
        fft_span = cache_size - sink_size - recent_size
        fft_size = int(fft_span * config.gamma)
        chunk_size = fft_span - fft_size

        if S <= cache_size:
            k_exp = _repeat_kv(k, self.num_heads // self.num_kv_heads)
            v_exp = _repeat_kv(v, self.num_heads // self.num_kv_heads)
            out = _compute_attention_sdpa(q, k_exp, v_exp, is_causal=True)
        else:
            num_groups = math.ceil((S - cache_size) / chunk_size)
            q_first = q[:, :, :cache_size, :]
            k_first = k[:, :, :cache_size, :]
            v_first = v[:, :, :cache_size, :]
            k_first_exp = _repeat_kv(k_first, self.num_heads // self.num_kv_heads)
            v_first_exp = _repeat_kv(v_first, self.num_heads // self.num_kv_heads)
            out = _compute_attention_sdpa(q_first, k_first_exp, v_first_exp, is_causal=True)
            group_key = k[:, :, :cache_size, :]
            group_value = v[:, :, :cache_size, :]

            for group_idx in range(num_groups):
                chunk_start = cache_size + group_idx * chunk_size
                chunk_end = min(cache_size + (group_idx + 1) * chunk_size, S)
                if chunk_end <= chunk_start:
                    break
                q_chunk = q[:, :, chunk_start:chunk_end, :]
                if fft_size > 0 and group_key.shape[-2] > sink_size + recent_size:
                    compress_end = min(sink_size + fft_span, group_key.shape[-2] - recent_size)
                    if compress_end > sink_size:
                        to_compress_k = group_key[:, :, sink_size:compress_end, :]
                        to_compress_v = group_value[:, :, sink_size:compress_end, :]
                        compressed_k = self.compressor.compress(to_compress_k, fft_size)
                        compressed_v = self.compressor.compress(to_compress_v, fft_size)
                    else:
                        compressed_k = group_key[:, :, sink_size:sink_size, :]
                        compressed_v = group_value[:, :, sink_size:sink_size, :]
                else:
                    compressed_k = group_key[:, :, sink_size:sink_size, :]
                    compressed_v = group_value[:, :, sink_size:sink_size, :]
                recent_start = max(group_key.shape[-2] - recent_size, sink_size)
                recent_k = group_key[:, :, recent_start:, :]
                recent_v = group_value[:, :, recent_start:, :]
                chunk_k = k[:, :, chunk_start:chunk_end, :]
                chunk_v = v[:, :, chunk_start:chunk_end, :]
                group_k_full = torch.cat([
                    group_key[:, :, :sink_size, :], compressed_k, recent_k, chunk_k
                ], dim=-2)
                group_v_full = torch.cat([
                    group_value[:, :, :sink_size, :], compressed_v, recent_v, chunk_v
                ], dim=-2)
                k_group_exp = _repeat_kv(group_k_full, self.num_heads // self.num_kv_heads)
                v_group_exp = _repeat_kv(group_v_full, self.num_heads // self.num_kv_heads)
                chunk_out = _compute_attention_sdpa(q_chunk, k_group_exp, v_group_exp, is_causal=True)
                out = torch.cat([out, chunk_out], dim=-2)
                group_key = group_k_full
                group_value = group_v_full

        out = out.transpose(1, 2).reshape(B, S, -1)
        return self.o_proj(out)


class TinyLM(nn.Module):
    def __init__(self, attn):
        super().__init__()
        self.embed = nn.Embedding(100, 64)
        self.attn = attn
        self.norm = nn.LayerNorm(64)
        self.lm_head = nn.Linear(64, 100, bias=False)

    def forward(self, input_ids):
        x = self.embed(input_ids)
        x = x + self.attn(self.norm(x))
        return self.lm_head(x)


class TestSmokeTestLossSanity:
    """The test that would have caught the 100x loss gap."""

    @pytest.mark.parametrize("transform_type,filter_type,gamma", [
        ("dct", "fixed", 0.5),
        ("fft", "fixed", 0.5),
        ("dct", "learnable", 0.5),
        ("fft", "learnable", 0.5),
        ("dct", "fixed", 0.22),
        ("dct", "fixed", 0.01),
    ])
    def test_compressed_loss_not_suspiciously_low(self, transform_type, filter_type, gamma):
        """Compressed loss should NOT be dramatically lower than baseline.

        Old broken code: ~0.02 (100x lower than ~2.09 baseline) due to
        future leakage. New chunk-wise code: should be >= 1.0x baseline.
        """
        torch.manual_seed(42)
        D, S, VOCAB = 64, 128, 100

        baseline_model = TinyLM(TinyAttention(D))

        torch.manual_seed(42)
        config = CompressionConfig(
            transform_type=transform_type, filter_type=filter_type, gamma=gamma,
            cache_size=64, sink_size=4, recent_size=8, max_seq_len=256,
        )
        compressor = SpectralKVCompressor(config, num_kv_heads=2, head_dim=D // 4)
        compressed_model = TinyLM(TinyFreqKVAttention(D, compressor=compressor, config=config))
        compressed_model.load_state_dict(baseline_model.state_dict(), strict=False)

        input_ids = torch.randint(0, VOCAB, (1, S))
        labels = input_ids.clone()

        baseline_model.eval()
        compressed_model.eval()

        with torch.no_grad():
            baseline_logits = baseline_model(input_ids)
            compressed_logits = compressed_model(input_ids)

        shift_labels = labels[..., 1:].contiguous()
        loss_baseline = F.cross_entropy(
            baseline_logits[..., :-1, :].contiguous().view(-1, VOCAB),
            shift_labels.view(-1),
        ).item()
        loss_compressed = F.cross_entropy(
            compressed_logits[..., :-1, :].contiguous().view(-1, VOCAB),
            shift_labels.view(-1),
        ).item()

        ratio = loss_compressed / loss_baseline

        # Compressed loss should be >= 0.5x baseline (not 100x lower!)
        assert ratio >= 0.5, (
            f"Causality violation detected: {transform_type}/{filter_type}/gamma={gamma}: "
            f"compressed loss {loss_compressed:.4f} is {1/ratio:.1f}x LOWER than "
            f"baseline {loss_baseline:.4f} (ratio={ratio:.2f})"
        )
        # Should not be NaN/inf
        assert not math.isnan(loss_compressed), "NaN loss"
        assert not math.isinf(loss_compressed), "inf loss"
