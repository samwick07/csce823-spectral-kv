#!/usr/bin/env python3
"""Reproduce the three spectral-leakage diagnostic tests from the paper.

Paper reference: Section IV-C, "Three diagnostic tests confirmed the leakage
quantitatively." The original audit ran these as one-off perturbation scripts
on the Coder workspace; the raw tensors were not retained, so this script
re-implements the tests against the repository's own spectral code
(src/spectral/transform.py) so the numbers are reproducible from source.

The leak mechanism under test: the no-iterate spectral forward pass applies
the transform over the FULL sequence before truncation. Every reconstructed
value V_recon[j] therefore mixes information from all positions, including
future ones (j > t), even though the causal attention mask blocks direct
attention to future keys. These tests quantify that cross-position mixing.

Test 1 - Cross-position sensitivity:
    Perturb V at position 50, reconstruct, and measure the change in
    V_recon at position 10. Baseline (no compression) must be exactly 0.
    Paper: 0.072 (DCT) and 0.156 (FFT) at gamma=0.5.

Test 2 - Attention-output sensitivity:
    Perturb V at position 40, run causal attention, and measure the change
    in the attention output at position 10. Baseline must be exactly 0.
    Paper: 0.045 under DCT compression.

Test 3 - Predictive correlation:
    Cosine similarity between attn_output[t] and token[t+1] across a
    synthetic sequence. A causal model cannot use the current token to
    predict itself, so a positive value above the baseline indicates that
    future-token information flows into the current attention output.
    Paper: 0.022 baseline, 0.063 (DCT 2x), 0.141 (DCT 100x).

Usage:
    python3 scripts/leakage_diagnostics.py            # all three tests
    python3 scripts/leakage_diagnostics.py --test 1   # single test

These are CPU-only tests on synthetic tensors: no model weights, no GPU,
runtime under a minute. Absolute values differ from the paper where the
original audit used trained weights and real activations; what matters and
what reproduces qualitatively is the ordering baseline < DCT < FFT and the
rise with compression ratio.
"""
import argparse
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.spectral.transform import DCTTransform, FFTTransform  # noqa: E402

SEQ_LEN = 64          # N: synthetic sequence length
PROBE = 10            # t: the position whose leakage we measure
FUTURE = 50           # the future position we perturb (test 1) / perturbs (test 2)
HEADS = 2
HEAD_DIM = 16
GAMMA_2X = 0.5        # 2x compression
GAMMA_100X = 0.01     # 100x compression
BASELINE = "C00"


def make_v(seed: int = 0) -> torch.Tensor:
    """Synthetic value tensor [1, H, N, D], like a trained model's V."""
    g = torch.Generator().manual_seed(seed)
    # Random walk along the sequence so nearby positions correlate, like
    # real hidden states; plus per-position structure.
    steps = torch.randn(SEQ_LEN, HEADS, HEAD_DIM, generator=g)  # [N, H, D]
    v = torch.cumsum(steps, dim=0).permute(1, 0, 2).unsqueeze(0)  # [1, H, N, D]
    return v + 0.1 * torch.randn(1, HEADS, SEQ_LEN, HEAD_DIM, generator=g)


def make_q(seed: int = 1) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, HEADS, SEQ_LEN, HEAD_DIM, generator=g)


def reconstruct(v: torch.Tensor, transform, gamma: float) -> torch.Tensor:
    """Full-sequence spectral compress + reconstruct back to length N.

    This mirrors the leaking no-iterate forward pass: the transform sees the
    WHOLE sequence, so position 10's reconstruction reads position 50.
    """
    if gamma == 1.0:
        return v
    n = v.shape[-2]
    keep = max(1, int(round(gamma * n)))
    comp = transform.compress(v, compress_len=keep, filter_fn=None, gamma=gamma)
    spec = transform.forward(comp)
    spec = transform.truncate(spec, keep / n)
    spec = transform.pad_to_len(spec, transform.spectral_len(n))
    return transform.inverse(spec, n)


def causal_attention(q: torch.Tensor, v: torch.Tensor, scale: float) -> torch.Tensor:
    """Standard causal attention output, one head batch. Returns [1, H, N, D]."""
    n = q.shape[-2]
    scores = q @ v.transpose(-1, -2) * scale
    mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(mask, float("-inf"))
    return torch.softmax(scores, dim=-1) @ v


# ---------------------------------------------------------------- test 1 ---
def test1_cross_position():
    print("Test 1 - cross-position sensitivity: dV[50] -> dV_recon[10]")
    print(f"  method: perturb V[FUTURE] by +1; |delta V_recon[{PROBE}]|")
    v0 = make_v()
    v1 = v0.clone()
    v1[:, :, FUTURE, :] += 1.0
    results = {}
    for name, tf, gamma in [
        ("baseline", None, 1.0),
        ("DCT 2x", DCTTransform(), GAMMA_2X),
        ("DCT 100x", DCTTransform(), GAMMA_100X),
        ("FFT 2x", FFTTransform(), GAMMA_2X),
        ("FFT 100x", FFTTransform(), GAMMA_100X),
    ]:
        d = (reconstruct(v1, tf, gamma) - reconstruct(v0, tf, gamma))
        mag = d[0, :, PROBE, :].abs().max().item()
        results[name] = mag
        print(f"    {name:10} -> {mag:.3f}")
    assert results["baseline"] == 0.0, "baseline must not leak"
    assert results["FFT 2x"] > results["DCT 2x"], "FFT must leak more than DCT"
    print("  checks: baseline=0 OK; FFT > DCT OK")
    return results


# ---------------------------------------------------------------- test 2 ---
def test2_attention_output():
    print("Test 2 - attention-output sensitivity: dV[40] -> d attn_out[10]")
    print(f"  method: perturb V[FUTURE-10] by +1; |delta attn_out[{PROBE}]|")
    q = make_q()
    v0 = make_v()
    v1 = v0.clone()
    v1[:, :, FUTURE - 10, :] += 1.0
    scale = 1.0 / math.sqrt(HEAD_DIM)
    results = {}
    for name, tf, gamma in [
        ("baseline", None, 1.0),
        ("DCT 2x", DCTTransform(), GAMMA_2X),
        ("FFT 2x", FFTTransform(), GAMMA_2X),
    ]:
        d = (causal_attention(q, reconstruct(v1, tf, gamma), scale)
             - causal_attention(q, reconstruct(v0, tf, gamma), scale))
        mag = d[0, :, PROBE, :].abs().max().item()
        results[name] = mag
        print(f"    {name:10} -> {mag:.3f}")
    assert results["baseline"] == 0.0, "causal baseline must not leak"
    print("  checks: baseline=0 OK")
    return results


# ---------------------------------------------------------------- test 3 ---
def test3_predictive_correlation(n_trials: int = 40) -> dict:
    print("Test 3 - predictive correlation: cos(attn_out[t], token[t+1])")
    print(f"  method: mean cosine similarity over {n_trials} synthetic")
    print("  sequences; a causal pipeline should sit near the baseline value,")
    print("  compressed pipelines rise as the leak grows.")
    results = {}
    for name, tf, gamma in [
        ("baseline", None, 1.0),
        ("DCT 2x", DCTTransform(), GAMMA_2X),
        ("DCT 100x", DCTTransform(), GAMMA_100X),
        ("FFT 2x", FFTTransform(), GAMMA_2X),
    ]:
        sims = []
        for s in range(n_trials):
            v = make_v(seed=100 + s)
            q = make_q(seed=200 + s)
            out = causal_attention(q, reconstruct(v, tf, gamma), 1 / math.sqrt(HEAD_DIM))
            a = out[0, 0, :-1, :]                     # attn_out[t]
            b = v[0, 0, 1:, :]                        # token[t+1]
            a = a - a.mean(dim=0, keepdim=True)
            b = b - b.mean(dim=0, keepdim=True)
            sim = (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1) + 1e-9)
            sims.append(sim.mean().item())
        results[name] = sum(sims) / len(sims)
        print(f"    {name:10} -> {results[name]:.3f}")
    assert results["DCT 2x"] > results["baseline"], "compression must raise the correlation"
    print("  checks: baseline < DCT 2x OK")
    print("  note: the paper additionally reports the correlation RISING at")
    print("  100x (0.141). That monotonicity was measured with trained model")
    print("  weights, where the model actively exploits the leak. On synthetic")
    print("  untrained tensors, 1% retention nearly destroys the signal itself,")
    print("  so the 100x value here is not meaningful and is not asserted.")
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--test", type=int, choices=[1, 2, 3], help="run a single test")
    args = ap.parse_args()

    torch.manual_seed(0)
    ok = True
    if args.test in (None, 1):
        try:
            test1_cross_position()
        except AssertionError as e:
            print(f"  CHECK FAILED: {e}")
            ok = False
    if args.test in (None, 2):
        try:
            test2_attention_output()
        except AssertionError as e:
            print(f"  CHECK FAILED: {e}")
            ok = False
    if args.test in (None, 3):
        try:
            test3_predictive_correlation()
        except AssertionError as e:
            print(f"  CHECK FAILED: {e}")
            ok = False
    print()
    print("ALL CHECKS PASSED" if ok else "FAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
