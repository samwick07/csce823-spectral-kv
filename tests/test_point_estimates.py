"""Tests for the N=1 point-estimate table builder."""

from __future__ import annotations

import json

from src.stats.experiment_matrix import EXPERIMENT_MATRIX
from src.stats.point_estimates import build_point_table, write_point_table


def _write_seed_results(raw_dir, config_id, seed=0):
    d = raw_dir / config_id / f"seed_{seed}"
    d.mkdir(parents=True, exist_ok=True)
    cfg = next(c for c in EXPERIMENT_MATRIX if c.config_id == config_id)
    gamma = cfg.gamma
    (d / "pg19.json").write_text(json.dumps(
        {"mean_perplexity": 8.0 + 2.0 * (1 - gamma)}))
    (d / "proof_pile.json").write_text(json.dumps(
        {"mean_perplexity": 10.0 + 4.0 * (1 - gamma)}))
    (d / "longbench.json").write_text(json.dumps(
        {"overall_mean": 0.5 + 0.2 * gamma}))
    (d / "efficiency.json").write_text(json.dumps(
        {"decoding_latency_ms_per_token": 12.0,
         "peak_kv_memory_gb": 20.0 * gamma,
         "compression_overhead_pct": 5.0}))


def test_build_point_table_values(tmp_path):
    raw = tmp_path / "raw"
    for cid in ("C00", "C10"):
        _write_seed_results(raw, cid)
    df = build_point_table(raw_dir=raw, seed=0)
    assert set(df["config_id"]) == {"C00", "C10"}
    row = df[df["config_id"] == "C10"].iloc[0]
    assert row["pg19_mean_ppl"] == 8.0 + 2.0 * 0.5
    assert row["proofpile_mean_ppl"] == 10.0 + 4.0 * 0.5
    # C00 baseline: gamma 1.0 -> no degradation
    base = df[df["config_id"] == "C00"].iloc[0]
    assert base["pg19_mean_ppl"] == 8.0
    assert base["delta_pg19_mean_ppl"] == 0.0
    # Delta for C10 vs C00
    assert abs(row["delta_pg19_mean_ppl"] - 1.0) < 1e-9
    assert abs(row["delta_proofpile_mean_ppl"] - 2.0) < 1e-9


def test_missing_config_skipped(tmp_path):
    raw = tmp_path / "raw"
    _write_seed_results(raw, "C00")
    df = build_point_table(raw_dir=raw, seed=0)
    assert set(df["config_id"]) == {"C00"}


def test_write_point_table(tmp_path):
    raw = tmp_path / "raw"
    out = tmp_path / "point"
    _write_seed_results(raw, "C00")
    _write_seed_results(raw, "C07")
    df = build_point_table(raw_dir=raw, seed=0)
    paths = write_point_table(df, output_dir=out)
    assert (out / "point_table.csv").exists()
    assert (out / "point_table.md").exists()
    md = (out / "point_table.md").read_text()
    assert "point estimate" in md.lower()
    assert "C07" in md
