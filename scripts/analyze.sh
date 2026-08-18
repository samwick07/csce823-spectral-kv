#!/bin/bash
# Aggregate results and run statistical analysis
# Usage: bash scripts/analyze.sh

set -euo pipefail

echo "=== Aggregating raw results ==="
python -m src.stats.aggregate

echo ""
echo "=== Running statistical analysis ==="
python -m src.stats.analyze

echo ""
echo "=== Analysis complete ==="
echo "Results in:"
echo "  results/aggregated/per_seed.csv"
echo "  results/aggregated/summary.csv"
echo "  results/aggregated/summary.json"
echo "  results/statistical_analysis.json"
