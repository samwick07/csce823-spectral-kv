#!/usr/bin/env bash
# Smoke test runner for spectral KV-cache compression.
#
# Runs the full integration test with the small smoke-test model
# (default: Llama-3.2-1B-Instruct, see src/utils/constants.py)
# to verify the spectral compression pipeline works end-to-end before
# launching expensive GPU experiments.
#
# Usage:
#   bash scripts/smoke_test.sh
#   bash scripts/smoke_test.sh --model meta-llama/Llama-3.2-1B-Instruct
#   bash scripts/smoke_test.sh --transform dct --filter fixed --gamma 0.50
#   bash scripts/smoke_test.sh --skip-eval  # skip PG-19/LongBench (no network)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "============================================================"
echo "  SPECTRAL KV-CACHE SMOKE TEST"
echo "  Project: $PROJECT_ROOT"
echo "  Time: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "============================================================"

# Check GPU availability
if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "  GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
    echo "  CUDA: $(python -c 'import torch; print(torch.version.cuda)')"
else
    echo "  WARNING: No GPU detected. Tests will likely fail."
    echo "  This smoke test requires a CUDA GPU."
fi

echo ""

# Run the smoke test
python -m src.tests.smoke_test "$@"

EXIT_CODE=$?

echo ""
echo "============================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "  SMOKE TEST PASSED — pipeline is ready for GPU experiments."
else
    echo "  SMOKE TEST FAILED — fix issues before running experiments."
fi
echo "============================================================"

exit $EXIT_CODE
