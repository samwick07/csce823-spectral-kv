#!/bin/bash
# Relaunch script: self-contained experiment (re)start.
#
# Requirements:
#   1. Tokens live in /workspaces/.env.spectral (PVC, outside git,
#      survives workspace restarts):
#        export HF_TOKEN="hf_..."
#        export WANDB_API_KEY="wandb_..."
#        export WANDB_MODE="online"
#   2. HF model cache is auto-restored by scripts/run.sh from
#      /workspaces/hf-cache-backup if missing.
#
# Usage (from anywhere, including a future session):
#   /opt/coder ssh CSCE823-Spectral-KV-Cache 'bash /workspaces/csce823-spectral-kv/scripts/relaunch.sh'
#
# The orchestrator is idempotent: completed configs/evals are skipped,
# in-flight training resumes from the latest DeepSpeed checkpoint.
set -e
cd /workspaces/csce823-spectral-kv

if [ -f /workspaces/.env.spectral ]; then
  # shellcheck disable=SC1091
  source /workspaces/.env.spectral
else
  echo "WARNING: /workspaces/.env.spectral not found — relying on ambient env vars."
fi

# Pull latest fixes (code is the single source of truth on GitHub)
git pull --ff-only 2>&1 || echo "WARN: git pull failed, continuing with local code"
git log --oneline -1

bash scripts/run.sh --seeds 0
