#!/bin/bash
# Relaunch script: self-contained experiment (re)start.
#
# Requirements:
#   1. Tokens live in /workspaces/.env.spectral (PVC, outside git,
#      survives workspace restarts):
#        export HF_TOKEN="hf_..."; export WANDB_API_KEY="wandb_..."; export WANDB_MODE="online"
#   2. HF model cache is auto-restored by scripts/run.sh from
#      /workspaces/hf-cache-backup if missing.
#
# Usage (from anywhere, including a future session):
#   bash scripts/relaunch.sh                              # default: full sweep
#   bash scripts/relaunch.sh --phase eval --eval-phase quick --seeds 0   # quick eval only
#   bash scripts/relaunch.sh --phase eval --eval-phase longbench --seeds 0  # longbench only
#
# The orchestrator is idempotent: completed configs/evals are skipped,
# in-flight training resumes from the latest DeepSpeed checkpoint.
set -e
cd /workspaces/csce823-spectral-kv

if [ -f /workspaces/.env.spectral ]; then
  # shellcheck disable=SC1091
  source /workspaces/.env.spectral

# Pin the W&B project
export WANDB_PROJECT="${WANDB_PROJECT:-csce823-spectral-kv}"
else
  echo "WARNING: /workspaces/.env.spectral not found — relying on ambient env vars."
fi

# Pull latest fixes (code is the single source of truth on GitHub)
git pull --ff-only 2>&1 || echo "WARN: git pull failed, continuing with local code"
git log --oneline -1

# Pass through all arguments to run.sh → orchestrator
# If no args given, check for a saved eval phase (written by the orchestrator)
if [ $# -eq 0 ]; then
  if [ -f /workspaces/csce823-spectral-kv/.eval_phase ]; then
    SAVED_PHASE=$(cat /workspaces/csce823-spectral-kv/.eval_phase)
    echo "Resuming with saved eval phase: $SAVED_PHASE"
    bash scripts/run.sh --phase eval --eval-phase "$SAVED_PHASE" --seeds 0
  else
    bash scripts/run.sh --seeds 0
  fi
else
  # Save the eval phase for watchdog restarts
  for arg in "$@"; do
    case $arg in
      quick|longbench|all)
        echo "$arg" > /workspaces/csce823-spectral-kv/.eval_phase
        ;;
    esac
  done
  # Also handle --eval-phase X format
  prev=""
  for arg in "$@"; do
    if [ "$prev" = "--eval-phase" ]; then
      echo "$arg" > /workspaces/csce823-spectral-kv/.eval_phase
    fi
    prev="$arg"
  done
  bash scripts/run.sh "$@"
fi
