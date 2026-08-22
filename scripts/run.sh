#!/usr/bin/env bash
# =============================================================================
# run.sh — Single entry point for the spectral KV-cache experiment.
#
# Launches the orchestrator with nohup + setsid, logging all stdout/stderr
# to a timestamped log file. Survives SSH disconnects and workspace agent
# restarts. No tmux dependency.
#
# Usage:
#   bash scripts/run.sh                  # full experiment (13 configs x num_seeds from YAML)
#   bash scripts/run.sh --seeds 0        # N=1 point-estimate run (13 configs x 1 seed)
#   bash scripts/run.sh --pilot          # pilot (3 configs x 5 seeds)
#   bash scripts/run.sh --phase train    # training only
#   bash scripts/run.sh --phase eval     # eval only
#   bash scripts/run.sh --phase analyze  # statistical analysis only
#   bash scripts/run.sh --config C01     # single config
#   bash scripts/run.sh --status         # print current progress + process status
#   bash scripts/run.sh --stop           # gracefully stop the experiment
#   bash scripts/run.sh --log            # tail the latest log file (Ctrl+C to exit)
#
# Environment variables:
#   HF_TOKEN       — HuggingFace access token (required for model download)
#   WANDB_API_KEY  — Weights & Biases API key (optional, for experiment tracking)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

VENV_DIR="$PROJECT_ROOT/.venv"
LOG_DIR="$PROJECT_ROOT/logs"
PID_FILE="$PROJECT_ROOT/.orchestrator_pid"
mkdir -p "$LOG_DIR"

TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# --- Helper functions ---

print_header() {
    echo -e "${BLUE}============================================================${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}============================================================${NC}"
}

# --- Status mode ---
if [[ "${1:-}" == "--status" ]]; then
    print_header "EXPERIMENT STATUS"
    if [[ -f "$VENV_DIR/bin/python" ]]; then
        "$VENV_DIR/bin/python" -m src.orchestrator --status
    else
        echo -e "${YELLOW}venv not found. Run: bash scripts/setup_env.sh${NC}"
        # Try system python as fallback
        if [[ -f "results/orchestrator_state.json" ]]; then
            python3 -c "
import json
with open('results/orchestrator_state.json') as f:
    s = json.load(f)
print(f\"  Start time: {s.get('start_time', 'unknown')}\")
print(f\"  Last update: {s.get('last_update', 'unknown')}\")
print(f\"  Current task: {s.get('current_task', 'none')}\")
print(f\"  Training complete: {len(s.get('completed_training', []))} configs\")
print(f\"  Evals complete: {len(s.get('completed_evals', []))}\")
print(f\"  Analysis complete: {s.get('completed_analysis', False)}\")
print(f\"  Exfil complete: {s.get('completed_exfil', False)}\")
print(f\"  Crash count: {s.get('crash_count', 0)}\")
"
        else
            echo "  No state file found. Experiment has not been run yet."
        fi
    fi
    # Show process status
    echo ""
    if [[ -f "$PID_FILE" ]]; then
        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            echo -e "  Process: ${GREEN}RUNNING${NC} (PID: $PID)"
        else
            echo -e "  Process: ${YELLOW}STOPPED${NC} (PID $PID no longer alive)"
        fi
    else
        echo -e "  Process: ${YELLOW}NOT RUNNING${NC}"
    fi
    exit 0
fi

# --- Stop mode ---
if [[ "${1:-}" == "--stop" ]]; then
    print_header "STOPPING EXPERIMENT"
    if [[ -f "$PID_FILE" ]]; then
        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            # Send SIGTERM to the process group (negative PID = whole group).
            # setsid at launch time made the orchestrator a group leader, so
            # this kills Python + any DeepSpeed subprocesses in one shot.
            kill -TERM "-$PID" 2>/dev/null || kill -TERM "$PID" 2>/dev/null
            echo -e "${YELLOW}Sent SIGTERM to process group $PID.${NC}"
            echo -e "${YELLOW}The orchestrator will finish the current subprocess and then stop.${NC}"
            # Wait up to 15 seconds for graceful shutdown
            for i in $(seq 1 15); do
                kill -0 "$PID" 2>/dev/null || break
                sleep 1
            done
            if kill -0 "$PID" 2>/dev/null; then
                echo -e "${RED}Process did not exit after 15s, sending SIGKILL.${NC}"
                kill -KILL "-$PID" 2>/dev/null || kill -KILL "$PID" 2>/dev/null
            else
                echo -e "${GREEN}Process exited cleanly.${NC}"
            fi
            rm -f "$PID_FILE"
        else
            echo -e "${YELLOW}PID $PID is not running. Cleaning up PID file.${NC}"
            rm -f "$PID_FILE"
        fi
    else
        echo -e "${YELLOW}No PID file found. Experiment may not be running.${NC}"
        echo "Check for orphaned processes: pgrep -f 'src.orchestrator'"
    fi
    exit 0
fi

# --- Log mode (replaces --attach) ---
if [[ "${1:-}" == "--log" ]]; then
    LATEST_LOG="$LOG_DIR/latest.log"
    if [[ -L "$LATEST_LOG" ]]; then
        LATEST_LOG=$(readlink -f "$LATEST_LOG")
    elif [[ ! -f "$LATEST_LOG" ]]; then
        LATEST_LOG=$(ls -t "$LOG_DIR"/run_*.log 2>/dev/null | head -1)
    fi
    if [[ -n "$LATEST_LOG" && -f "$LATEST_LOG" ]]; then
        echo "Tailing $LATEST_LOG (Ctrl+C to exit)"
        echo ""
        tail -f "$LATEST_LOG"
    else
        echo "No log files found in $LOG_DIR"
        exit 1
    fi
    exit 0
fi

# --- Check if already running ---
if [[ -f "$PID_FILE" ]]; then
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        echo -e "${YELLOW}Experiment already running (PID: $PID).${NC}"
        echo "  Stop:    bash scripts/run.sh --stop"
        echo "  Status:  bash scripts/run.sh --status"
        echo "  Log:     bash scripts/run.sh --log"
        exit 1
    else
        echo -e "${YELLOW}Stale PID file found (PID $PID not running). Removing.${NC}"
        rm -f "$PID_FILE"
    fi
fi

# --- Check venv ---
if [[ ! -f "$VENV_DIR/bin/python" ]]; then
    echo -e "${RED}Virtual environment not found at $VENV_DIR${NC}"
    echo -e "${YELLOW}Run: bash scripts/setup_env.sh${NC}"
    exit 1
fi

# --- Check GPU ---
if ! command -v nvidia-smi &>/dev/null; then
    echo -e "${RED}nvidia-smi not found. This experiment requires GPUs.${NC}"
    exit 1
fi

GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l)
echo -e "${GREEN}GPUs detected: $GPU_COUNT${NC}"

# --- Check disk space (need ~200GB for model + datasets + checkpoints) ---
AVAILABLE_GB=$(df -BG --output=avail "$PROJECT_ROOT" 2>/dev/null | tail -1 | tr -dc '0-9')
if [[ -n "$AVAILABLE_GB" && "$AVAILABLE_GB" -lt 200 ]]; then
    echo -e "${RED}INSUFFICIENT DISK SPACE: ${AVAILABLE_GB}GB available${NC}"
    echo -e "${RED}Need at least 200GB for model (16GB) + datasets (50GB) + checkpoints (50GB) + cache${NC}"
    exit 1
fi
echo -e "${GREEN}Disk space: ${AVAILABLE_GB:-?}GB available${NC}"

# --- Check HF token ---
if [[ -z "${HF_TOKEN:-}" ]]; then
    echo -e "${YELLOW}WARNING: HF_TOKEN not set. Model download will fail${NC}"
    echo -e "${YELLOW}for gated models. Set it with: export HF_TOKEN=hf_your_token${NC}"
fi

# --- Restore HF cache from PVC backup ---
# The HF cache (~55GB: Llama-3.1-8B-Instruct + dataset caches) lives under
# $HOME/.cache, which is the ephemeral container overlay — wiped on every
# workspace stop/start. /workspaces is a persistent PVC, so a backup of the
# cache there survives restarts. If the cache is gone but a backup exists,
# restore it to avoid a ~16GB gated-model re-download.
HF_CACHE_DIR="$HOME/.cache/huggingface"
HF_BACKUP_DIR="/workspaces/hf-cache-backup"
MODEL_MARKER="$HF_CACHE_DIR/hub/models--meta-llama--Llama-3.1-8B-Instruct"
BACKUP_MARKER="$HF_BACKUP_DIR/hub/models--meta-llama--Llama-3.1-8B-Instruct"
if [[ "${1:-}" == "--backup-cache" ]]; then
    # Refresh the backup (e.g. after a run added dataset caches).
    mkdir -p "$HF_BACKUP_DIR"
    cp -a "$HF_CACHE_DIR/." "$HF_BACKUP_DIR/"
    echo -e "${GREEN}HF cache backed up to $HF_BACKUP_DIR: $(du -sh "$HF_BACKUP_DIR" | cut -f1)${NC}"
    exit 0
fi
if [[ ! -d "$MODEL_MARKER" && -d "$BACKUP_MARKER" ]]; then
    echo -e "${YELLOW}HF cache missing (workspace restart?). Restoring from $HF_BACKUP_DIR...${NC}"
    mkdir -p "$HF_CACHE_DIR"
    cp -a "$HF_BACKUP_DIR/." "$HF_CACHE_DIR/"
    echo -e "${GREEN}HF cache restored: $(du -sh "$HF_CACHE_DIR" | cut -f1)${NC}"
fi

# --- Pass through remaining args to orchestrator ---
ORCH_ARGS=("$@")

# --- Build the command ---
# Source venv, set env vars, run orchestrator.
# WANDB_MODE=offline is set as a fallback: if the network drops mid-run,
# W&B will buffer logs locally and sync them when connectivity returns.
COMMAND="source $VENV_DIR/bin/activate && "
COMMAND+="export HF_TOKEN=\"${HF_TOKEN:-}\" && "
COMMAND+="export WANDB_API_KEY=\"${WANDB_API_KEY:-}\" && "
COMMAND+="export WANDB_MODE=\"${WANDB_MODE:-online}\" && "
COMMAND+="export HF_DATASETS_TRUST_REMOTE_CODE=1 && "
COMMAND+="python -m src.orchestrator ${ORCH_ARGS[*]:-}"

LOG_FILE="$LOG_DIR/run_${TIMESTAMP}.log"

# --- Launch with nohup + setsid ---
# setsid creates a new session/process group so --stop can kill the
# entire tree (Python + DeepSpeed subprocesses) with a single
# `kill -- -$PID`. nohup ignores SIGHUP so the process survives
# SSH/VPN disconnects and workspace agent restarts.
print_header "LAUNCHING EXPERIMENT"
echo "  Log file: $LOG_FILE"
echo "  PID file: $PID_FILE"
echo "  Command:  python -m src.orchestrator ${ORCH_ARGS[*]:-}"
echo ""
echo -e "${GREEN}The experiment will continue running even if your SSH/VPN${NC}"
echo -e "${GREEN}connection drops. To monitor:${NC}"
echo "    bash scripts/run.sh --status    # print progress + process status"
echo "    bash scripts/run.sh --log       # tail the latest log (Ctrl+C to exit)"
echo "    bash scripts/monitor.sh         # one-shot dashboard"
echo "    tail -f $LOG_FILE             # raw log tail"

setsid nohup bash -c "cd $PROJECT_ROOT && $COMMAND" > "$LOG_FILE" 2>&1 &
PID=$!
echo "$PID" > "$PID_FILE"

# Convenience symlink so --log and monitor.sh always find the latest
ln -sf "$LOG_FILE" "$LOG_DIR/latest.log"

echo -e "\n${GREEN}Started with PID $PID${NC}"
echo "  Stop:  bash scripts/run.sh --stop"
echo "  Log:   bash scripts/run.sh --log"
