#!/usr/bin/env bash
# =============================================================================
# run.sh — Single entry point for the spectral KV-cache experiment.
#
# This script is designed for resilience:
#   - Launches the orchestrator inside a tmux session that survives
#     SSH disconnects and VPN drops.
#   - Uses nohup as a fallback if tmux is not available.
#   - The orchestrator itself auto-detects completed work and resumes
#     from the last checkpoint after crashes or power outages.
#
# Usage:
#   bash scripts/run.sh                  # full experiment (14 configs x 30 seeds)
#   bash scripts/run.sh --pilot          # pilot (3 configs x 5 seeds)
#   bash scripts/run.sh --phase train    # training only
#   bash scripts/run.sh --phase eval     # eval only
#   bash scripts/run.sh --phase analyze  # statistical analysis only
#   bash scripts/run.sh --config C01     # single config
#   bash scripts/run.sh --status         # print current progress
#   bash scripts/run.sh --attach         # attach to running tmux session
#   bash scripts/run.sh --stop           # gracefully stop the experiment
#
# Environment variables:
#   HF_TOKEN       — HuggingFace access token (required for model download)
#   WANDB_API_KEY  — Weights & Biases API key (optional, for experiment tracking)
#   SESSION_NAME   — tmux session name (default: spectral-kv)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

SESSION_NAME="${SESSION_NAME:-spectral-kv}"
VENV_DIR="$PROJECT_ROOT/.venv"
LOG_DIR="$PROJECT_ROOT/logs"
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
print(f\"  Crash count: {s.get('crash_count', 0)}\")
"
        else
            echo "  No state file found. Experiment has not been run yet."
        fi
    fi
    exit 0
fi

# --- Attach mode ---
if [[ "${1:-}" == "--attach" ]]; then
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        exec tmux attach -t "$SESSION_NAME"
    else
        echo -e "${YELLOW}No tmux session '$SESSION_NAME' found.${NC}"
        echo "Is the experiment running? Check: bash scripts/run.sh --status"
        exit 1
    fi
fi

# --- Stop mode ---
if [[ "${1:-}" == "--stop" ]]; then
    print_header "STOPPING EXPERIMENT"
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        # Send Ctrl+C (SIGINT) for graceful shutdown
        tmux send-keys -t "$SESSION_NAME" C-c
        echo -e "${YELLOW}Sent SIGINT to tmux session. The orchestrator will"
        echo -e "finish the current subprocess and then stop.${NC}"
        echo "Attach to watch: bash scripts/run.sh --attach"
    else
        echo -e "${YELLOW}No tmux session found. Experiment may not be running.${NC}"
    fi
    exit 0
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

# --- Check HF token ---
if [[ -z "${HF_TOKEN:-}" ]]; then
    echo -e "${YELLOW}WARNING: HF_TOKEN not set. Model download will fail${NC}"
    echo -e "${YELLOW}for gated models. Set it with: export HF_TOKEN=hf_your_token${NC}"
fi

# --- Pass through remaining args to orchestrator ---
ORCH_ARGS=("$@")

# --- Build the command ---
# Source venv, set env vars, run orchestrator
COMMAND="source $VENV_DIR/bin/activate && "
COMMAND+="export HF_TOKEN=\"\${HF_TOKEN:-}\" && "
COMMAND+="export WANDB_API_KEY=\"\${WANDB_API_KEY:-}\" && "
COMMAND+="python -m src.orchestrator ${ORCH_ARGS[*]:-}"

LOG_FILE="$LOG_DIR/run_${TIMESTAMP}.log"

# --- Launch in tmux (preferred) or nohup (fallback) ---
if command -v tmux &>/dev/null; then
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        echo -e "${YELLOW}Session '$SESSION_NAME' already exists.${NC}"
        echo -e "Attach with: bash scripts/run.sh --attach"
        echo -e "Or stop it first: bash scripts/run.sh --stop"
        exit 1
    fi

    print_header "LAUNCHING IN TMUX"
    echo "  Session:  $SESSION_NAME"
    echo "  Log file: $LOG_FILE"
    echo "  Command:  python -m src.orchestrator ${ORCH_ARGS[*]:-}"
    echo ""
    echo -e "${GREEN}The experiment will continue running even if your SSH/VPN${NC}"
    echo -e "${GREEN}connection drops. To monitor:${NC}"
    echo "    bash scripts/run.sh --attach    # attach to tmux session"
    echo "    bash scripts/run.sh --status    # print progress summary"
    echo "    bash scripts/monitor.sh         # live monitoring"
    echo "    tail -f $LOG_FILE             # raw log"

    tmux new-session -d -s "$SESSION_NAME" -x 200 -y 50 \
        "cd $PROJECT_ROOT && $COMMAND 2>&1 | tee $LOG_FILE; echo ''; echo 'Press Enter to close'; read"

    echo -e "\n${GREEN}Started. Attaching to session...${NC}"
    echo "(Press Ctrl+B then D to detach without stopping)"
    sleep 1
    exec tmux attach -t "$SESSION_NAME"

elif command -v nohup &>/dev/null; then
    print_header "LAUNCHING WITH NOHUP"
    echo "  Log file: $LOG_FILE"
    echo ""
    echo -e "${YELLOW}tmux not found. Using nohup (no attach capability).${NC}"

    nohup bash -c "cd $PROJECT_ROOT && $COMMAND" > "$LOG_FILE" 2>&1 &
    PID=$!
    echo "$PID" > "$PROJECT_ROOT/.orchestrator_pid"
    echo -e "${GREEN}Started with PID $PID${NC}"
    echo "Monitor with:"
    echo "    bash scripts/run.sh --status"
    echo "    tail -f $LOG_FILE"
    echo "    kill $PID  # to stop"

else
    echo -e "${RED}Neither tmux nor nohup found. Cannot run in background.${NC}"
    echo "Run directly:"
    echo "    source $VENV_DIR/bin/activate"
    echo "    python -m src.orchestrator $*"
    exit 1
fi
