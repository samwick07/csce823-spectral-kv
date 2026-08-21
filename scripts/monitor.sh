#!/usr/bin/env bash
# =============================================================================
# monitor.sh — Live monitoring dashboard for the spectral KV experiment.
#
# Shows: tmux session status, orchestrator state, GPU utilization,
# checkpoint progress, recent log tail, and disk usage.
#
# Usage:
#   bash scripts/monitor.sh              # one-shot snapshot
#   bash scripts/monitor.sh --watch      # auto-refresh every 30s
#   bash scripts/monitor.sh --watch 10   # refresh every 10s
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

SESSION_NAME="${SESSION_NAME:-spectral-kv}"
STATE_FILE="results/orchestrator_state.json"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

REFRESH="${2:-30}"
WATCH_MODE=false
if [[ "${1:-}" == "--watch" ]]; then
    WATCH_MODE=true
fi

show_dashboard() {
    clear
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║  SPECTRAL KV-CACHE EXPERIMENT — LIVE MONITOR              ║${NC}"
    echo -e "${CYAN}║  $(date '+%Y-%m-%d %H:%M:%S UTC')                            ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════╝${NC}"

    # --- tmux session ---
    echo -e "\n${BLUE}── SESSION ──${NC}"
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        PID=$(tmux list-panes -t "$SESSION_NAME" -F '#{pane_pid}' 2>/dev/null | head -1)
        echo -e "  Status: ${GREEN}RUNNING${NC} (tmux: $SESSION_NAME, PID: ${PID:-unknown})"
        echo -e "  Attach: bash scripts/run.sh --attach"
    elif [[ -f .orchestrator_pid ]]; then
        PID=$(cat .orchestrator_pid)
        if kill -0 "$PID" 2>/dev/null; then
            echo -e "  Status: ${GREEN}RUNNING${NC} (nohup, PID: $PID)"
        else
            echo -e "  Status: ${YELLOW}STOPPED${NC} (nohup PID $PID no longer alive)"
        fi
    else
        echo -e "  Status: ${YELLOW}NOT RUNNING${NC}"
        echo -e "  Start:  bash scripts/run.sh"
    fi

    # --- Orchestrator state ---
    echo -e "\n${BLUE}── PROGRESS ──${NC}"
    if [[ -f "$STATE_FILE" ]]; then
        python3 -c "
import json
from datetime import datetime

with open('$STATE_FILE') as f:
    s = json.load(f)

completed_train = s.get('completed_training', [])
completed_evals = s.get('completed_evals', [])
current = s.get('current_task', 'none')
crashes = s.get('crash_count', 0)
analysis = s.get('completed_analysis', False)
last_update = s.get('last_update', 'unknown')

# Count checkpoints on disk
import os
from pathlib import Path

ckpt_dir = Path('checkpoints')
trained_on_disk = []
if ckpt_dir.exists():
    for d in sorted(ckpt_dir.iterdir()):
        if d.is_dir():
            p2 = d / 'phase2_longalpaca' / 'final' / '.training_complete'
            if p2.exists():
                trained_on_disk.append(d.name)

# Count eval results on disk
eval_count = 0
raw_dir = Path('results/raw')
if raw_dir.exists():
    for config_dir in raw_dir.iterdir():
        for seed_dir in config_dir.iterdir():
            if (seed_dir / 'all_results.json').exists():
                eval_count += 1

print(f'  Current task:    {current}')
print(f'  Last update:     {last_update}')
print(f'  Crash count:     {crashes}')
print(f'  Training done:   {len(trained_on_disk)}/13 configs')
if trained_on_disk:
    print(f'    {\" \".join(trained_on_disk)}')
print(f'  Evals done:      {eval_count}/390')
print(f'  Analysis done:   {\"yes\" if analysis else \"no\"}')

# Progress bar
total_evals = 390  # 13 configs x 30 seeds (see src/stats/experiment_matrix.py)
pct = (eval_count / total_evals * 100) if total_evals > 0 else 0
bar_len = 40
filled = int(bar_len * eval_count / total_evals) if total_evals > 0 else 0
bar = '█' * filled + '░' * (bar_len - filled)
print(f'  Progress: [{bar}] {pct:.1f}%')
" 2>/dev/null || echo "  (state file exists but could not be parsed)"
    else
        echo -e "  No state file. Experiment has not been started."
    fi

    # --- GPU status ---
    echo -e "\n${BLUE}── GPU STATUS ──${NC}"
    if command -v nvidia-smi &>/dev/null; then
        nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu \
            --format=csv,noheader,nounits 2>/dev/null | while IFS=',' read -r idx name util mem_used mem_total temp; do
            idx=$(echo "$idx" | xargs)
            name=$(echo "$name" | xargs)
            util=$(echo "$util" | xargs)
            mem_used=$(echo "$mem_used" | xargs)
            mem_total=$(echo "$mem_total" | xargs)
            temp=$(echo "$temp" | xargs)
            printf "  GPU %s: %3s%% util | %6s/%6s MiB | %3s°C | %s\n" \
                "$idx" "$util" "$mem_used" "$mem_total" "$temp" "$name"
        done
    else
        echo "  nvidia-smi not available"
    fi

    # --- Disk usage ---
    echo -e "\n${BLUE}── DISK ──${NC}"
    DF_OUT=$(df -h "$PROJECT_ROOT" 2>/dev/null | tail -1)
    if [[ -n "$DF_OUT" ]]; then
        echo "  Workspace: $(echo "$DF_OUT" | awk '{print $3 " used / " $2 " total (" $5 ")"}')"
    fi
    CKPT_SIZE=$(du -sh checkpoints/ 2>/dev/null | cut -f1 || echo "0")
    RESULTS_SIZE=$(du -sh results/ 2>/dev/null | cut -f1 || echo "0")
    HF_CACHE_SIZE=$(du -sh ~/.cache/huggingface/ 2>/dev/null | cut -f1 || echo "0")
    echo "  checkpoints/: $CKPT_SIZE"
    echo "  results/:     $RESULTS_SIZE"
    echo "  HF cache:     $HF_CACHE_SIZE"

    # --- Recent log ---
    echo -e "\n${BLUE}── RECENT LOG (last 15 lines) ──${NC}"
    LATEST_LOG=$(ls -t logs/orchestrator_*.log 2>/dev/null | head -1)
    if [[ -n "$LATEST_LOG" ]]; then
        tail -15 "$LATEST_LOG" 2>/dev/null | sed 's/^/  /'
    else
        echo "  (no orchestrator logs found)"
    fi

    echo ""
    if $WATCH_MODE; then
        echo -e "${CYAN}Refreshing in ${REFRESH}s... (Ctrl+C to exit)${NC}"
    else
        echo "Live mode: bash scripts/monitor.sh --watch"
    fi
}

# --- Main loop ---
if $WATCH_MODE; then
    while true; do
        show_dashboard
        sleep "$REFRESH"
    done
else
    show_dashboard
fi
