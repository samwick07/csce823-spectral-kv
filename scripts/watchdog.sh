#!/bin/bash
# Lightweight watchdog: restarts orchestrator if it dies.
POLL_INTERVAL=300  # 5 minutes
PID_FILE="/workspaces/csce823-spectral-kv/.orchestrator_pid"
PROJECT_ROOT="/workspaces/csce823-spectral-kv"

while true; do
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            sleep $POLL_INTERVAL
            continue
        fi
    fi
    
    if [ -f "$PROJECT_ROOT/results/orchestrator_state.json" ]; then
        COMPLETED=$(python3 -c "
import json
with open('$PROJECT_ROOT/results/orchestrator_state.json') as f:
    s = json.load(f)
training = len(s.get('completed_training', []))
analysis = s.get('completed_analysis', False)
exfil = s.get('completed_exfil', False)
done = training >= 13 and analysis and exfil
print('false' if done else 'true')
" 2>/dev/null)
        
        if [ "$COMPLETED" = "true" ]; then
            echo "[$(date)] Orchestrator not running but work remains. Restarting..."
            cd "$PROJECT_ROOT"
            source /workspaces/.env.spectral 2>/dev/null
            bash scripts/relaunch.sh >> "$PROJECT_ROOT/logs/watchdog_relaunch.log" 2>&1
        fi
    fi
    
    sleep $POLL_INTERVAL
done
