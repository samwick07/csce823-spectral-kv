#!/bin/bash
# Auto-restart orchestrator after pod/workspace restart.
# Called by cron @reboot. Waits for GPU availability, then relaunches.
set -e
cd /workspaces/csce823-spectral-kv

# Wait for GPUs (up to 5 min)
for i in $(seq 1 30); do
    if nvidia-smi &>/dev/null; then
        echo "[$(date)] GPUs available, proceeding with relaunch"
        break
    fi
    echo "[$(date)] Waiting for GPUs... ($i/30)"
    sleep 10
done

if [ -f /workspaces/.env.spectral ]; then
    source /workspaces/.env.spectral
fi

bash /workspaces/csce823-spectral-kv/scripts/relaunch.sh
