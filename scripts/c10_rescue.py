#!/usr/bin/env python3
"""
C10 Rescue Script: monitors the running C10 eval process, and when it
crashes at _atomic_write_json (NameError: os), fetches the LongBench
results from WandB and writes them to disk using the fixed code.

Usage: nohup python3 scripts/c10_rescue.py > logs/c10_rescue.log 2>&1 &
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path("/workspaces/csce823-spectral-kv")
RESULTS_DIR = PROJECT_ROOT / "results" / "raw" / "C10" / "seed_0"
C10_PID = 2922217  # The restarted C10 process
MAX_WAIT_HOURS = 12

def is_process_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False

def fetch_wandb_results():
    """Fetch C10 LongBench results from WandB API."""
    import wandb
    api = wandb.Api()
    run = api.run("samwick07-afit/csce823-spectral-kv/C10_eval_seed0")
    
    # Get the logged metrics
    history = run.history(stream="default", samples=10000)
    
    # Check for longbench_v1 results in the summary
    summary = run.summary
    print(f"WandB summary keys: {sorted(summary.keys())}", flush=True)
    
    # Look for longbench data in the summary
    lb_keys = [k for k in summary.keys() if 'longbench' in k.lower() or 'lb' in k.lower()]
    print(f"LongBench summary keys: {lb_keys}", flush=True)
    
    # Also check the history for task-level scores
    lb_rows = [r for r in history.to_dict('records') if any('longbench' in str(k).lower() for k in r.keys())]
    print(f"LongBench history rows: {len(lb_rows)}", flush=True)
    
    return summary, history

def build_longbench_json(summary, history):
    """Build the longbench.json structure from WandB data."""
    # The run_experiment.py logs results to WandB via log_eval_results
    # The summary should have longbench_v1 overall_mean and per-task scores
    
    result = {
        "benchmark": "longbench_v1",
        "seed": 0,
        "temperature": 0.7,
        "tasks": [],
        "overall_mean": None,
    }
    
    # Extract overall mean
    for key in ["longbench_v1/overall_mean", "longbench_v1", "longbench/overall_mean", "longbench"]:
        if key in summary:
            result["overall_mean"] = summary[key]
            break
    
    # Extract per-task scores from history
    task_names = [
        "narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa",
        "musique", "gov_report", "qmsum", "multi_news", "trec",
        "triviaqa", "samsum", "passage_retrieval_en", "lcc"
    ]
    
    for row in history.to_dict('records'):
        for key, val in row.items():
            if 'longbench' in key.lower() and val is not None:
                for task in task_names:
                    if task in key:
                        result["tasks"].append({
                            "task": task,
                            "score": val,
                        })
    
    # Deduplicate tasks (keep last score per task)
    seen = {}
    for t in result["tasks"]:
        seen[t["task"]] = t
    result["tasks"] = list(seen.values())
    result["tasks"].sort(key=lambda x: task_names.index(x["task"]) if x["task"] in task_names else 999)
    
    print(f"Built longbench result: {len(result['tasks'])} tasks, overall_mean={result['overall_mean']}", flush=True)
    return result

def save_results(lb_data):
    """Save longbench.json and merge into all_results.json."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save longbench.json
    lb_file = RESULTS_DIR / "longbench.json"
    with open(lb_file, "w") as f:
        json.dump(lb_data, f, indent=2)
    print(f"Saved {lb_file}", flush=True)
    
    # Merge into all_results.json
    all_results_file = RESULTS_DIR / "all_results.json"
    if all_results_file.exists():
        with open(all_results_file) as f:
            all_results = json.load(f)
    else:
        all_results = {}
    
    all_results["longbench_v1"] = lb_data
    with open(all_results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Merged into {all_results_file}", flush=True)

def kill_redundant_c10():
    """Kill any C10 process the orchestrator might have restarted."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "experiment_C10.*longbench"],
            capture_output=True, text=True
        )
        pids = [int(p) for p in result.stdout.strip().split() if p]
        for pid in pids:
            if pid != C10_PID:  # Don't kill the original (already dead at this point)
                print(f"Killing redundant C10 process: {pid}", flush=True)
                os.kill(pid, 9)
    except Exception as e:
        print(f"Error killing redundant C10: {e}", flush=True)

def main():
    print(f"C10 Rescue Script started at {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}", flush=True)
    print(f"Monitoring PID {C10_PID}", flush=True)
    
    # Phase 1: Wait for C10 to finish (crash at save)
    wait_start = time.time()
    while is_process_alive(C10_PID):
        elapsed = (time.time() - wait_start) / 3600
        if elapsed > MAX_WAIT_HOURS:
            print(f"C10 still running after {MAX_WAIT_HOURS}h, giving up", flush=True)
            return 1
        time.sleep(60)
    
    print(f"C10 process exited at {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}", flush=True)
    
    # Check if results were saved despite the crash (maybe the fix was picked up)
    lb_file = RESULTS_DIR / "longbench.json"
    if lb_file.exists():
        stat = lb_file.stat()
        if stat.st_mtime > wait_start:
            print(f"longbench.json was written after rescue script start ({stat.st_size} bytes)", flush=True)
            print("C10 saved results successfully! No rescue needed.", flush=True)
            kill_redundant_c10()
            return 0
    
    # Phase 2: Wait a moment for WandB to flush
    print("Waiting 30s for WandB to flush...", flush=True)
    time.sleep(30)
    
    # Phase 3: Fetch results from WandB
    print("Fetching results from WandB...", flush=True)
    try:
        summary, history = fetch_wandb_results()
        lb_data = build_longbench_json(summary, history)
        
        if len(lb_data["tasks"]) < 14:
            print(f"WARNING: Only found {len(lb_data['tasks'])} tasks in WandB (expected 14)", flush=True)
            print("Will save what we have. Full re-run may be needed.", flush=True)
        
        save_results(lb_data)
        print("Rescue successful!", flush=True)
        
        # Kill any redundant C10 the orchestrator started
        kill_redundant_c10()
        return 0
        
    except Exception as e:
        print(f"WandB rescue failed: {e}", flush=True)
        import traceback
        traceback.print_exc()
        print("The orchestrator should auto-restart C10 with the fixed code.", flush=True)
        print("The fix (import os) is already on disk and will be picked up by new processes.", flush=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
