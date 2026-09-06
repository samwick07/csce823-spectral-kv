#!/usr/bin/env python3
"""Pull full training-loss histories from WandB for all 13 configs x 2 phases.

Writes normalized JSON: {"C00_phase1": [{"step": <trainer step>, "loss": v,
"epoch": e}, ...], ...}

DATA INTEGRITY NOTE: trainer step = row["train/global_step"] when present
(C02-C12), else row["_step"] (C00/C01 log trainer step natively). Using
_step unconditionally truncates C02-C12 to ~100 WandB rows each.

Requires: WANDB_API_KEY env var (entity samwick07-afit,
project csce823-spectral-kv).
"""
import json
import os
import sys
import time

import wandb

ENTITY = "samwick07-afit"
PROJECT = "csce823-spectral-kv"
OUT = sys.argv[1] if len(sys.argv) > 1 else "wandb_training_loss.json"

api = wandb.Api()
runs = {r.name: r for r in api.runs(f"{ENTITY}/{PROJECT}")}

data = {}
missing = []
for cfg in [f"C{i:02d}" for i in range(13)]:
    for phase, rname in [
        ("phase1", f"{cfg}_phase1_redpajama"),
        ("phase2", f"{cfg}_phase2_longalpaca"),
    ]:
        if rname not in runs:
            missing.append(rname)
            continue
        r = runs[rname]
        hist = list(r.history(samples=100000, pandas=False))  # full, unsampled
        pts = []
        for row in hist:
            loss = row.get("train/loss")
            if not isinstance(loss, (int, float)):
                continue
            x = row.get("train/global_step")
            if not isinstance(x, (int, float)):
                x = row.get("_step")
            pts.append({"step": x, "loss": loss, "epoch": row.get("train/epoch")})
        pts.sort(key=lambda p: p["step"])
        key = f"{cfg}_{phase}"
        data[key] = pts
        print(f"{key:15} n={len(pts):3d}  steps {pts[0]['step']}..{pts[-1]['step']}  "
              f"final={pts[-1]['loss']:.4f}")
        time.sleep(0.1)

with open(OUT, "w") as f:
    json.dump(data, f)
print(f"\nSaved {len(data)}/26 curves to {OUT}")
if missing:
    print(f"MISSING RUNS: {missing}", file=sys.stderr)
    sys.exit(1)
