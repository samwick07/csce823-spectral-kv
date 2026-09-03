#!/usr/bin/env python3
"""Pre-download eval datasets to avoid 429 rate limits during parallel eval."""
import os
os.environ.setdefault("HF_HOME", "/workspaces/.cache/huggingface")
from datasets import load_dataset

print("Pre-downloading hoskinson-center/proof-pile (test split)...")
ds = load_dataset("hoskinson-center/proof-pile", split="test", trust_remote_code=True)
print(f"  Cached: {len(ds)} rows")
print("Pre-download complete.")
