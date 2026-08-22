#!/usr/bin/env python3
"""
06b_train_seeds.py
==================
Special script for STATISTICAL VALIDATION (Section 3.8.4).
Automatically trains Baseline vs Proposed Method 
on the remaining Seeds (43 and 44).
"""

import subprocess
from pathlib import Path

# Main training parameter configuration (Fast Mode)
EPOCHS = 300
IMG_SIZE = 640
BATCH_SIZE = 32        
WORKERS = 16           
DEVICE = "0"           
OPTIMIZER = "SGD"      
PATIENCE = 20

# Speed & Stability Parameters
AMP = "True"           
DFL = 0.0              
MOSAIC = 0.0           
MIXUP = 0.0            
LR0 = 0.005            
WARMUP_EPOCHS = 10.0   

# SEEDS TO RUN (42 is done, so we run 43 and 44)
SEEDS_TO_RUN = [43, 44]

# ONLY FOCUS ON MODELS COMPARED IN THE PAPER
# If there are other models to be averaged, add them here
MODELS_TO_VALIDATE = [
    {"name": "yolo12_raw", "data": "dataset/data.yaml", "model": "yolo12s.pt"},
    {"name": "p2_yolo12_full", "data": "dataset_full/data.yaml", "model": "p2_yolo12s.yaml"}
]

def main():
    print("=" * 70)
    print("AUTOMATED TRAINING SCRIPT — STATISTICAL VALIDATION (MULTIPLE SEEDS)")
    print(f"Seeds in queue: {SEEDS_TO_RUN}")
    print("=" * 70)

    for seed in SEEDS_TO_RUN:
        for exp in MODELS_TO_VALIDATE:
            run_name = f"{exp['name']}_seed{seed}"
            print(f"\n[START] Training: {run_name}")
            print(f"        Seed: {seed} | Data: {exp['data']} | Model: {exp['model']}")
            
            cmd = [
                str(Path("venv/bin/yolo")), "train",
                f"model={exp['model']}",
                f"data={exp['data']}",
                f"epochs={EPOCHS}",
                f"imgsz={IMG_SIZE}",
                f"batch={BATCH_SIZE}",
                f"workers={WORKERS}",
                f"device={DEVICE}",
                f"optimizer={OPTIMIZER}",
                f"patience={PATIENCE}",
                f"seed={seed}",
                f"amp={AMP}",
                f"dfl={DFL}",
                f"mosaic={MOSAIC}",
                f"mixup={MIXUP}",
                f"lr0={LR0}",
                f"warmup_epochs={WARMUP_EPOCHS}",
                f"name={run_name}"
            ]
            
            try:
                subprocess.run(cmd, check=True)
                print(f"[SUCCESS] Finished: {run_name}")
            except subprocess.CalledProcessError as e:
                print(f"[ERROR] FAILED ON: {run_name}")
                print(f"Error details: {e}")
                print(">>> Continuing to the next queue...\n")
                continue

    print("\n" + "=" * 70)
    print("ALL SEED QUEUES (STATISTICAL VALIDATION) HAVE BEEN COMPLETED!")
    print("=" * 70)

if __name__ == "__main__":
    main()
