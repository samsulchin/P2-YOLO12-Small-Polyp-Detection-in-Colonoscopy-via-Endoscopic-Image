#!/usr/bin/env python3
"""
06_train_all.py
===============
Automated training script (Phase 1: Single Seed) 
According to Table 4 order (YOLOv9 up to P2-YOLO12 Proposed Method)
Utilizing maximum hardware (RTX 5090).
Optimized for SPEED: AMP enabled, Batch Size increased.
"""

import subprocess
from pathlib import Path

# Main training parameter configuration
EPOCHS = 300
IMG_SIZE = 640
BATCH_SIZE = 32        # INCREASED to 32 because AMP is active (VRAM is more efficient)
WORKERS = 16           
DEVICE = "0"           
OPTIMIZER = "SGD"      # Kept using SGD for stability and to prevent gradient explosion
PATIENCE = 20
SEED = 42              

# Speed & Stability Parameters (FAST MODE)
AMP = "True"           # RE-ENABLED so RTX 5090 can run fast using FP16
DFL = 0.0              # Kept off so small box loss doesn't explode
MOSAIC = 0.0           # Kept off so small polyps do not shrink
MIXUP = 0.0            
LR0 = 0.005            # LOWERED SLIGHTLY (from 0.01) as a shield because AMP is on
WARMUP_EPOCHS = 10.0   # INCREASED (from 5 to 10) so the model adapts slowly

# List of 22 Experiments According to Table 4
EXPERIMENTS = [
    # ---------------------------------------------------------
    # 1. YOLOv9 Series
    # ---------------------------------------------------------
    {"name": "yolov9_raw", "data": "dataset/data.yaml", "model": "yolov9s.pt"},
    {"name": "yolov9_zerodce", "data": "dataset_zerodce/data.yaml", "model": "yolov9s.pt"},
    {"name": "yolov9_zdce_sai", "data": "dataset_zdce_sai/data.yaml", "model": "yolov9s.pt"},
    {"name": "yolov9_full", "data": "dataset_full/data.yaml", "model": "yolov9s.pt"},

    # ---------------------------------------------------------
    # 2. YOLOv10 Series
    # ---------------------------------------------------------
    {"name": "yolov10_raw", "data": "dataset/data.yaml", "model": "yolov10s.pt"},
    {"name": "yolov10_zerodce", "data": "dataset_zerodce/data.yaml", "model": "yolov10s.pt"},
    {"name": "yolov10_zdce_sai", "data": "dataset_zdce_sai/data.yaml", "model": "yolov10s.pt"},
    {"name": "yolov10_full", "data": "dataset_full/data.yaml", "model": "yolov10s.pt"},

    # ---------------------------------------------------------
    # 3. YOLO11 Series
    # ---------------------------------------------------------
    {"name": "yolo11_raw", "data": "dataset/data.yaml", "model": "yolo11s.pt"},
    {"name": "yolo11_zerodce", "data": "dataset_zerodce/data.yaml", "model": "yolo11s.pt"},
    {"name": "yolo11_zdce_sai", "data": "dataset_zdce_sai/data.yaml", "model": "yolo11s.pt"},
    {"name": "yolo11_full", "data": "dataset_full/data.yaml", "model": "yolo11s.pt"},

    # ---------------------------------------------------------
    # 4. YOLO12 Series (Factorial Ablation)
    # ---------------------------------------------------------
    {"name": "yolo12_raw", "data": "dataset/data.yaml", "model": "yolo12s.pt"},
    {"name": "yolo12_zerodce", "data": "dataset_zerodce/data.yaml", "model": "yolo12s.pt"},
    {"name": "yolo12_sai", "data": "dataset_sai/data.yaml", "model": "yolo12s.pt"},
    {"name": "yolo12_dwt", "data": "dataset_dwt/data.yaml", "model": "yolo12s.pt"},
    {"name": "yolo12_zdce_sai", "data": "dataset_zdce_sai/data.yaml", "model": "yolo12s.pt"},
    {"name": "yolo12_zdce_dwt", "data": "dataset_zdce_dwt/data.yaml", "model": "yolo12s.pt"},
    {"name": "yolo12_sai_dwt", "data": "dataset_sai_dwt/data.yaml", "model": "yolo12s.pt"},
    {"name": "yolo12_full", "data": "dataset_full/data.yaml", "model": "yolo12s.pt"},

    # ---------------------------------------------------------
    # 5. Proposed Method (P2-YOLO12)
    # ---------------------------------------------------------
    {"name": "p2_yolo12_raw", "data": "dataset/data.yaml", "model": "p2_yolo12s.yaml"},
    {"name": "p2_yolo12_full", "data": "dataset_full/data.yaml", "model": "p2_yolo12s.yaml"},
]

def main():
    print("=" * 70)
    print("AUTOMATED TRAINING SCRIPT — PHASE 1 (FAST MODE)")
    print(f"Device: GPU {DEVICE} | Optimizer: {OPTIMIZER} | Batch: {BATCH_SIZE}")
    print(f"Speed Increased: amp={AMP}, lr0={LR0}, warmup={WARMUP_EPOCHS}")
    print("=" * 70)

    for exp in EXPERIMENTS:
        run_name = f"{exp['name']}_seed{SEED}"
        print(f"\n[START] Training: {run_name}")
        print(f"        Data: {exp['data']} | Model: {exp['model']}")
        
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
            f"seed={SEED}",
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
    print("ALL PHASE 1 TRAINING QUEUES HAVE BEEN COMPLETED!")
    print("=" * 70)

if __name__ == "__main__":
    main()
