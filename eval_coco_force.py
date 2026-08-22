#!/usr/bin/env python3
import subprocess
import os

# Only focus on YOLO12 and P2-YOLO12 families
EXPERIMENTS = [
    # 4. YOLO12 Series (Ablation)
    {"name": "yolo12_raw", "data": "dataset/data.yaml"},
    {"name": "yolo12_zerodce", "data": "dataset_zerodce/data.yaml"},
    {"name": "yolo12_sai", "data": "dataset_sai/data.yaml"},
    {"name": "yolo12_dwt", "data": "dataset_dwt/data.yaml"},
    {"name": "yolo12_zdce_sai", "data": "dataset_zdce_sai/data.yaml"},
    {"name": "yolo12_zdce_dwt", "data": "dataset_zdce_dwt/data.yaml"},
    {"name": "yolo12_sai_dwt", "data": "dataset_sai_dwt/data.yaml"},
    {"name": "yolo12_full", "data": "dataset_full/data.yaml"},

    # 5. Proposed Method (P2-YOLO12)
    {"name": "p2_yolo12_raw", "data": "dataset/data.yaml"},
    {"name": "p2_yolo12_full", "data": "dataset_full/data.yaml"},
]

SEED = 42

def main():
    print("=" * 70)
    print("STARTING COCO EVALUATION SPECIFICALLY FOR YOLO12 & P2-YOLO12")
    print("Focus on monitoring the line: 'Average Precision  (AP) @[ IoU=0.50:0.95 | area= small ]'")
    print("=" * 70)

    for exp in EXPERIMENTS:
        run_name = f"{exp['name']}_seed{SEED}"
        weight_path = f"runs/detect/{run_name}/weights/best.pt"
        
        if not os.path.exists(weight_path):
            print(f"\n[SKIP] Weights not found: {weight_path}")
            continue

        print(f"\n\n{'='*70}")
        print(f"👉 DISPLAYING COCO METRICS FOR: {run_name}")
        print(f"{'='*70}")
        
        # Using subprocess to call YOLO CLI to force the COCO table to display
        cmd = f"venv/bin/yolo val model='{weight_path}' data='{exp['data']}' split=test save_json=True"
        
        try:
            subprocess.run(cmd, shell=True, check=True)
        except subprocess.CalledProcessError:
            print(f"[ERROR] Failed to process {run_name}")

if __name__ == "__main__":
    main()
