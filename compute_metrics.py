#!/usr/bin/env python3
import os
import csv
from ultralytics import YOLO

# We input Parameters and GFLOPs manually based on your previous terminal logs
# to avoid bugs in the latest Ultralytics library version.
MODELS_TO_TEST = [
    {
        "name": "YOLOv9s", 
        "weight": "runs/detect/yolov9_raw_seed42/weights/best.pt", 
        "data": "dataset/data.yaml",
        "params": 7.17, "gflops": 26.7
    },
    {
        "name": "YOLOv10s", 
        "weight": "runs/detect/yolov10_raw_seed42/weights/best.pt", 
        "data": "dataset/data.yaml",
        "params": 7.22, "gflops": 21.5
    },
    {
        "name": "YOLO11s", 
        "weight": "runs/detect/yolo11_raw_seed42/weights/best.pt", 
        "data": "dataset/data.yaml",
        "params": 9.41, "gflops": 21.4
    },
    {
        "name": "YOLO12s (Baseline)", 
        "weight": "runs/detect/yolo12_raw_seed42/weights/best.pt", 
        "data": "dataset/data.yaml",
        "params": 9.23, "gflops": 23.2
    },
    {
        "name": "P2-YOLO12s (Proposed)", 
        "weight": "runs/detect/p2_yolo12_full_seed42/weights/best.pt", 
        "data": "dataset_full/data.yaml",
        "params": 9.36, "gflops": 29.9
    }
]

OUTPUT_CSV = "computational_metrics_summary.csv"

def main():
    print("=" * 75)
    print("CALCULATING COMPUTATIONAL METRICS (FIXED VERSION - INFERENCE ONLY)")
    print("=" * 75)

    with open(OUTPUT_CSV, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Model", "Parameters (Millions)", "Computational Cost (GFLOPs)", "Inference Time (ms/frame)"])

        for exp in MODELS_TO_TEST:
            print(f"\n[INFO] Measuring inference speed for: {exp['name']}...")
            
            if not os.path.exists(exp['weight']):
                print(f"  -> [SKIP] Weight file not found: {exp['weight']}")
                writer.writerow([exp['name'], exp['params'], exp['gflops'], "N/A"])
                continue

            try:
                # 1. Load Model
                model = YOLO(exp['weight'])
                
                # 2. Print Parameters & GFLOPs (Static)
                print(f"  -> Total Parameter : {exp['params']} M")
                print(f"  -> GFLOPs          : {exp['gflops']}")
                
                # 3. Calculate Inference Time on RTX 5090
                # Performing silent testing simulation to record inference time (ms/frame)
                metrics = model.val(data=exp['data'], split="test", verbose=False, device="0")
                
                inference_ms = round(metrics.speed['inference'], 2)
                print(f"  -> Inference Time  : {inference_ms} ms/frame")

                # 4. Save to CSV
                writer.writerow([exp['name'], exp['params'], exp['gflops'], inference_ms])

            except Exception as e:
                print(f"  -> [ERROR] An error occurred while processing {exp['name']}: {e}")

    print("\n" + "=" * 75)
    print(f"DONE! Computational and speed data successfully exported to: {OUTPUT_CSV}")
    print("=" * 75)

if __name__ == "__main__":
    main()
