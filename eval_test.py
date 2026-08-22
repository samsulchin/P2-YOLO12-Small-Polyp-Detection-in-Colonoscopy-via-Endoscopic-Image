import os
import csv
from pathlib import Path
from ultralytics import YOLO

# Configuration
SEED = 42
SPLIT = 'test'  # Ensure your data.yaml has a path for 'test: ...'
OUTPUT_CSV = "evaluation_summary.csv"

# List of 22 Experiments exactly as during Training
EXPERIMENTS = [
    # 1. YOLOv9 Series
    {"name": "yolov9_raw", "data": "dataset/data.yaml"},
    {"name": "yolov9_zerodce", "data": "dataset_zerodce/data.yaml"},
    {"name": "yolov9_zdce_sai", "data": "dataset_zdce_sai/data.yaml"},
    {"name": "yolov9_full", "data": "dataset_full/data.yaml"},

    # 2. YOLOv10 Series
    {"name": "yolov10_raw", "data": "dataset/data.yaml"},
    {"name": "yolov10_zerodce", "data": "dataset_zerodce/data.yaml"},
    {"name": "yolov10_zdce_sai", "data": "dataset_zdce_sai/data.yaml"},
    {"name": "yolov10_full", "data": "dataset_full/data.yaml"},

    # 3. YOLO11 Series
    {"name": "yolo11_raw", "data": "dataset/data.yaml"},
    {"name": "yolo11_zerodce", "data": "dataset_zerodce/data.yaml"},
    {"name": "yolo11_zdce_sai", "data": "dataset_zdce_sai/data.yaml"},
    {"name": "yolo11_full", "data": "dataset_full/data.yaml"},

    # 4. YOLO12 Series
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

def main():
    print("=" * 70)
    print("STARTING MODEL EVALUATION (TESTING PHASE)")
    print("=" * 70)

    # Prepare a CSV file to save the results summary
    with open(OUTPUT_CSV, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            "Model Name", "Precision", "Recall", "mAP50", "mAP50-95"
        ])

        for exp in EXPERIMENTS:
            run_name = f"{exp['name']}_seed{SEED}"
            weight_path = Path(f"runs/detect/{run_name}/weights/best.pt")

            print(f"\n[INFO] Evaluating model: {run_name}")
            
            if not weight_path.exists():
                print(f"[WARNING] Weight file not found: {weight_path}. Skipping this model.")
                continue

            try:
                # Load best model
                model = YOLO(str(weight_path))
                
                # Run evaluation on test data
                # save_json=True is MANDATORY to trigger pycocotools (calculates AP_small, AP_medium, AP_large)
                metrics = model.val(
                    data=exp['data'], 
                    split=SPLIT, 
                    save_json=True, 
                    device="0", 
                    batch=32
                )

                # Extract main metrics
                precision = metrics.results_dict['metrics/precision(B)']
                recall = metrics.results_dict['metrics/recall(B)']
                map50 = metrics.results_dict['metrics/mAP50(B)']
                map50_95 = metrics.results_dict['metrics/mAP50-95(B)']

                # Save to CSV
                writer.writerow([run_name, round(precision, 4), round(recall, 4), round(map50, 4), round(map50_95, 4)])
                
                print(f"[SUCCESS] {run_name} evaluation finished.")
                
            except Exception as e:
                print(f"[ERROR] Failed to evaluate {run_name}: {e}")

    print("\n" + "=" * 70)
    print(f"EVALUATION COMPLETED! Main summary results saved in: {OUTPUT_CSV}")
    print("For AP_small, AP_medium, and AP_large metrics, please check the terminal logs above (COCOeval table).")
    print("=" * 70)

if __name__ == "__main__":
    main()
