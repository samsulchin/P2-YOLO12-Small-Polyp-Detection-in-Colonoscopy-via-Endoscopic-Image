#!/usr/bin/env python3
import os
import csv
import statistics
from pathlib import Path
from ultralytics import YOLO

# Configuration
SEEDS = [42, 43, 44]
OUTPUT_CSV = "statistical_validation_summary.csv"

# Models to validate (Baseline vs Proposed)
MODELS = [
    {"base_name": "yolo12_raw", "data": "dataset/data.yaml"},
    {"base_name": "p2_yolo12_full", "data": "dataset_full/data.yaml"}
]

def main():
    print("=" * 75)
    print("SUMMARIZING STATISTICAL VALIDATION (MEAN ± SD) FOR ALL SEEDS")
    print("=" * 75)

    # Prepare CSV file
    with open(OUTPUT_CSV, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write Header
        writer.writerow(["Model", "Seed", "Precision", "Recall", "mAP50", "mAP50-95"])

        # Loop for each model (Baseline & Proposed)
        for model_info in MODELS:
            base_name = model_info["base_name"]
            data_yaml = model_info["data"]
            
            print(f"\n[INFO] Processing model group: {base_name.upper()}")
            
            # Storage for 3 seed results to calculate the average later
            metrics_history = {
                "precision": [],
                "recall": [],
                "map50": [],
                "map50_95": []
            }

            for seed in SEEDS:
                run_name = f"{base_name}_seed{seed}"
                weight_path = Path(f"runs/detect/{run_name}/weights/best.pt")

                if not weight_path.exists():
                    print(f"  -> [WARNING] Weights not found: {weight_path}")
                    writer.writerow([base_name, seed, "N/A", "N/A", "N/A", "N/A"])
                    continue

                print(f"  -> Evaluating {run_name} on test data...")
                try:
                    # Load model and run silent validation (verbose=False to keep terminal clean)
                    model = YOLO(str(weight_path))
                    metrics = model.val(data=data_yaml, split="test", verbose=False, device="0")

                    # Extract metrics
                    p = metrics.results_dict['metrics/precision(B)']
                    r = metrics.results_dict['metrics/recall(B)']
                    m50 = metrics.results_dict['metrics/mAP50(B)']
                    m95 = metrics.results_dict['metrics/mAP50-95(B)']

                    # Save to memory to calculate SD
                    metrics_history["precision"].append(p)
                    metrics_history["recall"].append(r)
                    metrics_history["map50"].append(m50)
                    metrics_history["map50_95"].append(m95)

                    # Write individual results per seed to CSV
                    writer.writerow([base_name, seed, round(p, 4), round(r, 4), round(m50, 4), round(m95, 4)])

                except Exception as e:
                    print(f"  -> [ERROR] Failed to evaluate {run_name}: {e}")

            # After 3 seeds are evaluated, calculate Mean and Standard Deviation (SD)
            if len(metrics_history["map50"]) == 3:
                print(f"  -> Calculating Mean ± SD for {base_name}...")
                
                def calc_mean_sd(data_list):
                    mean_val = statistics.mean(data_list)
                    sd_val = statistics.stdev(data_list)
                    return f"{mean_val:.4f} ± {sd_val:.4f}"

                mean_sd_p = calc_mean_sd(metrics_history["precision"])
                mean_sd_r = calc_mean_sd(metrics_history["recall"])
                mean_sd_m50 = calc_mean_sd(metrics_history["map50"])
                mean_sd_m95 = calc_mean_sd(metrics_history["map50_95"])

                # Write conclusion row (Mean ± SD) to CSV
                writer.writerow([f"{base_name} (Mean ± SD)", "ALL", mean_sd_p, mean_sd_r, mean_sd_m50, mean_sd_m95])
                # Add an empty row as a separator between models
                writer.writerow([]) 

    print("\n" + "=" * 75)
    print(f"DONE! Summary results with Mean ± SD saved in: {OUTPUT_CSV}")
    print("This format is 100% ready to be copy-pasted to your Q1 Journal manuscript!")
    print("=" * 75)

if __name__ == "__main__":
    main()
