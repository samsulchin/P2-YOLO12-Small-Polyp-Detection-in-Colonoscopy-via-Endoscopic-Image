#!/usr/bin/env python3
import os
import csv
from ultralytics import YOLO

# List of YOLO12 & P2-YOLO12 Family Experiments
EXPERIMENTS = [
    {"name": "yolo12_raw", "data_dir": "dataset", "weight": "runs/detect/yolo12_raw_seed42/weights/best.pt"},
    {"name": "yolo12_zerodce", "data_dir": "dataset_zerodce", "weight": "runs/detect/yolo12_zerodce_seed42/weights/best.pt"},
    {"name": "yolo12_sai", "data_dir": "dataset_sai", "weight": "runs/detect/yolo12_sai_seed42/weights/best.pt"},
    {"name": "yolo12_dwt", "data_dir": "dataset_dwt", "weight": "runs/detect/yolo12_dwt_seed42/weights/best.pt"},
    {"name": "yolo12_zdce_sai", "data_dir": "dataset_zdce_sai", "weight": "runs/detect/yolo12_zdce_sai_seed42/weights/best.pt"},
    {"name": "yolo12_zdce_dwt", "data_dir": "dataset_zdce_dwt", "weight": "runs/detect/yolo12_zdce_dwt_seed42/weights/best.pt"},
    {"name": "yolo12_sai_dwt", "data_dir": "dataset_sai_dwt", "weight": "runs/detect/yolo12_sai_dwt_seed42/weights/best.pt"},
    {"name": "yolo12_full", "data_dir": "dataset_full", "weight": "runs/detect/yolo12_full_seed42/weights/best.pt"},
    {"name": "p2_yolo12_raw", "data_dir": "dataset", "weight": "runs/detect/p2_yolo12_raw_seed42/weights/best.pt"},
    {"name": "p2_yolo12_full", "data_dir": "dataset_full", "weight": "runs/detect/p2_yolo12_full_seed42/weights/best.pt"}
]

OUTPUT_CSV = "clinical_fp_summary.csv"

def get_negative_image_paths(data_dir):
    """Searching for healthy images (empty/missing labels) in a specific folder."""
    img_dir = os.path.join(data_dir, "images", "test")
    lbl_dir = os.path.join(data_dir, "labels", "test")
    
    negative_images = []
    
    # Check if image folder exists
    if not os.path.exists(img_dir):
        return []

    for img_name in os.listdir(img_dir):
        if not img_name.endswith(('.jpg', '.png')): continue
        
        stem = os.path.splitext(img_name)[0]
        lbl_path = os.path.join(lbl_dir, f"{stem}.txt")
        
        is_negative = False
        if not os.path.exists(lbl_path):
            is_negative = True
        else:
            with open(lbl_path, 'r') as f:
                content = f.read().strip()
                if len(content) == 0:
                    is_negative = True
                    
        if is_negative:
            negative_images.append(os.path.join(img_dir, img_name))
            
    return negative_images

def main():
    print("=" * 70)
    print("CALCULATING CLINICAL FALSE-ALARM (FP/frame) FOR ALL MODELS")
    print("=" * 70)

    with open(OUTPUT_CSV, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Model Name", "Total Negatives (Frames)", "Total False Positives (Boxes)", "FP/frame"])

        for exp in EXPERIMENTS:
            print(f"\n[INFO] Processing model: {exp['name']}...")
            
            # Check weight existence
            if not os.path.exists(exp['weight']):
                print(f"  -> [SKIP] Weights not found: {exp['weight']}")
                writer.writerow([exp['name'], "N/A", "N/A", "N/A"])
                continue
                
            # Get negative image data according to this model's preprocessing folder
            negative_images = get_negative_image_paths(exp['data_dir'])
            total_negatives = len(negative_images)
            
            if total_negatives == 0:
                print(f"  -> [SKIP] No negative images in {exp['data_dir']}")
                writer.writerow([exp['name'], 0, 0, "N/A"])
                continue
                
            print(f"  -> Evaluating {total_negatives} negative frames...")
            
            # Load model and perform inference
            model = YOLO(exp['weight'])
            
            total_false_positives = 0
            # conf=0.25 is the default YOLO standard for mAP validation
            results = model.predict(source=negative_images, verbose=False, conf=0.25)
            
            for res in results:
                total_false_positives += len(res.boxes)
                
            fp_per_frame = total_false_positives / total_negatives
            
            print(f"  -> Result: {total_false_positives} FP Boxes | FP/frame: {fp_per_frame:.4f}")
            writer.writerow([exp['name'], total_negatives, total_false_positives, round(fp_per_frame, 4)])

    print("\n" + "=" * 70)
    print(f"DONE! Results have been summarized in the CSV file: {OUTPUT_CSV}")
    print("Note: The SMALLER the FP/frame value, the better the model!")
    print("=" * 70)

if __name__ == "__main__":
    main()
