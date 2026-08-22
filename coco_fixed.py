#!/usr/bin/env python3
import os
import json
import csv
import cv2
from pathlib import Path
import subprocess

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
except ImportError:
    subprocess.run(["pip", "install", "pycocotools"], check=True)
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

EXPERIMENTS = [
    {"name": "yolo12_raw", "data_dir": "dataset", "val_dir": "runs/detect/val-23"},
    {"name": "yolo12_zerodce", "data_dir": "dataset_zerodce", "val_dir": "runs/detect/val-24"},
    {"name": "yolo12_sai", "data_dir": "dataset_sai", "val_dir": "runs/detect/val-25"},
    {"name": "yolo12_dwt", "data_dir": "dataset_dwt", "val_dir": "runs/detect/val-26"},
    {"name": "yolo12_zdce_sai", "data_dir": "dataset_zdce_sai", "val_dir": "runs/detect/val-27"},
    {"name": "yolo12_zdce_dwt", "data_dir": "dataset_zdce_dwt", "val_dir": "runs/detect/val-28"},
    {"name": "yolo12_sai_dwt", "data_dir": "dataset_sai_dwt", "val_dir": "runs/detect/val-29"},
    {"name": "yolo12_full", "data_dir": "dataset_full", "val_dir": "runs/detect/val-30"},
    {"name": "p2_yolo12_raw", "data_dir": "dataset", "val_dir": "runs/detect/val-31"},
    {"name": "p2_yolo12_full", "data_dir": "dataset_full", "val_dir": "runs/detect/val-32"}
]

OUTPUT_CSV = "coco_small_polyp_summary_FIXED.csv"

def evaluate_coco(name, data_dir, val_dir):
    img_dir = os.path.join(data_dir, "images", "test")
    lbl_dir = os.path.join(data_dir, "labels", "test")
    pred_file = os.path.join(val_dir, "predictions.json")
    
    if not os.path.exists(pred_file):
        return None, None, None

    images_json = []
    ann_json = []
    img_stem_to_id = {}
    ann_id = 1
    
    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))])
    
    for idx, img_name in enumerate(img_files):
        img_stem = Path(img_name).stem
        img_stem_to_id[img_stem] = idx  
        img_stem_to_id[img_name] = idx  # Anticipate if YOLO saves with .jpg extension
        
        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path)
        if img is None: continue
        h, w = img.shape[:2]
        
        images_json.append({"id": idx, "file_name": img_name, "width": w, "height": h})
        
        lbl_path = os.path.join(lbl_dir, f"{img_stem}.txt")
        if os.path.exists(lbl_path):
            with open(lbl_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        x_c, y_c, bw, bh = map(float, parts[1:5])
                        abs_w, abs_h = bw * w, bh * h
                        tl_x, tl_y = (x_c - bw / 2) * w, (y_c - bh / 2) * h
                        
                        ann_json.append({
                            "id": ann_id, "image_id": idx, "category_id": 1, # Category changed to 1
                            "bbox": [tl_x, tl_y, abs_w, abs_h], "area": abs_w * abs_h, "iscrowd": 0
                        })
                        ann_id += 1

    gt_dict = {"images": images_json, "annotations": ann_json, "categories": [{"id": 1, "name": "polyp"}]}
    gt_temp_path = f"temp_gt_{name}.json"
    with open(gt_temp_path, 'w') as f: json.dump(gt_dict, f)

    with open(pred_file, 'r') as f: preds = json.load(f)
    
    fixed_preds = []
    for p in preds:
        stem = str(p["image_id"])
        if stem.endswith('.jpg'): stem = stem[:-4] # Remove .jpg if exists
        
        if stem in img_stem_to_id:
            p["image_id"] = img_stem_to_id[stem]
            p["category_id"] = 1 # Adjust to Ground Truth category
            fixed_preds.append(p)
            
    pred_temp_path = f"temp_pred_{name}.json"
    with open(pred_temp_path, 'w') as f: json.dump(fixed_preds, f)

    try:
        cocoGt = COCO(gt_temp_path)
        cocoDt = cocoGt.loadRes(pred_temp_path)
        cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
        
        import contextlib, io
        with contextlib.redirect_stdout(io.StringIO()): 
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            
        stats = cocoEval.stats
        os.remove(gt_temp_path)
        os.remove(pred_temp_path)
        return stats[3], stats[4], stats[5] 
    except Exception as e:
        return None, None, None

def main():
    print("=" * 70)
    print("CALCULATING COCO METRICS (FIXED VERSION)")
    print("=" * 70)

    with open(OUTPUT_CSV, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Model Name", "AP_small", "AP_medium", "AP_large"])

        for exp in EXPERIMENTS:
            ap_s, ap_m, ap_l = evaluate_coco(exp['name'], exp['data_dir'], exp['val_dir'])
            if ap_s is not None:
                s_val = round(ap_s, 4) if ap_s != -1.0 else "N/A"
                m_val = round(ap_m, 4) if ap_m != -1.0 else "N/A"
                l_val = round(ap_l, 4) if ap_l != -1.0 else "N/A"
                writer.writerow([exp['name'], s_val, m_val, l_val])
                print(f"[{exp['name']}] AP_small: {s_val} | AP_medium: {m_val} | AP_large: {l_val}")

if __name__ == "__main__":
    main()
