
import os, csv, json, shutil
from collections import defaultdict
import numpy as np
from PIL import Image
from scipy import ndimage
 
# ============ EDIT THIS ============
AUDIT_DIR = r"./audit_out"
OUT_DIR = r"./dataset"          # YOLO+COCO structure written here
TARGET = 640
MIN_COMPONENT_AREA = 25
LINK_MODE = "symlink"          # "symlink" (save space) or "copy"
COCO_SPLITS = ["val", "test", "ood_test"]   # splits for which COCO GT is generated
# ==================================
 
IMG_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
 
 
def read_mask_binary(path):
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., :3].max(axis=2)
    return arr > 127
 
 
def polyps_from_mask(mask_bool):
    labeled, n = ndimage.label(mask_bool)
    out = []
    for i, sl in enumerate(ndimage.find_objects(labeled), start=1):
        if sl is None:
            continue
        ys, xs = sl
        seg = int((labeled[sl] == i).sum())
        if seg < MIN_COMPONENT_AREA:
            continue
        out.append((int(xs.start), int(ys.start), int(xs.stop - 1), int(ys.stop - 1), seg))
    return out
 
 
def sanitize(s):
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in s)
 
 
def place_image(src, dst):
    if os.path.exists(dst):
        return
    if LINK_MODE == "symlink":
        os.symlink(os.path.abspath(src), dst)
    else:
        shutil.copy2(src, dst)
 
 
def main():
    # join manifest (has mask) + splits (has split)
    man = {r["image"]: r for r in csv.DictReader(open(os.path.join(AUDIT_DIR, "manifest.csv"), encoding="utf-8"))}
    splits = {r["image"]: r["split"] for r in csv.DictReader(open(os.path.join(AUDIT_DIR, "splits.csv"), encoding="utf-8"))}
 
    for sub in ["images", "labels"]:
        for sp in set(splits.values()):
            os.makedirs(os.path.join(OUT_DIR, sub, sp), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "coco"), exist_ok=True)
 
    # COCO accumulators per split
    coco = {sp: {"images": [], "annotations": [], "categories": [{"id": 1, "name": "polyp"}]} for sp in COCO_SPLITS}
    img_id = {sp: 0 for sp in COCO_SPLITS}
    ann_id = {sp: 0 for sp in COCO_SPLITS}
    counts = defaultdict(lambda: [0, 0, 0])   # split -> [n_img, n_polyp, n_small640]
 
    for image_path, split in splits.items():
        r = man.get(image_path)
        if r is None:
            continue
        mask_path = r["mask"]
        # original image size
        try:
            with Image.open(image_path) as im:
                W, H = im.size
        except Exception as e:
            print(f"[skip] failed to open image {image_path}: {e}"); continue
 
        # unique filename across datasets: dataset__group__filename
        stem = f"{sanitize(r['dataset'])}__{sanitize(r['group_id'])}__{sanitize(os.path.splitext(os.path.basename(image_path))[0])}"
        ext = os.path.splitext(image_path)[1].lower()
        dst_img = os.path.join(OUT_DIR, "images", split, stem + ext)
        dst_lbl = os.path.join(OUT_DIR, "labels", split, stem + ".txt")
        place_image(image_path, dst_img)
 
        polyps = []
        if mask_path and os.path.isfile(mask_path):
            try:
                polyps = polyps_from_mask(read_mask_binary(mask_path))
            except Exception as e:
                print(f"[warn] mask failed {mask_path}: {e}")
 
        scale2 = (TARGET / max(H, W)) ** 2
        # write YOLO labels (empty if negative)
        lines = []
        for (xmin, ymin, xmax, ymax, seg) in polyps:
            bw, bh = xmax - xmin + 1, ymax - ymin + 1
            cx, cy = (xmin + bw / 2) / W, (ymin + bh / 2) / H
            lines.append(f"0 {cx:.6f} {cy:.6f} {bw / W:.6f} {bh / H:.6f}")
        open(dst_lbl, "w").write("\n".join(lines))
 
        counts[split][0] += 1
        counts[split][1] += len(polyps)
 
        # COCO GT (only for evaluation splits)
        if split in COCO_SPLITS:
            iid = img_id[split]; img_id[split] += 1
            coco[split]["images"].append({"id": iid, "file_name": stem + ext, "width": W, "height": H})
            for (xmin, ymin, xmax, ymax, seg) in polyps:
                bw, bh = xmax - xmin + 1, ymax - ymin + 1
                area640 = float(round(seg * scale2, 2))   # AREA AT 640 -> fair AP_small stratification
                if area640 < 32 * 32:
                    counts[split][2] += 1
                coco[split]["annotations"].append({
                    "id": ann_id[split], "image_id": iid, "category_id": 1,
                    "bbox": [xmin, ymin, bw, bh], "area": area640, "iscrowd": 0})
                ann_id[split] += 1
 
    # write COCO json
    for sp in COCO_SPLITS:
        p = os.path.join(OUT_DIR, "coco", f"instances_{sp}.json")
        json.dump(coco[sp], open(p, "w"))
        print(f"[coco] {p}: {len(coco[sp]['images'])} images, {len(coco[sp]['annotations'])} polyps")
 
    # data.yaml
    yaml = (
        f"# polyp dataset - leakage-free split\n"
        f"path: {os.path.abspath(OUT_DIR)}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"# ood_test (ETIS) evaluated separately: images/ood_test\n"
        f"names:\n  0: polyp\n"
    )
    open(os.path.join(OUT_DIR, "data.yaml"), "w").write(yaml)
 
    print("\n" + "=" * 60)
    print("CONVERSION SUMMARY")
    print(f"{'split':<10}{'images':>8}{'polyps':>8}{'small@640':>11}")
    for sp in ["train", "val", "test", "ood_test"]:
        n, p, s = counts[sp]
        print(f"{sp:<10}{n:>8}{p:>8}{s:>11}")
    print("=" * 60)
    print(f"[saved] {os.path.join(OUT_DIR, 'data.yaml')}")
    print("Train with data.yaml (train/val). Evaluate 'test' & 'ood_test' + COCOeval for AP_small.")
    print("Negative frames automatically become empty labels (helps suppress false-positives).")
 
 
if __name__ == "__main__":
    main()
