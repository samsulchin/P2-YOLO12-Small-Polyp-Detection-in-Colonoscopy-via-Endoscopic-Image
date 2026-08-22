
import os, csv, json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage
 
# ============ EDIT THIS ============
BASE = r"/home/samsul/Documents/Polyp Detection"          # main folder
OUT_DIR = r"./audit_out"             # output folder
MIN_COMPONENT_AREA = 25              # remove mask blobs < 25 px (noise)
# ==================================
 
# COCO Threshold (on SEGMENTATION AREA / number of mask pixels):
#   small < 32^2 (1024) ; medium < 96^2 (9216) ; large >= 9216
COCO_SMALL, COCO_LARGE = 32 * 32, 96 * 96
TARGET = 640   # YOLO input resolution -> size recalculated here (letterbox)
# Clinical threshold (on BBOX AREA, PolypDB style): small<=100x100 ; large>200x200
CLIN_SMALL, CLIN_LARGE = 100 * 100, 200 * 200
 
IMG_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
 
def _norm(s):
    """normalize folder name: lowercase, remove non-alphanumeric (spaces, -, :, _)."""
    return "".join(ch for ch in s.lower() if ch.isalnum())
 
def find_subdir(base, *aliases):
    """find subdirectory in base matching one of the aliases (case-insensitive, ignore punctuation)."""
    if not os.path.isdir(base):
        return None
    targets = [_norm(a) for a in aliases]
    for name in os.listdir(base):
        full = os.path.join(base, name)
        if os.path.isdir(full) and _norm(name) in targets:
            return full
    # partial match (e.g. "polypgen" inside "polypgenvideo")
    for name in os.listdir(base):
        full = os.path.join(base, name)
        if os.path.isdir(full) and any(t in _norm(name) for t in targets):
            return full
    return None
 
 
 
def read_mask_binary(path):
    """Read mask -> boolean array (True = polyp). Polyp = bright pixel (>127)."""
    im = Image.open(path)
    arr = np.array(im)
    if arr.ndim == 3:
        arr = arr[..., :3].max(axis=2)   # merge channels; white remains bright
    return arr > 127
 
 
def polyps_from_mask(mask_bool):
    """Connected components -> list of polyps: (xmin,ymin,xmax,ymax,seg_area,bbox_area)."""
    labeled, n = ndimage.label(mask_bool)
    out = []
    for i, sl in enumerate(ndimage.find_objects(labeled), start=1):
        if sl is None:
            continue
        ys, xs = sl
        comp = (labeled[sl] == i)
        seg_area = int(comp.sum())
        if seg_area < MIN_COMPONENT_AREA:
            continue
        xmin, xmax = int(xs.start), int(xs.stop - 1)
        ymin, ymax = int(ys.start), int(ys.stop - 1)
        bbox_area = (xmax - xmin + 1) * (ymax - ymin + 1)
        out.append((xmin, ymin, xmax, ymax, seg_area, bbox_area))
    return out
 
 
def coco_size(seg_area):
    if seg_area < COCO_SMALL:
        return "small"
    if seg_area < COCO_LARGE:
        return "medium"
    return "large"
 
 
def clinical_size(bbox_area):
    if bbox_area <= CLIN_SMALL:
        return "small"
    if bbox_area <= CLIN_LARGE:
        return "medium"
    return "large"
 
 
def diameter_px(seg_area):
    return round(2 * (seg_area / np.pi) ** 0.5, 1)   # equivalent circle diameter
 
 
# ---------- annotation collector ----------
ANNS = []   # row per polyp
IMGS = []   # row per image
 
 
def add_image(dataset, image_path, mask_path, group_id):
    """Process 1 labeled image: extract polyps from mask, record. Size calculated at 640."""
    try:
        mb = read_mask_binary(mask_path)
        polyps = polyps_from_mask(mb)
    except Exception as e:
        print(f"    [skip] failed to read mask {mask_path}: {e}")
        return
    H, W = mb.shape[:2]
    scale = TARGET / max(H, W)          # letterbox: uniform scale
    area_factor = scale * scale
    IMGS.append(dict(dataset=dataset, image=image_path, mask=mask_path,
                     group_id=group_id, n_polyp=len(polyps),
                     has_polyp=int(len(polyps) > 0), img_h=H, img_w=W))
    for j, (xmin, ymin, xmax, ymax, seg, bba) in enumerate(polyps):
        seg640 = int(round(seg * area_factor))      # segmentation area at 640
        bba640 = int(round(bba * area_factor))      # bbox area at 640
        rel = round(100 * seg / (H * W), 3)         # % of image area (resolution-free)
        ANNS.append(dict(dataset=dataset, image=image_path, group_id=group_id,
                         polyp_idx=j, xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax,
                         img_h=H, img_w=W,
                         seg_area_native=seg, seg_area_640=seg640, bbox_area_640=bba640,
                         rel_area_pct=rel,
                         size_coco=coco_size(seg640),            # PRIMARY: 640-based
                         size_coco_native=coco_size(seg),        # reference: original resolution
                         size_clinical=clinical_size(bba640),
                         diam_px_640=diameter_px(seg640)))
 
 
def add_negative(dataset, image_path, group_id):
    IMGS.append(dict(dataset=dataset, image=image_path, mask="", group_id=group_id,
                     n_polyp=0, has_polyp=0))
 
 
# ---------- per dataset ----------
def do_cvc(base):
    root = find_subdir(base, "CVC-ClinicDB", "CVCClinicDB")
    if not root:
        print("[CVC] CVC-ClinicDB folder not found - skipping"); return
    meta = os.path.join(root, "metadata.csv")
    if not os.path.isfile(meta):
        print(f"[CVC] metadata.csv not found in {root} - skipping"); return
    n = 0
    with open(meta, newline="", encoding="utf-8", errors="replace") as fh:
        for row in csv.DictReader(fh):
            img = os.path.join(root, row["png_image_path"])
            msk = os.path.join(root, row["png_mask_path"])
            gid = f"cvc_seq_{row['sequence_id']}"
            if os.path.isfile(msk):
                add_image("CVC-ClinicDB", img, msk, gid); n += 1
            else:
                print(f"    [CVC] missing mask: {msk}")
    print(f"[CVC] {n} images processed.")
 
 
def _pair_by_stem(img_dir, msk_dir):
    """Match image<->mask based on filename (without extension)."""
    def index(d):
        out = {}
        for f in os.listdir(d):
            if os.path.splitext(f)[1].lower() in IMG_EXT:
                out[os.path.splitext(f)[0]] = os.path.join(d, f)
        return out
    imgs, msks = index(img_dir), index(msk_dir)
    pairs = []
    for stem, ip in imgs.items():
        mp = msks.get(stem)
        if mp:
            pairs.append((ip, mp))
    return pairs, len(imgs), len(msks)
 
 
def do_simple(base, folder, dataset, group_id):
    """ETIS / Kvasir: images/ + masks/."""
    root = find_subdir(base, *folder) if isinstance(folder, (list, tuple)) else find_subdir(base, folder)
    if not root:
        print(f"[{dataset}] dataset folder not found - skipping"); return
    img_dir = find_subdir(root, "images", "image", "original", "frames")
    msk_dir = find_subdir(root, "masks", "mask", "groundtruth", "ground truth", "gt")
    if not (img_dir and msk_dir):
        print(f"[{dataset}] images/masks subfolder not found in {root} - skipping"); return
    pairs, ni, nm = _pair_by_stem(img_dir, msk_dir)
    for ip, mp in pairs:
        add_image(dataset, ip, mp, group_id)
    print(f"[{dataset}] {len(pairs)} pairs processed (images={ni}, masks={nm}).")
 
 
def do_polypgen(base, folder="Polyp-Gen"):
    """
    PolypGen sequence: seqN/ contains frames + masks (+ possibly negative frames).
    DEFENSIVE: auto-detect mask vs image subfolders. Adjust if necessary after seeing output.
    """
    root = find_subdir(base, "Polyp-Gen","PolypGen","polypgen","Polyp Gen")
    if not root:
        print("[PolypGen] Polyp-Gen folder not found - skipping"); return
    seqs = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
    total_img = total_neg = 0
    for s in seqs:
        sdir = os.path.join(root, s)
        gid = f"pgen_{s}"
        subs = {d.lower(): os.path.join(sdir, d) for d in os.listdir(sdir)
                if os.path.isdir(os.path.join(sdir, d))}
        # guess image & mask folder from their names
        img_dir = find_subdir(sdir, "images","image","frames","original")
        msk_dir = find_subdir(sdir, "masks","mask","gt","groundtruth","ground truth","label")
        if img_dir and msk_dir:
            pairs, ni, nm = _pair_by_stem(img_dir, msk_dir)
            paired_stems = {os.path.splitext(os.path.basename(ip))[0] for ip, _ in pairs}
            for ip, mp in pairs:
                add_image("PolypGen", ip, mp, gid); total_img += 1
            # frames without matching mask = negative candidates
            for f in os.listdir(img_dir):
                stem = os.path.splitext(f)[0]
                if os.path.splitext(f)[1].lower() in IMG_EXT and stem not in paired_stems:
                    add_negative("PolypGen", os.path.join(img_dir, f), gid); total_neg += 1
        else:
            print(f"    [PolypGen/{s}] cannot guess img/mask dir. Subfolders: {list(subs.keys())}")
    print(f"[PolypGen] {len(seqs)} sequences, {total_img} positive frames, {total_neg} negative frames (estimate).")
 
 
# ---------- summary ----------
def summarize():
    datasets = sorted({a["dataset"] for a in ANNS} | {i["dataset"] for i in IMGS})
    print("\n" + "=" * 78)
    print("SIZE AUDIT SUMMARY - COCO threshold on segmentation area @ 640x640 (letterbox)")
    print("=" * 78)
    header = f"{'dataset':<16}{'img+':>6}{'img-':>6}{'group':>6}{'polyps':>7}{'small':>7}{'med':>6}{'large':>7}{'%small':>8}"
    print(header); print("-" * len(header))
    for d in datasets:
        anns = [a for a in ANNS if a["dataset"] == d]
        imgs = [i for i in IMGS if i["dataset"] == d]
        img_pos = sum(1 for i in imgs if i["has_polyp"])
        img_neg = sum(1 for i in imgs if not i["has_polyp"])
        grp = len({i["group_id"] for i in imgs})
        ns = sum(1 for a in anns if a["size_coco"] == "small")
        nm = sum(1 for a in anns if a["size_coco"] == "medium")
        nl = sum(1 for a in anns if a["size_coco"] == "large")
        tot = len(anns)
        pct = f"{100*ns/tot:.1f}%" if tot else "-"
        print(f"{d:<16}{img_pos:>6}{img_neg:>6}{grp:>6}{tot:>7}{ns:>7}{nm:>6}{nl:>7}{pct:>8}")
    print("-" * len(header))
    print("img+ = positive images | img- = negative frames | group = number of group_id (anti-leakage unit)")
    print("Note: area calculated at 640x640 RESOLUTION (as seen by model). small<1024px, medium<9216px, large>=9216px.")
    print("Column size_coco_native in annotations.csv = classification at original resolution (reference).")
 
 
def write_csv():
    os.makedirs(OUT_DIR, exist_ok=True)
    ann_p = os.path.join(OUT_DIR, "annotations.csv")
    man_p = os.path.join(OUT_DIR, "manifest.csv")
    if ANNS:
        with open(ann_p, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(ANNS[0].keys())); w.writeheader(); w.writerows(ANNS)
    if IMGS:
        with open(man_p, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(IMGS[0].keys())); w.writeheader(); w.writerows(IMGS)
    print(f"\n[saved] {ann_p}  ({len(ANNS)} polyps)")
    print(f"[saved] {man_p}  ({len(IMGS)} images)")
 
 
if __name__ == "__main__":
    print("SIZE AUDIT - BASE =", os.path.abspath(BASE))
    do_cvc(BASE)
    do_simple(BASE, ["ETIS-LaribPolypDB","ETIS-Larib","ETIS"], "ETIS-Larib", "etis")
    do_simple(BASE, ["kvasir-seg","kvasirseg","Kvasir-SEG"], "Kvasir-SEG", "kvasir")
    do_polypgen(BASE, "Polyp-Gen")
    summarize()
    write_csv()
    print("\n[done] Paste the summary above into the chat to decide train/test composition & title framing.")
