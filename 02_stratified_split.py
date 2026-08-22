
import os, csv, random
from collections import defaultdict
 
# ============ EDIT THIS (if necessary) ============
AUDIT_DIR = r"./audit_out"
SEED = 42
RATIO_TRAIN, RATIO_VAL = 0.80, 0.10   # remainder -> test (0.10)
GROUPED = {"CVC-ClinicDB", "PolypGen"} # datasets that MUST be grouped per-sequence
IMAGE_LEVEL = {"Kvasir-SEG"}           # independent image datasets
OOD = {"ETIS-Larib"}                   # OOD test dataset (fully held-out)
# ===============================================
 
 
def load():
    man = list(csv.DictReader(open(os.path.join(AUDIT_DIR, "manifest.csv"), encoding="utf-8")))
    ann = list(csv.DictReader(open(os.path.join(AUDIT_DIR, "annotations.csv"), encoding="utf-8")))
    small_per_img = defaultdict(int)
    for a in ann:
        if a.get("size_coco") == "small":
            small_per_img[a["image"]] += 1
    return man, small_per_img
 
 
SPLITS3 = ["train", "val", "test"]
def _ratio(s):
    return {"train": RATIO_TRAIN, "val": RATIO_VAL, "test": 1 - RATIO_TRAIN - RATIO_VAL}[s]
 
 
def greedy_group_split(rows, small_per_img, seed):
    """STRATIFIED per-group split: distribute groups with small polyps proportionally to train/val/test."""
    groups = defaultdict(list)
    for r in rows:
        groups[r["group_id"]].append(r)
    gsize = {g: len(rs) for g, rs in groups.items()}
    gsmall = {g: sum(small_per_img.get(r["image"], 0) for r in rs) for g, rs in groups.items()}
    tot_img, tot_small = sum(gsize.values()), sum(gsmall.values())
    tgt_img = {s: _ratio(s) * tot_img for s in SPLITS3}
    tgt_small = {s: _ratio(s) * tot_small for s in SPLITS3}
    cur_img = {s: 0 for s in SPLITS3}; cur_small = {s: 0 for s in SPLITS3}
    rng = random.Random(seed)
    assign = {}
    # 1) small-polyp groups first (desc), place in split with largest small deficit
    small_groups = sorted([g for g in groups if gsmall[g] > 0], key=lambda g: (-gsmall[g], rng.random()))
    for g in small_groups:
        s = max(SPLITS3, key=lambda s: (tgt_small[s] - cur_small[s]) / (tgt_small[s] + 1e-9))
        assign[g] = s; cur_img[s] += gsize[g]; cur_small[s] += gsmall[g]
    # 2) the rest (no small polyps) to balance the number of images
    rest = [g for g in groups if gsmall[g] == 0]; rng.shuffle(rest)
    for g in rest:
        s = max(SPLITS3, key=lambda s: (tgt_img[s] - cur_img[s]) / (tgt_img[s] + 1e-9))
        assign[g] = s; cur_img[s] += gsize[g]
    return {r["image"]: assign[r["group_id"]] for r in rows}
 
 
def image_level_split(rows, small_per_img, seed):
    """STRATIFIED image-level: separate images with small vs no-small, split each 80/10/10."""
    out = {}
    for subset in ([r for r in rows if small_per_img.get(r["image"], 0) > 0],
                   [r for r in rows if small_per_img.get(r["image"], 0) == 0]):
        subset = subset[:]; random.Random(seed).shuffle(subset)
        n = len(subset); n_tr, n_va = int(RATIO_TRAIN * n), int((RATIO_TRAIN + RATIO_VAL) * n)
        for i, r in enumerate(subset):
            out[r["image"]] = "train" if i < n_tr else ("val" if i < n_va else "test")
    return out
 
 
def main():
    man, small_per_img = load()
    by_ds = defaultdict(list)
    for r in man:
        by_ds[r["dataset"]].append(r)
 
    split_of = {}
    for ds, rows in by_ds.items():
        if ds in OOD:
            for r in rows:
                split_of[r["image"]] = "ood_test"
        elif ds in GROUPED:
            split_of.update(greedy_group_split(rows, small_per_img, SEED))
        elif ds in IMAGE_LEVEL:
            split_of.update(image_level_split(rows, small_per_img, SEED))
        else:
            print(f"[warn] dataset '{ds}' unclassified -> default image-level")
            split_of.update(image_level_split(rows, small_per_img, SEED))
 
    # ---- write splits.csv ----
    out_p = os.path.join(AUDIT_DIR, "splits.csv")
    with open(out_p, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["image", "dataset", "group_id", "split", "has_polyp", "n_small"])
        for r in man:
            w.writerow([r["image"], r["dataset"], r["group_id"], split_of[r["image"]],
                        r["has_polyp"], small_per_img.get(r["image"], 0)])
 
    # ---- ANTI-LEAKAGE VERIFICATION ----
    grp_splits = defaultdict(set)
    for r in man:
        if r["dataset"] in GROUPED:
            grp_splits[(r["dataset"], r["group_id"])].add(split_of[r["image"]])
    leaks = {k: v for k, v in grp_splits.items() if len(v) > 1}
    print("=" * 74)
    print("ANTI-LEAKAGE VERIFICATION (grouped datasets):")
    if leaks:
        print("  [FAILED] the following groups appear in >1 split:")
        for k, v in list(leaks.items())[:10]:
            print("   ", k, "->", v)
    else:
        print("  [OK] no sequence leaks across splits. ZERO leakage.")
    # ETIS must be 100% ood
    etis_splits = {split_of[r["image"]] for r in man if r["dataset"] in OOD}
    print(f"  ETIS/OOD unique splits: {etis_splits} (must be {{'ood_test'}})")
 
    # ---- summary ----
    def agg(pred):
        imgs = [r for r in man if pred(r)]
        pos = sum(1 for r in imgs if r["has_polyp"] == "1")
        neg = sum(1 for r in imgs if r["has_polyp"] != "1")
        small = sum(small_per_img.get(r["image"], 0) for r in imgs)
        grps = len({r["group_id"] for r in imgs})
        return len(imgs), pos, neg, small, grps
 
    print("\n" + "=" * 74)
    print("SPLIT SUMMARY")
    hdr = f"{'split':<10}{'img':>6}{'img+':>6}{'img-':>6}{'small':>7}{'group':>6}"
    print(hdr); print("-" * len(hdr))
    for s in ["train", "val", "test", "ood_test"]:
        n, pos, neg, small, grps = agg(lambda r: split_of[r["image"]] == s)
        print(f"{s:<10}{n:>6}{pos:>6}{neg:>6}{small:>7}{grps:>6}")
    print("-" * len(hdr))
    # per dataset x split (small polyps)
    print("\nSmall polyps per split x dataset:")
    ds_list = sorted({r['dataset'] for r in man})
    print(f"{'dataset':<16}" + "".join(f"{s:>10}" for s in ['train','val','test','ood_test']))
    for ds in ds_list:
        line = f"{ds:<16}"
        for s in ['train','val','test','ood_test']:
            sm = sum(small_per_img.get(r['image'],0) for r in man
                     if r['dataset']==ds and split_of[r['image']]==s)
            line += f"{sm:>10}"
        print(line)
    print(f"\n[saved] {out_p}")
    print("Note: 'test' = in-distribution test (held-out group). 'ood_test' = ETIS (generalization).")
    print("Negative frames (img-) in train can be used for training; in test/ood used for false-positive evaluation.")
 
 
if __name__ == "__main__":
    main()
