
import os
import numpy as np
import pandas as pd

# =========================== CONFIG ===========================
ROOT = r"C:\VISKOM"          # <-- SESUAIKAN path root di komputer kamu

TEST_SIZE = 0.20
VAL_SIZE = 0.10
RANDOM_STATE = 42
CVC_USE_FORMAT = "png"  # "png" atau "tif" -- pilih format yang dipakai untuk training
# ================================================================

VALID_EXT = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


def scan_images(img_dir):
    if not os.path.isdir(img_dir):
        return []
    return sorted(f for f in os.listdir(img_dir) if f.lower().endswith(VALID_EXT))


def numeric_key(fname):
    """Sort numerik aman untuk nama file seperti 0.jpg, 1.jpg, ..., 10.jpg."""
    stem = os.path.splitext(fname)[0]
    return int(stem) if stem.isdigit() else stem


# ---------------------------------------------------------------
# 1) KVASIR-SEG  -> image-level
# ---------------------------------------------------------------
def build_kvasir_seg(root):
    img_dir = os.path.join(root, "kvasir-seg", "images")
    mask_dir = os.path.join(root, "kvasir-seg", "masks")
    records = []
    for fname in scan_images(img_dir):
        image_id = os.path.splitext(fname)[0]
        records.append({
            "dataset": "Kvasir-SEG",
            "image_path": os.path.join(img_dir, fname),
            "mask_path": os.path.join(mask_dir, fname),
            "group_id": f"KvasirSEG_{image_id}",
            "grouping_level": "image",
        })
    return records


# ---------------------------------------------------------------
# 2) ETIS-LaribPolypDB -> image-level, external test set (tidak displit)
# ---------------------------------------------------------------
def build_etis(root):
    img_dir = os.path.join(root, "ETIS-LaribPolypDB", "images")
    mask_dir = os.path.join(root, "ETIS-LaribPolypDB", "masks")
    records = []
    for fname in sorted(scan_images(img_dir), key=numeric_key):
        image_id = os.path.splitext(fname)[0]
        records.append({
            "dataset": "ETIS-LaribPolypDB",
            "image_path": os.path.join(img_dir, fname),
            "mask_path": os.path.join(mask_dir, fname),
            "group_id": f"ETIS_{image_id}",
            "grouping_level": "image",
            "split": "external_test",
        })
    return records


# ---------------------------------------------------------------
# 3) POLYP-GEN -> sequence-level (seq1 .. seq23)
# ---------------------------------------------------------------
def build_polypgen(root):
    base = os.path.join(root, "Polyp-Gen")
    records = []
    seq_folders = sorted(
        (d for d in os.listdir(base) if d.lower().startswith("seq")),
        key=lambda x: int(''.join(filter(str.isdigit, x)) or 0),
    )
    for seq_folder in seq_folders:
        img_dir = os.path.join(base, seq_folder, "images")
        mask_dir = os.path.join(base, seq_folder, "masks")
        for fname in sorted(scan_images(img_dir), key=numeric_key):
            mask_path = os.path.join(mask_dir, fname)
            if not os.path.exists(mask_path):
                print(f"WARNING: mask hilang -> {mask_path}")
                continue
            records.append({
                "dataset": "PolypGen",
                "image_path": os.path.join(img_dir, fname),
                "mask_path": mask_path,
                "group_id": f"PolypGen_{seq_folder}",
                "grouping_level": "sequence",
            })
    return records


# ---------------------------------------------------------------
# 4) CVC-CLINICDB -> dari metadata.csv, kolom sequence_id (VERIFIED: 29 grup unik)
# ---------------------------------------------------------------
def build_cvc_clinicdb(root):
    meta_path = os.path.join(root, "CVC-ClinicDB", "metadata.csv")
    df = pd.read_csv(meta_path)
    base = os.path.join(root, "CVC-ClinicDB")

    img_col = "png_image_path" if CVC_USE_FORMAT == "png" else "tif_image_path"
    mask_col = "png_mask_path" if CVC_USE_FORMAT == "png" else "tif_mask_path"

    records = []
    for _, row in df.iterrows():
        records.append({
            "dataset": "CVC-ClinicDB",
            "image_path": os.path.join(base, row[img_col]),
            "mask_path": os.path.join(base, row[mask_col]),
            "group_id": f"ClinicDB_seq{row['sequence_id']}",
            "grouping_level": "sequence",
        })
    return records


# ---------------------------------------------------------------
# GROUP-AWARE SPLIT (train/val/test tanpa group yang sama nyebrang split)
# ---------------------------------------------------------------
def split_by_group(df, test_size=TEST_SIZE, val_size=VAL_SIZE, seed=RANDOM_STATE):
    groups = df["group_id"].unique()
    rng = np.random.RandomState(seed)
    rng.shuffle(groups)

    n = len(groups)
    n_test = max(1, round(n * test_size))
    n_val = max(1, round(n * val_size)) if n - n_test > 1 else 0

    test_groups = set(groups[:n_test])
    val_groups = set(groups[n_test:n_test + n_val])

    def label(g):
        if g in test_groups:
            return "test"
        if g in val_groups:
            return "val"
        return "train"

    df = df.copy()
    df["split"] = df["group_id"].map(label)
    return df


def main():
    records = []
    records += build_kvasir_seg(ROOT)
    records += build_etis(ROOT)
    records += build_polypgen(ROOT)
    records += build_cvc_clinicdb(ROOT)

    manifest = pd.DataFrame(records)

    # split in-distribution dataset satu per satu (ETIS sudah fixed "external_test")
    parts = []
    for ds in manifest["dataset"].unique():
        subset = manifest[manifest["dataset"] == ds].copy()
        if ds == "ETIS-LaribPolypDB":
            parts.append(subset)  # sudah punya kolom split = external_test
        else:
            parts.append(split_by_group(subset))
    manifest = pd.concat(parts, ignore_index=True)

    out_path = "all_datasets_group_manifest.csv"
    manifest.to_csv(out_path, index=False)

    print("\n=== RINGKASAN JUMLAH GROUP UNIK PER DATASET ===")
    print(manifest.groupby(["dataset", "grouping_level"])["group_id"].nunique())

    print("\n=== JUMLAH GAMBAR PER DATASET x SPLIT ===")
    print(manifest.groupby(["dataset", "split"]).size())

    print(f"\nManifest tersimpan di: {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()