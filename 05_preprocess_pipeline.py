#!/usr/bin/env python3
"""
05_preprocess_pipeline.py  (IMPROVED VERSION)
============================================
Improvements from previous version:
  1. DCENet: e_conv* layer names match Epoch99.pth (not conv*)
  2. SAI: threshold 220->240 + adaptive guard (avoids inpainting non-specular lumen)
  3. DWT: threshold 15->5 + adaptive BayesShrink (preserves fine polyp edges)
  4. Zero-DCE sanity check: verifies the model actually modifies the image
  5. Before/after sample: saves visual comparisons for R2-4 verification

Pipeline order: Zero-DCE -> SAI -> DWT
"""

import os, cv2, shutil
import numpy as np
import pywt
import torch
import torch.nn as nn
from pathlib import Path

# ============ CONFIGURATION ============
DATASET_DIR   = r"./dataset"
OUT_BASE      = r"."
ZERODCE_CKPT  = r"./Epoch99.pth"
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
SPLITS        = ["train", "val", "test", "ood_test"]
IMG_EXTS      = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
SAVE_SAMPLES  = True
N_SAMPLES     = 5

# SAI - IMPROVED: 220 -> 240
SAI_THRESHOLD = 240
SAI_KERNEL    = 3
SAI_RADIUS    = 3

# DWT - IMPROVED: 15 -> 5
DWT_WAVELET   = "haar"
DWT_LEVEL     = 1
DWT_THRESHOLD = 5
# =====================================


# ─────────────────────────────────────
# MODULE 1: Zero-DCE — IMPROVED
# Layer names e_conv* match Epoch99.pth
# ─────────────────────────────────────

class DCENet(nn.Module):
    """
    Official DCE-Net (Li-Chongyi/Zero-DCE).
    Layer names e_conv1..e_conv7 match Epoch99.pth checkpoint.
    e_conv5/6/7 receives concatenated input (skip connection).
    """
    def __init__(self, num_iterations=8):
        super().__init__()
        self.num_iterations = num_iterations
        self.relu    = nn.ReLU(inplace=True)
        self.e_conv1 = nn.Conv2d(3,  32, 3, padding=1)
        self.e_conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.e_conv3 = nn.Conv2d(32, 32, 3, padding=1)
        self.e_conv4 = nn.Conv2d(32, 32, 3, padding=1)
        self.e_conv5 = nn.Conv2d(64, 32, 3, padding=1)   # cat(x3,x4)
        self.e_conv6 = nn.Conv2d(64, 32, 3, padding=1)   # cat(x2,x5)
        self.e_conv7 = nn.Conv2d(64, 3*num_iterations, 3, padding=1)  # cat(x1,x6)
        self.tanh    = nn.Tanh()

    def forward(self, x):
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], dim=1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], dim=1)))
        x_r = self.tanh(self.e_conv7(torch.cat([x1, x6], dim=1)))
        return x_r

    def enhance(self, x):
        x_r = self.forward(x)
        enhanced = x
        for i in range(self.num_iterations):
            alpha = x_r[:, i*3:(i+1)*3, :, :]
            enhanced = enhanced + alpha * enhanced * (1 - enhanced)
        return torch.clamp(enhanced, 0, 1)


def load_zerodce(ckpt_path, device):
    model = DCENet(num_iterations=8).to(device)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"[Zero-DCE] Checkpoint not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    missing = set(model.state_dict().keys()) - set(state.keys())
    if missing:
        raise RuntimeError(f"[Zero-DCE] Missing key(s): {missing}\nCheck layer names in the checkpoint.")
    model.load_state_dict(state)
    model.eval()
    print(f"[Zero-DCE] OK — {len(state)} layers loaded | device={device}")
    return model


@torch.no_grad()
def apply_zerodce(img_bgr, model, device):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor  = torch.from_numpy(img_rgb.transpose(2,0,1)).unsqueeze(0).to(device)
    out     = model.enhance(tensor)
    out_rgb = (out.squeeze(0).permute(1,2,0).cpu().numpy()*255).clip(0,255).astype(np.uint8)
    return cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)


def quick_sanity_check(model, device):
    """Ensure Zero-DCE actually brightens dark images."""
    dummy = np.random.randint(20, 80, (288,384,3), dtype=np.uint8)
    out   = apply_zerodce(dummy, model, device)
    delta = out.mean() - dummy.mean()
    ok    = "OK" if delta > 1.0 else "WARNING: delta is too small, check checkpoint!"
    print(f"  [Sanity Zero-DCE] Δbrightness={delta:+.1f} — {ok}")


# ─────────────────────────────────────
# MODULE 2: SAI — IMPROVED
# threshold 220->240, adaptive guard >30%
# ─────────────────────────────────────

def apply_sai(img_bgr, threshold=SAI_THRESHOLD, kernel_size=SAI_KERNEL, radius=SAI_RADIUS):
    """
    Specular-Aware Inpainting.
    Improvement: threshold increased 220->240 to avoid inpainting bright lumen
    (200-230) that is not specular. If >30% of the image is detected
    as specular, threshold is automatically raised to 245 for that image.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    # adaptive guard: if too much area is detected, raise threshold
    if mask.sum() / 255 / mask.size > 0.30:
        _, mask = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY)
    kernel  = np.ones((kernel_size, kernel_size), np.uint8)
    mask_d  = cv2.dilate(mask, kernel, iterations=1)
    return cv2.inpaint(img_bgr, mask_d, radius, cv2.INPAINT_TELEA)


# ─────────────────────────────────────
# MODULE 3: DWT — IMPROVED
# threshold 15->5 + adaptive BayesShrink
# ─────────────────────────────────────

def bayes_threshold(coeff):
    """
    BayesShrink: adaptive threshold per sub-band.
    Safer than fixed=15 to preserve fine polyp edges (coeff 8-14).
    """
    sigma   = np.median(np.abs(coeff)) / 0.6745
    if sigma < 1e-6: return 0.0
    sigma_x = max(0.0, float(np.var(coeff)) - float(sigma**2)) ** 0.5
    if sigma_x < 1e-6: return float('inf')
    return float(sigma**2) / float(sigma_x)


def soft_threshold(coeff, t):
    return np.sign(coeff) * np.maximum(np.abs(coeff) - t, 0)


def apply_dwt(img_bgr, wavelet=DWT_WAVELET, fixed_thr=DWT_THRESHOLD, use_bayes=True):
    """
    DWT Haar level-1 per BGR channel.
    LL (approximation) is preserved without modification.
    LH/HL/HH are thresholded:
      - use_bayes=True  -> adaptive BayesShrink, capped at max 3x fixed_thr
      - use_bayes=False -> fixed threshold=5
    Improvement from threshold=15 which erases polyp edges (coeff 8-14).
    """
    channels = cv2.split(img_bgr)
    denoised = []
    for ch in channels:
        cA, (cH, cV, cD) = pywt.dwt2(ch.astype(np.float32), wavelet)
        if use_bayes:
            tH = min(bayes_threshold(cH), fixed_thr * 3)
            tV = min(bayes_threshold(cV), fixed_thr * 3)
            tD = min(bayes_threshold(cD), fixed_thr * 3)
        else:
            tH = tV = tD = fixed_thr
        cH = soft_threshold(cH, tH)
        cV = soft_threshold(cV, tV)
        cD = soft_threshold(cD, tD)
        rec = pywt.idwt2((cA, (cH, cV, cD)), wavelet)
        denoised.append(np.clip(rec, 0, 255).astype(np.uint8))
    return cv2.merge(denoised)


# ─────────────────────────────────────
# PIPELINE & UTILITIES
# ─────────────────────────────────────

VARIANTS = {
    "dataset_zerodce":  (True,  False, False),
    "dataset_sai":      (False, True,  False),
    "dataset_dwt":      (False, False, True),
    "dataset_zdce_sai": (True,  True,  False),
    "dataset_zdce_dwt": (True,  False, True),
    "dataset_sai_dwt":  (False, True,  True),
    "dataset_full":     (True,  True,  True),   # proposed method
}


def process_image(img, use_zdce, use_sai, use_dwt, model, device):
    result = img.copy()
    if use_zdce: result = apply_zerodce(result, model, device)
    if use_sai:  result = apply_sai(result)
    if use_dwt:  result = apply_dwt(result)
    return result


def save_sample(src, dst, out_path):
    """Save before/after comparison for visual verification (R2-4)."""
    h = max(src.shape[0], dst.shape[0])
    s = cv2.resize(src, (int(src.shape[1]*h/src.shape[0]), h))
    d = cv2.resize(dst, (int(dst.shape[1]*h/dst.shape[0]), h))
    cv2.putText(s, "BEFORE", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.putText(d, "AFTER",  (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, np.hstack([s, d]))


def build_variant(name, use_zdce, use_sai, use_dwt, model, device):
    src = Path(DATASET_DIR)
    dst = Path(OUT_BASE) / name
    ok = skip = n_sample = 0
    sample_dir = str(dst / "samples_before_after")

    for split in SPLITS:
        si = src/"images"/split; sl = src/"labels"/split
        di = dst/"images"/split; dl = dst/"labels"/split
        di.mkdir(parents=True, exist_ok=True)
        dl.mkdir(parents=True, exist_ok=True)
        if not si.exists(): continue

        for fp in si.iterdir():
            if fp.suffix.lower() not in IMG_EXTS: continue
            try:
                img = cv2.imread(str(fp))
                if img is None: skip += 1; continue
                proc = process_image(img, use_zdce, use_sai, use_dwt, model, device)
                cv2.imwrite(str(di/fp.name), proc)
                if SAVE_SAMPLES and split=="train" and n_sample < N_SAMPLES:
                    save_sample(img, proc, f"{sample_dir}/s{n_sample:02d}_{fp.name}")
                    n_sample += 1
            except Exception as e:
                print(f"  [skip] {fp.name}: {e}"); skip += 1; continue
            lbl_src = sl/(fp.stem+".txt")
            lbl_dst = dl/(fp.stem+".txt")
            if lbl_src.exists(): shutil.copy2(str(lbl_src), str(lbl_dst))
            else: lbl_dst.touch()
            ok += 1

        # copy COCO json
        coco_s = src/"coco"
        if coco_s.exists():
            coco_d = dst/"coco"; coco_d.mkdir(exist_ok=True)
            for jf in coco_s.glob(f"instances_{split}.json"):
                shutil.copy2(str(jf), str(coco_d/jf.name))

    (dst/"data.yaml").write_text(
        f"# {name} | DCE={use_zdce} SAI={use_sai} DWT={use_dwt}\n"
        f"path: {dst.resolve()}\n"
        f"train: images/train\nval: images/val\ntest: images/test\n"
        f"names:\n  0: polyp\n"
    )
    return ok, skip


def verify_labels(src_dir, dst_dir, n=5):
    s = Path(src_dir)/"labels"/"train"
    d = Path(dst_dir)/"labels"/"train"
    if not (s.exists() and d.exists()): return
    files = sorted(s.glob("*.txt"))[:n]
    all_ok = all((d/f.name).exists() and f.read_text()==(d/f.name).read_text() for f in files)
    print(f"  {'[OK]' if all_ok else '[FAILED]'} Identical labels ({len(files)} samples) — R2-4 satisfied.")


def main():
    print("="*65)
    print("PREPROCESSING PIPELINE (IMPROVED VERSION)")
    print(f"  Zero-DCE : e_conv* (matches Epoch99.pth)")
    print(f"  SAI      : threshold={SAI_THRESHOLD} + adaptive guard (previously 220)")
    print(f"  DWT      : threshold={DWT_THRESHOLD} + BayesShrink (previously 15)")
    print(f"  Device   : {DEVICE}")
    print("="*65)

    model = load_zerodce(ZERODCE_CKPT, DEVICE)
    quick_sanity_check(model, DEVICE)

    for name, (zd, sa, dw) in VARIANTS.items():
        dst = Path(OUT_BASE)/name
        if dst.exists() and any(dst.glob("images/**/*.*")):
            print(f"\n[skip] {name} already exists — delete the folder to regenerate.")
            continue
        print(f"\n--- {name}  (DCE={zd} SAI={sa} DWT={dw}) ---")
        ok, skip = build_variant(name, zd, sa, dw, model, DEVICE)
        print(f"    {ok} images OK, {skip} skipped.")
        verify_labels(DATASET_DIR, str(dst))

    print("\n"+"="*65)
    print("DONE. Delete the old folder then re-run training with")
    print("dataset_full/ for proposed method.")
    print()
    print("Training example after this:")
    print("  yolo train model=yolo12s.pt data=dataset_full/data.yaml \\")
    print("    epochs=300 imgsz=640 batch=32 optimizer=AdamW seed=42")


if __name__ == "__main__":
    main()
