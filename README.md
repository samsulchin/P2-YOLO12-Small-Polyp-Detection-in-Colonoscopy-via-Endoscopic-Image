# P2-YOLO12: Small Polyp Detection in Colonoscopy

This repository contains the official implementation of the paper:
**"P2-YOLO12: Small Polyp Detection in Colonoscopy via Endoscopic Image Preprocessing and a High-Resolution Detection Head"**

## Overview

Accurate polyp detection during colonoscopy is crucial but challenging due to the small size, low contrast, and irregular shape of polyps. This project introduces **P2-YOLO12**, a framework that addresses these challenges through a dual approach:
1.  **Image Preprocessing Pipeline:** A sequential pipeline combining Zero-Reference Deep Curve Estimation (Zero-DCE) for low-light correction, Specular-Aware Inpainting (SAI) for reflection removal, and Discrete Wavelet Transform (DWT) for edge-preserving noise reduction.
2.  **Architectural Modification:** The addition of a high-resolution **P2 detection head** (stride 4) to the YOLO12 backbone, specifically designed to capture the fine details of small and diminutive polyps.

Our framework achieves state-of-the-art performance, particularly in small-polyp detection, evaluated under a strict, leakage-free (patient/sequence-grouped) protocol with out-of-distribution (OOD) cross-dataset testing.

## Repository Structure

Based on the provided implementation, the repository is structured as follows:

### 1. Code & Scripts
*   `01_audit_native.py` & `01_audit_640.py`: Scripts for auditing dataset resolution and object sizes.
*   `02_stratified_split.py`: Generates leakage-free, patient/sequence-grouped data splits.
*   `03_yolo_coco_format.py`: Converts mask annotations into YOLO bounding box format and COCO JSON format.
*   `05_preprocess_pipeline.py`: The core image preprocessing pipeline (Zero-DCE -> SAI -> DWT).
*   `06_train_all.py` & `06b_train_seeds.py`: Automated training scripts for ablation studies and statistical validation.
*   `clinical_fp.py`, `coco_fixed.py`, `compute_metrics.py`, `eval_coco_force.py`, `eval_test.py`, `statistical_val.py`: Comprehensive evaluation scripts for COCO metrics, false alarms, and computational efficiency.
*   `12_benchmark_e2e.py`: Benchmarking script to rigorously measure the end-to-end inference latency, total FPS, and peak GPU memory (at batch=1 with a warm-up phase) for the complete proposed pipeline (Zero-DCE -> SAI -> DWT -> Detector).
*   `build_group_manifest.py`: **(new)** Builds the unified, per-image group-ID manifest used for leakage-free splitting across all four datasets (see [Data Splitting & Group-ID Manifest](#data-splitting--group-id-manifest-leakage-prevention) below).

### 2. Configuration & Results
*   `p2_yolov12s.yaml`: The modified YOLO12 architecture file with the integrated P2 head.
*   `summary_Table_*.csv`: CSV files containing the complete summarized results of ablation studies, size-stratified evaluations, OOD performance, and SOTA comparisons as presented in the paper.
*   `audit_out/`: Contains the audit trails and dataset split configurations (`manifest.csv`, `annotations.csv`, `splits.csv`).
*   `data_splits/all_datasets_group_manifest.csv`: **(new)** Per-image group-ID manifest for all four datasets (see below).

### 3. Model P2-YOLO12
*   `Best Model P2-YOLO12.pt`: Pre-trained Model P2-YOLO12

## Data Splitting & Group-ID Manifest (Leakage Prevention)

To ensure that the reported in-distribution results are not inflated by data leakage (e.g., near-duplicate frames from the same colonoscopy sequence appearing in both training and test partitions), every image in every dataset is assigned a **group ID**, and the train/validation/test split is performed at the group level rather than the image level. Groups are never split across partitions.

The grouping level used for each dataset reflects exactly what metadata is publicly available — we do not claim a finer grouping level than the source data actually supports:

| Dataset | Grouping level | Source of group ID | Number of groups | Number of images |
|---|---|---|---|---|
| CVC-ClinicDB | Sequence | `sequence_id` field in the dataset's official `metadata.csv` | 29 | 612 |
| PolypGen | Sequence | Native folder structure (`seq1`–`seq23`), one folder per colonoscopy procedure | 23 | 2,225 |
| Kvasir-SEG | Image | N/A — no case, sequence, or patient metadata is publicly released for this dataset | 1,000 | 1,000 |
| ETIS-LaribPolypDB | N/A (external test set) | Used in its entirety as a held-out external test set; not partitioned | — | 196 |

**Reproducing the manifest:** running `build_group_manifest.py` against the four raw dataset folders regenerates `data_splits/all_datasets_group_manifest.csv`, a per-image table with the following columns:

```
dataset, image_path, mask_path, group_id, grouping_level, split
```

This file is provided so that reviewers and future users can verify, image by image, exactly which group and which split (`train` / `val` / `test` / `external_test`) each frame belongs to, and confirm that no group appears in more than one split.

## Datasets and External Files

To ensure the repository remains lightweight and easily cloneable, the heavy raw datasets, preprocessed dataset versions, and pre-trained weights are hosted externally on Google Drive.

**Dataset Access:** [Google Drive Link](https://drive.google.com/drive/folders/1_MmclaB8WSUJOzyYqDVHdAoJw3BR3thf)


The Google Drive contains the following directories:
*   Raw Datasets: `CVC-ClinicDB`, `ETIS-LaribPolypDB`, `Kvasir-SEG`, `Polyp-Gen`
*   Preprocessed Datasets (YOLO Format): `dataset_raw`, `dataset_zerodce`, `dataset_sai`, `dataset_dwt`, `dataset_zdce_sai`, `dataset_zdce_dwt`, `dataset_sai_dwt`, `dataset_full`
*   Pre-trained Models: `model_p2_yolo12_full_seed42/`, `seed43/`, `seed44/` containing the pre-trained weights (`best.pt`) for the proposed method across 3 independent random seeds.

### Setup Instructions

1.  Clone this repository.
2.  Install the required dependencies: `pip install ultralytics pycocotools PyWavelets opencv-python torch pandas`.
3.  Request access and download the datasets from the Google Drive link above.
4.  Place the downloaded dataset folders in the root directory of this repository to match the paths expected by the training and evaluation scripts.
5.  (Optional, for verifying data splits) Run `python build_group_manifest.py` to regenerate `data_splits/all_datasets_group_manifest.csv` from the raw dataset folders.

## Citation

This paper is currently under review. The citation information will be updated upon formal acceptance and publication. 

If you find this repository and our proposed preprocessing pipeline useful for your research in the meantime, please consider starring this repository and citing it as follows:

```bibtex
@article{arifin2026p2yolo12,
  title={P2-YOLO12: Small Polyp Detection in Colonoscopy via Endoscopic Image Preprocessing and a High-Resolution Detection Head},
  author={Arifin, Samsul and Suciati, Nanik and Azhar, Daffa Muhammad},
  journal={Under Review},
  year={2026}
}
```
