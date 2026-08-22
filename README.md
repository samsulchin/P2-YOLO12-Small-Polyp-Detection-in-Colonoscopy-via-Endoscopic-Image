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

### 2. Configuration & Results
*   `p2_yolov12s.yaml`: The modified YOLO12 architecture file with the integrated P2 head.
*   `summary_Table_*.csv`: CSV files containing the complete summarized results of ablation studies, size-stratified evaluations, OOD performance, and SOTA comparisons as presented in the paper.
*   `audit_out/`: Contains the audit trails and dataset split configurations (`manifest.csv`, `annotations.csv`, `splits.csv`).

## Datasets and External Files

To ensure the repository remains lightweight and easily cloneable, the heavy raw datasets and various preprocessed dataset versions are hosted externally on Google Drive.

**Best Weight Model and Dataset Access:** [Google Drive Link](https://drive.google.com/drive/folders/1_MmclaB8WSUJOzyYqDVHdAoJw3BR3thf)
> **Note:** The datasets are available in "View Only" mode to prevent unauthorized redistribution of medical data. If you intend to download the datasets to reproduce this research, please request access via the Google Drive interface. Your email request will be reviewed, and download permissions will be granted accordingly.

### Pre-trained Models
*   `model_p2_yolo12_full_seed42/`, `seed43/`, `seed44/`: Folders containing the pre-trained weights (`best.pt`) for the proposed method across 3 independent random seeds.

The Google Drive contains the following directories:
*   **Raw Datasets:** `CVC-ClinicDB`, `ETIS-LaribPolypDB`, `Kvasir-SEG`, `Polyp-Gen`
*   **Preprocessed Datasets (YOLO Format):** `dataset_raw`, `dataset_zerodce`, `dataset_sai`, `dataset_dwt`, `dataset_zdce_sai`, `dataset_zdce_dwt`, `dataset_sai_dwt`, `dataset_full`

### Setup Instructions

1.  Clone this repository.
2.  Install the required dependencies: `pip install ultralytics pycocotools PyWavelets opencv-python torch pandas`.
3.  Request access and download the datasets from the Google Drive link above.
4.  Place the downloaded dataset folders in the root directory of this repository to match the paths expected by the training and evaluation scripts.

## Citation
*-*
