# P2-YOLO12: Small Polyp Detection in Colonoscopy

This repository contains the official implementation of the paper:
**"P2-YOLO12: Small Polyp Detection in Colonoscopy via Endoscopic Image Preprocessing and a High-Resolution Detection Head"**

## Overview

Accurate polyp detection during colonoscopy is crucial but challenging due to the small size, low contrast, and irregular shape of polyps. This project introduces **P2-YOLO12**, a framework that addresses these challenges through a dual approach:
1.  **Image Preprocessing Pipeline:** A sequential pipeline combining Zero-Reference Deep Curve Estimation (Zero-DCE) for low-light correction, Specular-Aware Inpainting (SAI) for reflection removal, and Discrete Wavelet Transform (DWT) for edge-preserving noise reduction.
2.  **Architectural Modification:** The addition of a high-resolution **P2 detection head** (stride 4) to the YOLO12 backbone, specifically designed to capture the fine details of small and diminutive polyps.

Compared with the unmodified YOLO12 baseline and with earlier YOLO generations (YOLOv9, YOLOv10, YOLO11) retrained on the same split, the framework improves small-polyp sensitivity. It is evaluated under a **leakage-controlled**, group-based protocol (sequence-level grouping for CVC-ClinicDB and PolypGen, image-level partitioning for Kvasir-SEG) and by external testing on a single, fully held-out dataset (ETIS-LaribPolypDB).

The stride-4 (P2) head is an established small-object technique and is not claimed as novel in itself. The contribution of this work lies in (i) its adaptation to YOLO12 for polyp detection, (ii) its joint use with the preprocessing pipeline, and (iii) the evaluation protocol with size-stratified metrics and a held-out external dataset.

## Key Results

The numbers below are the means over three training runs (seeds 42, 43, 44), as reported in the paper. Please see the paper for standard deviations, per-seed values, and the full discussion.

| Item | Result |
|---|---|
| P2-YOLO12 Full, in-distribution test | Precision 95.80%, Recall 93.15%, F1 94.46%, mAP50 97.25%, mAP50-95 71.85% |
| Unmodified YOLO12 (raw images), mAP50 | 87.76% |
| P2-YOLO12 (raw images), mAP50 | 95.15% (most of the accuracy gain comes from the P2 head itself) |
| Effect of the full preprocessing pipeline on mAP50 | +2.10 points on P2-YOLO12, but -1.27 points on standard YOLO12 |
| External test (ETIS-LaribPolypDB), mAP50 | 85.25% for the proposed method (12.3% relative drop versus 22.0% for the YOLO12 baseline) |
| Small-polyp recall | 40.91% to 77.27% in-distribution (n = 22); 25.00% to 50.00% on ETIS (n = 16). Exploratory, given the very small lesion counts |
| False-positive boxes on the 92 held-out negative test frames | 2 (proposed) versus 69 (YOLO12 baseline). Frame-level count only; it does not capture temporal behavior in continuous video |
| Computational cost | 9.36 M parameters (9.23 M for YOLO12); 29.9 versus 23.2 GFLOPs; detector-only latency 5.36 versus 3.28 ms; full pipeline 23.44 ms per frame (about 42.7 FPS) |

Timing was measured on a single workstation (NVIDIA RTX 5090 and Intel Core Ultra 9 285K, batch size 1) and does not include video decoding. It has not been measured on lower-tier clinical hardware, so no claim about deployment readiness is made.

## Repository Structure

Based on the provided implementation, the repository is structured as follows:

### 1. Code & Scripts
*   `01_audit_native.py` & `01_audit_640.py`: Scripts for auditing dataset resolution and object sizes (small/medium/large polyp categories follow the COCO convention and are computed at the 640 x 640 model input resolution).
*   `02_stratified_split.py`: Generates the leakage-controlled, group-based data splits actually used for all reported experiments (80/10/10, seed 42). CVC-ClinicDB and PolypGen are split at the sequence level, Kvasir-SEG at the image level, and group assignment is stratified by the number of small polyps. ETIS-LaribPolypDB is fully withheld as the external test set. The output is written to `audit_out/splits.csv`.
*   `03_yolo_coco_format.py`: Converts mask annotations into YOLO bounding box format and COCO JSON format.
*   `05_preprocess_pipeline.py`: The core image preprocessing pipeline (Zero-DCE -> SAI -> DWT).
*   `06_train_all.py` & `06b_train_seeds.py`: Automated training scripts for ablation studies (28 scenarios) and statistical validation (three matched seeds: 42, 43, 44).
*   `clinical_fp.py`, `coco_fixed.py`, `compute_metrics.py`, `eval_coco_force.py`, `eval_test.py`, `statistical_val.py`: Comprehensive evaluation scripts for COCO metrics, false alarms, and statistical analysis.
*   `12_benchmark_e2e.py`: Benchmarking script to rigorously measure the end-to-end inference latency, total FPS, and peak GPU memory (at batch=1 with a warm-up phase) for the complete proposed pipeline (Zero-DCE -> SAI -> DWT -> Detector).
*   `build_group_manifest.py`: **(new)** Builds the unified, per-image group-ID manifest for all four datasets, using the split assignment stored in `audit_out/splits.csv` (see [Data Splitting & Group-ID Manifest](#data-splitting--group-id-manifest-leakage-control) below).

### 2. Configuration & Results
*   `p2_yolov12s.yaml`: The modified YOLO12 architecture file with the integrated P2 head.
*   `summary_Table_*.csv`: CSV files containing the complete summarized results of ablation studies, size-stratified evaluations, external-test performance, and same-split comparisons as presented in the paper. Note that the file names follow an earlier numbering of the manuscript tables; see the mapping below.
*   `audit_out/`: Contains the audit trails and dataset split configurations (`manifest.csv`, `annotations.csv`, `splits.csv`). `splits.csv` is the authoritative record of the split used for all experiments.
*   `data_splits/all_datasets_group_manifest.csv`: **(new)** Per-image group-ID manifest for all four datasets (see below).

#### Mapping of result files to the tables in the paper

| File | Table in the paper |
|---|---|
| `summary_Table_5_Ablation_Studies.csv` | Table 5 |
| `summary_Table_6_standard_deviation.csv` | Table 6 |
| `summary_Table_7_Size_Stratified_FP.csv` | Table 9 |
| `summary_Table_8_Computational_Efficiency.csv` | Table 10 |
| `summary_Table_9_OOD_Performance.csv` | Table 11 |
| `summary_Table_10_Relative_Degradation.csv` | Table 12 |
| `summary_Table_11_Head_to_Head_SOTA.csv` | Table 13 (same-split comparison with earlier YOLO generations and a CLAHE baseline) |
| `summary_Table_12_Contextual_SOTA.csv` | Table 14 (contextual literature values, not reproduced on the same split) |

### 3. Model P2-YOLO12
*   `Best Model P2-YOLO12.pt`: Pre-trained Model P2-YOLO12

## Data Splitting & Group-ID Manifest (Leakage Control)

To reduce the risk that the reported in-distribution results are inflated by data leakage (e.g., near-duplicate frames from the same colonoscopy sequence appearing in both training and test partitions), every image in every dataset is assigned a **group ID**, and the train/validation/test split is performed at the group level rather than the image level wherever sequence metadata exist. Sequences are never split across partitions. We describe the protocol as *leakage-controlled* rather than leakage-free, because of the limitations listed below.

The grouping level used for each dataset reflects exactly what metadata is publicly available — we do not claim a finer grouping level than the source data actually supports:

| Dataset | Grouping level | Source of group ID | Number of groups | Number of images |
|---|---|---|---|---|
| CVC-ClinicDB | Sequence | `sequence_id` field in the dataset's official `metadata.csv` | 29 | 612 |
| PolypGen | Sequence | Native folder structure (`seq1`–`seq23`), one folder per colonoscopy procedure | 23 | 2,225 (1,710 positive + 515 negative) |
| Kvasir-SEG | Image | N/A — no case, sequence, or patient metadata is publicly released for this dataset | 1,000 | 1,000 |
| ETIS-LaribPolypDB | N/A (external test set) | Used in its entirety as a held-out external test set; not partitioned | — | 196 |

**Partition.** The in-distribution pool (Kvasir-SEG + CVC-ClinicDB + PolypGen) is divided into training, validation, and in-distribution test subsets at a ratio of approximately 80/10/10, with a fixed seed (42) for reproducibility. Because small polyps are a rare minority, group assignment is stratified by the number of small polyps so that the validation and test subsets contain an adequate cohort of them. Zero overlap between partitions is verified programmatically. ETIS-LaribPolypDB is never used for training, validation, in-distribution testing, or preprocessing-parameter selection.

| Split | Total images | Positive | Negative | Small polyps (@640) | Groups (sequences + images) |
|---|---|---|---|---|---|
| Training | 3,025 | 2,653 | 372 | 83 | 44 sequences + 799 Kvasir-SEG images |
| Validation | 419 | 368 | 51 | 27 | 4 sequences + 100 Kvasir-SEG images |
| Test (in-distribution) | 393 | 301 | 92 | 22 | 4 sequences + 101 Kvasir-SEG images |
| External test (ETIS-LaribPolypDB) | 196 | 196 | 0 | 16 | 196 images (held out) |

In total, the in-distribution partition involves 52 sequences (29 + 23) and 1,000 Kvasir-SEG images.

**Reproducing the manifest:** running `build_group_manifest.py` regenerates `data_splits/all_datasets_group_manifest.csv`, a per-image table with the following columns:

```
dataset, image_path, mask_path, group_id, grouping_level, split
```

The `split` column (`train` / `val` / `test` / `external_test`) is identical to the assignment in `audit_out/splits.csv`. This file is provided so that reviewers and future users can verify, image by image, exactly which group and which split each frame belongs to, and confirm that no sequence appears in more than one split.

### Limitations of the protocol

*   No patient identifier or center-level holdout rule is available in the public metadata of the datasets. Kvasir-SEG is therefore partitioned at the image level, and residual leakage between its training and test images cannot be fully excluded.
*   The external evaluation is a single held-out dataset (ETIS-LaribPolypDB), not a complete leave-one-dataset-out study, and external validity is limited accordingly.
*   The small-polyp evaluation relies on very few lesions (n = 22 in-distribution, n = 16 on ETIS) and should be regarded as exploratory.
*   The 92 held-out negative test frames are not a realistic surrogate for untrimmed colonoscopy video. False-alarm counts are frame-level and do not capture temporal persistence or detection delay.
*   No comparison with other small-target-oriented polyp detectors on the identical split is included, so superiority claims are restricted to the compared YOLO generations and the CLAHE baseline.

## Evaluation Protocol Summary

*   **Scenarios:** 28 in total (12 for YOLOv9/YOLOv10/YOLO11 under four preprocessing levels, 8 for the full factorial preprocessing ablation on YOLO12, and 8 for the same ablation on P2-YOLO12).
*   **Training seeds:** each reported configuration is trained with seeds 42, 43, and 44 on the identical fixed data split. The seeds vary only weight initialization and data-loading order, so the reported standard deviations capture optimization variability and not the sampling uncertainty of the test partition.
*   **Interaction analysis:** the interaction between preprocessing and the P2 head is evaluated as a paired contrast within matched seeds. With three seeds (df = 2), the mAP50 result is suggestive rather than conclusive, and the mAP50-95 result is inconclusive.
*   **Small-polyp recall:** 95% Wilson intervals are computed on the seed-averaged true-positive count out of the number of small-polyp instances (the lesions, not the seeds, are the sampling unit).
*   **Preprocessing parameters:** all preprocessing hyperparameters were selected by visual inspection on a small subset of the training split only. No validation, in-distribution test, or ETIS image was used for parameter selection. The pretrained Zero-DCE checkpoint from the [official repository](https://github.com/Li-Chongyi/Zero-DCE) is used at inference only, without fine-tuning.

## Environment

All experiments were run on Linux with a single NVIDIA GeForce RTX 5090 GPU, an Intel Core Ultra 9 285K processor, and 64 GB of RAM, using Ultralytics 8.4.50 (YOLO12s, 640 x 640 input, up to 300 epochs with early stopping at patience 20, AdamW, batch size 64, initial learning rate 0.002).

## Datasets and External Files

To ensure the repository remains lightweight and easily cloneable, the heavy raw datasets, preprocessed dataset versions, and pre-trained weights are hosted externally on Google Drive.

**Dataset Access:** [Google Drive Link](https://drive.google.com/drive/folders/1_MmclaB8WSUJOzyYqDVHdAoJw3BR3thf)


The Google Drive contains the following directories:
*   Raw Datasets: `CVC-ClinicDB`, `ETIS-LaribPolypDB`, `Kvasir-SEG`, `Polyp-Gen`
*   Preprocessed Datasets (YOLO Format): `dataset_raw`, `dataset_zerodce`, `dataset_sai`, `dataset_dwt`, `dataset_zdce_sai`, `dataset_zdce_dwt`, `dataset_sai_dwt`, `dataset_full`
*   Pre-trained Models: `model_p2_yolo12_full_seed42/`, `seed43/`, `seed44/` containing the pre-trained weights (`best.pt`) for the proposed method across 3 independent random seeds.

### Setup Instructions

1.  Clone this repository.
2.  Install the required dependencies: `pip install ultralytics pycocotools PyWavelets opencv-python torch pandas` (the reported experiments used Ultralytics 8.4.50).
3.  Request access and download the datasets from the Google Drive link above.
4.  Place the downloaded dataset folders in the root directory of this repository to match the paths expected by the training and evaluation scripts.
5.  (Optional, for verifying data splits) Run `python build_group_manifest.py` to regenerate `data_splits/all_datasets_group_manifest.csv` from the raw dataset folders and `audit_out/splits.csv`.

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
