# Repository Manifest

Detailed description of every file in this repository.

## Root Files

| File | Description |
|------|-------------|
| `README.md` | Project overview, installation, usage instructions |
| `LICENSE` | MIT License |
| `CITATION.cff` | Citation metadata (CFF format) |
| `requirements.txt` | Python package dependencies |
| `.gitignore` | Git ignore rules for data, checkpoints, logs, etc. |
| `REPOSITORY_MANIFEST.md` | This file |
| `EXCLUDED_FILES_REPORT.md` | Files excluded from release and reasons |

## `src/` — Core Source Code

| File | Lines | Description |
|------|-------|-------------|
| `__init__.py` | 1 | Package marker |
| `model.py` | 386 | Model architectures: REDCNN, REDCNN_SE (paper backbone), SEBlock, DilatedREDCNN, UNetSmall, NAFNet |
| `dataset.py` | 368 | DICOM loading (`load_ima`), window normalization, `compute_metrics` (PSNR/SSIM), `predict_overlap_tile`, dataset classes (CTDenoiseDataset, CachedCTDataset, CTFullSliceDataset), augmentation |
| `swt_loss.py` | 146 | Stationary Wavelet Transform loss: HaarSWT2D, SWTLoss, AlternatingLoss (L1/L2/SWT cycling) |
| `diagnostics_v2.py` | 420 | Physics-informed noise diagnostics: τ2 (Gaussianity via excess kurtosis + Anderson-Darling), τ4 (heteroscedasticity via log-log regression) |
| `v7_simple.py` | 898 | AttentionUNet_Pro_v7 architecture with MHDC bottleneck (used in complementary experiments) |

## `scripts/` — Training, Evaluation, and Analysis

| File | Lines | Description |
|------|-------|-------------|
| `train_loocv.py` | 580 | **Main training script.** Unified LOO-CV for 4 methods: supervised, N2N, Nei2Nei, Noisier2Noise. All share REDCNN-SE c=74, AdamW, cosine LR, EMA, Charbonnier loss. Supports `--method`, `--test-patient`, `--device` CLI args. |
| `train_complementary.py` | 637 | Complementary experiments: v7_supervised (AttentionUNet 12M), warmstart_n2n (supervised→N2N transfer), awgn_sanity (synthetic i.i.d. validation). |
| `analyze_loocv_results.py` | 442 | Post-hoc analysis: aggregates 40 result.json files → PSNR/SSIM table, Wilcoxon signed-rank tests, scatter plots, box plots, LaTeX table. |
| `figures_and_stats.py` | 605 | Publication figures: box plots, controlled distortion, 3mm vs 1mm comparison, convergence curves. |
| `generate_paper4_figures.py` | 436 | Paper-specific figures: architecture diagram, convergence, visual comparison, diagnostic severity scatter. |
| `compute_diagnostics_per_patient.py` | 304 | Computes τ1–τ7 noise diagnostics per patient (10 data points for scatter analysis). |
| `validate_noisier2noise_synthetic.py` | 207 | Validates Noisier2Noise on synthetic AWGN to confirm implementation correctness. |
| `validate_diagnostics_1mm_and_controlled.py` | 344 | Cross-protocol τ diagnostics (1mm vs 3mm) and controlled distortion experiments. |
| `eval_bm3d_loocv.py` | 140 | BM3D classical denoising baseline (non-learned, CPU-only). |

## `configs/` — Configuration Files

Reserved for future YAML configuration files. Currently, all hyperparameters are defined as constants at the top of each training script for transparency and reproducibility.

## `docs/` — Documentation

Reserved for additional documentation (e.g., extended method descriptions, supplementary results).
