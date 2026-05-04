# Excluded Files Report

This document details files and directories from the original project that were **excluded** from this public release, along with the reason for exclusion.

## Exclusion Categories

| Category | Reason |
|----------|--------|
| **Patient Data** | AAPM Mayo dataset is not redistributable; requires official access |
| **Checkpoints** | Large binary files (.pt); users retrain from scratch |
| **Logs** | Training logs contain local paths and are environment-specific |
| **Figures** | Generated from code; reproducible by running analysis scripts |
| **Results** | Intermediate results; reproducible from training |
| **Other Projects** | Unrelated to this paper (SAD, NoisePhysics-SSL, etc.) |
| **Dashboards** | Runtime monitoring tools, not needed for reproduction |
| **Archives** | Large compressed files, deprecated experiments |
| **Cache** | Python bytecode, IDE settings |

## Detailed Exclusions

### Patient Data

| Path | Size | Reason |
|------|------|--------|
| `Traning_Image_Data/` | ~50 GB | AAPM Mayo LDCT training images (.IMA DICOM). Not redistributable. |
| `Testing_Image_Data/` | ~5 GB | AAPM Mayo test images. Not redistributable. |
| `L506_sinograms/` | ~20 GB | Raw sinogram data. Not redistributable. |

### Model Checkpoints

| Path | Size | Reason |
|------|------|--------|
| `ldct_denoiser/checkpoints/` | ~2 GB | All trained model weights (.pt files) |
| `ldct_denoiser/loocv_results/fold_*/*/best.pt` | ~400 MB | LOO-CV best checkpoints (40 files) |
| `ldct_denoiser/FINAL_RESULTS/` | ~1 GB | Archived final checkpoints (read-only) |

### Training Logs

| Path | Reason |
|------|--------|
| `ldct_denoiser/logs/` | Training logs containing local paths |
| `ldct_denoiser/loocv_results/fold_*/*/train.log` | Per-fold training logs |

### Generated Figures and Results

| Path | Reason |
|------|--------|
| `ldct_denoiser/figures/` | All generated figures (reproducible from scripts) |
| `ldct_denoiser/paper4_results/` | Paper figures and intermediate data |
| `ldct_denoiser/stats/` | Statistical test outputs |
| `ldct_denoiser/loocv_results/*/result.json` | Per-fold result JSONs |

### Other Projects (Not Part of This Paper)

| Path | Description |
|------|-------------|
| `Structure-aware diffusion for low-dose CT imaging/` | SAD diffusion model project |
| `noisephysics_ssl/` | NoisePhysics-SSL framework (NeurIPS project) |
| `Controlled_Comparison_of_Self-Supervised_Low-Dose_CT_Denoising_Methods/` | LaTeX paper source |
| `controlled_comparison_paper/` | Paper drafts |

### Dashboards and Monitoring

| Path | Reason |
|------|--------|
| `ldct_denoiser/dashboard.py` | Runtime monitoring, not for reproduction |
| `ldct_denoiser/dashboard_complementary.py` | Complementary experiment monitoring |

### Excluded Source Files (Not Relevant to Paper)

| File | Reason |
|------|--------|
| `ldct_denoiser/train_n2n_selfsup.py` | Superseded by `train_loocv.py` (unified script) |
| `ldct_denoiser/train_n2v.py` | Superseded by `train_loocv.py` |
| `ldct_denoiser/train_complementary_single.py` | Earlier version; `train_complementary.py` included instead |
| `ldct_denoiser/eval_l506.py` | Single-patient eval; LOO-CV analysis scripts included instead |
| `ldct_denoiser/eval_paper4_comprehensive.py` | Comprehensive eval; analysis scripts included |
| `ldct_denoiser/inference_tta.py` | TTA inference; not part of paper protocol |
| `ldct_denoiser/NOISE_PHYSICS_RESEARCH.md` | Research notes (not code) |
| Various `train_redcnn_*.py` | Earlier training scripts for other papers |

### Cache and System Files

| Pattern | Reason |
|---------|--------|
| `__pycache__/` | Python bytecode cache |
| `.venv/` | Virtual environment (not portable) |
| `*.pyc` | Compiled Python files |
| `.vscode/`, `.idea/` | IDE configuration |

## Path Sanitization

All absolute paths of the form `/mnt/A-SSD/ughini/Mayo_Grand_Challenge/...` were replaced with:

| Original Pattern | Replacement |
|-----------------|-------------|
| `.../Traning_Image_Data` | `os.environ.get('MAYO_DATA_ROOT', './data/Traning_Image_Data')` |
| `.../ldct_denoiser/loocv_results` | Relative path via `os.path.dirname(__file__)` |
| `.../ldct_denoiser/checkpoints` | Relative path via `os.path.dirname(__file__)` |
| `.../ldct_denoiser/logs` | Relative path via `os.path.dirname(__file__)` |
| `.../ldct_denoiser/figures` | Relative path via `os.path.dirname(__file__)` |
| `.../ldct_denoiser/stats` | Relative path via `os.path.dirname(__file__)` |
| `.../ldct_denoiser/paper4_results` | Relative path via `os.path.dirname(__file__)` |

Users configure the dataset path via the `MAYO_DATA_ROOT` environment variable.
