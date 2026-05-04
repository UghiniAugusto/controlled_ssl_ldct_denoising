# Controlled Comparison of Self-Supervised Low-Dose CT Denoising Methods

Official code for the paper:

> **Controlled Comparison of Self-Supervised Low-Dose CT Denoising Methods**
> BRACIS 2026

## Overview

This repository provides a **controlled experimental framework** for comparing self-supervised denoising methods on low-dose CT (LDCT) images. All methods share:

- **Same backbone architecture**: REDCNN-SE (c=74, ~1.1M parameters)
- **Same dataset**: AAPM Mayo Low-Dose CT Grand Challenge (3mm B30)
- **Same optimizer**: AdamW (LR 2e-4 → 1e-6 cosine, WD=1e-4)
- **Same evaluation protocol**: 10-fold Leave-One-Out Cross-Validation (LOO-CV)
- **Same loss**: Charbonnier loss
- **Same augmentation**: geometric (flips + 90° rotations)

The only variable is the **supervision signal**, isolating its effect on denoising quality.

## Implemented Training Regimes

| Method | Training Signal | Requires Pairs? | Reference |
|--------|----------------|-----------------|-----------|
| **Supervised** | Quarter-dose → Full-dose | Yes (noisy–clean) | Chen et al., IEEE TMI 2017 |
| **Noise2Noise (N2N)** | Adjacent CT slices | Yes (noisy–noisy) | Lehtinen et al., ICML 2018 |
| **Neighbor2Neighbor (Nei2Nei)** | 8-connected subsampling | No | Huang et al., NeurIPS 2021 |
| **Noisier2Noise** | Noise injection + correction | No | Moran et al., ICML 2020 |

### Complementary Experiments

| Experiment | Description |
|-----------|-------------|
| **v7_supervised** | AttentionUNet_Pro_v7 (12M params) supervised LOO-CV |
| **warmstart_n2n** | REDCNN-SE initialized from supervised, fine-tuned with N2N |
| **awgn_sanity** | Noisier2Noise on synthetic AWGN (validates i.i.d. assumptions) |

## Project Structure

```
├── src/
│   ├── model.py                    # REDCNN, REDCNN_SE, SEBlock, NAFNet, UNetSmall
│   ├── dataset.py                  # Data loading, metrics (PSNR/SSIM), overlap-tile inference
│   ├── swt_loss.py                 # Stationary Wavelet Transform loss
│   ├── diagnostics_v2.py           # Physics-informed noise diagnostics (τ2, τ4)
│   └── v7_simple.py                # AttentionUNet_Pro_v7 architecture
├── scripts/
│   ├── train_loocv.py              # Main LOO-CV training (supervised, n2n, nei2nei, noisier2noise)
│   ├── train_complementary.py      # Complementary experiments (v7, warmstart, AWGN sanity)
│   ├── analyze_loocv_results.py    # Post-hoc analysis: tables, Wilcoxon tests, LaTeX
│   ├── figures_and_stats.py        # Publication figures (box plots, scatter, heatmaps)
│   ├── generate_paper4_figures.py  # Paper-specific figures (architecture, convergence, visual)
│   ├── compute_diagnostics_per_patient.py  # τ1–τ7 per patient
│   ├── validate_noisier2noise_synthetic.py # AWGN sanity validation
│   ├── validate_diagnostics_1mm_and_controlled.py  # Controlled distortion experiments
│   └── eval_bm3d_loocv.py         # BM3D classical baseline
├── configs/                        # (reserved for future YAML configs)
├── docs/                           # Additional documentation
├── requirements.txt                # Python dependencies
├── LICENSE                         # MIT License
├── CITATION.cff                    # Citation metadata
├── REPOSITORY_MANIFEST.md          # Detailed file descriptions
└── EXCLUDED_FILES_REPORT.md        # What was excluded and why
```

## Dataset

This repository uses the **AAPM Mayo Low-Dose CT Grand Challenge** dataset, which is **not included** due to licensing restrictions.

### Obtaining the Dataset

1. Request access from [AAPM/Mayo Clinic](https://www.aapm.org/grandchallenge/lowdosect/)
2. Download the training data
3. Organize it as:
   ```
   data/Traning_Image_Data/
   └── 3mm B30/
       ├── quarter_3mm/
       │   ├── L067/
       │   │   └── <dose_folder>/*.IMA
       │   ├── L096/
       │   └── ...
       └── full_3mm/
           ├── L067/
           │   └── <dose_folder>/*.IMA
           ├── L096/
           └── ...
   ```
4. Set the environment variable:
   ```bash
   export MAYO_DATA_ROOT=/path/to/Traning_Image_Data
   ```

**10 patients**: L067, L096, L109, L143, L192, L286, L291, L310, L333, L506

## Installation

```bash
# Clone this repository
git clone https://github.com/anonymous/controlled_ssl_ldct_denoising.git
cd controlled_ssl_ldct_denoising

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python ≥ 3.8
- PyTorch ≥ 1.12 (with CUDA support)
- See `requirements.txt` for complete list

## Usage

### Training (LOO-CV)

Train one method on one fold:

```bash
# Supervised (requires paired QD/FD data)
python scripts/train_loocv.py --method supervised --test-patient L506 --device cuda:0

# Noise2Noise (adjacent slices as noisy-noisy pairs)
python scripts/train_loocv.py --method n2n --test-patient L506 --device cuda:0

# Neighbor2Neighbor (single-image self-supervision)
python scripts/train_loocv.py --method nei2nei --test-patient L506 --device cuda:0

# Noisier2Noise (noise injection)
python scripts/train_loocv.py --method noisier2noise --test-patient L506 --device cuda:0
```

Run all 40 jobs (10 patients × 4 methods) for full LOO-CV:

```bash
for method in supervised n2n nei2nei noisier2noise; do
  for patient in L067 L096 L109 L143 L192 L286 L291 L310 L333 L506; do
    python scripts/train_loocv.py --method $method --test-patient $patient --device cuda:0
  done
done
```

### Complementary Experiments

```bash
# v7 architecture supervised
python scripts/train_complementary.py --experiment v7_supervised --device cuda:0

# N2N with supervised warm-start
python scripts/train_complementary.py --experiment warmstart_n2n --device cuda:1

# AWGN sanity check (i.i.d. noise validation)
python scripts/train_complementary.py --experiment awgn_sanity --device cuda:2
```

### Analysis

```bash
# Aggregate LOO-CV results (run after all 40 training jobs complete)
python scripts/analyze_loocv_results.py

# Generate publication figures
python scripts/figures_and_stats.py
python scripts/generate_paper4_figures.py

# Compute noise diagnostics per patient
python scripts/compute_diagnostics_per_patient.py

# BM3D classical baseline
python scripts/eval_bm3d_loocv.py

# Validate Noisier2Noise on synthetic AWGN
python scripts/validate_noisier2noise_synthetic.py --device cuda:0
```

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Architecture | REDCNN-SE (c=74) | ~1.1M params, fair comparison across methods |
| Loss | Charbonnier | Robust to outliers, smooth near zero |
| Optimizer | AdamW | Standard for medical imaging |
| LR schedule | Cosine (2e-4 → 1e-6) | Smooth decay, good convergence |
| EMA | decay=0.999 | Stabilizes evaluation |
| Eval | LOO-CV (10 folds) | Maximizes data usage, per-patient statistics |
| Metrics | PSNR, SSIM (win=7) | Standard for LDCT denoising literature |
| Window | WL=40, WW=400 HU | Abdomen soft-tissue standard |

## Model Architecture

**REDCNN-SE**: 5-layer encoder-decoder with Squeeze-and-Excitation (SE) blocks on skip connections.

```
Input (1ch) → Enc1→Enc2→Enc3→Enc4→Enc5 → Dec5+SE(e4)→Dec4+SE(e3)→Dec3+SE(e2)→Dec2+SE(e1)→Dec1+x → Output
```

- All convolutions: 5×5, padding=2
- SE blocks: reduction=4, channel attention on skip connections
- Global residual: output = decoder_output + input

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{anon2026controlled,
  title={Controlled Comparison of Self-Supervised Low-Dose CT Denoising Methods},
  author={Anonymous},
  booktitle={Brazilian Conference on Intelligent Systems (BRACIS)},
  year={2026}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

- [AAPM Mayo Low-Dose CT Grand Challenge](https://www.aapm.org/grandchallenge/lowdosect/) for the dataset
- Original method papers: Noise2Noise (Lehtinen et al.), Noise2Void (Krull et al.), Neighbor2Neighbor (Huang et al.), Noisier2Noise (Moran et al.)
- RED-CNN architecture (Chen et al., IEEE TMI 2017)
