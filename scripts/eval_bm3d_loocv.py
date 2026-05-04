#!/usr/bin/env python3
"""
BM3D baseline evaluation — CPU-only, per-patient LOO-CV.
No training needed: BM3D is a classical (non-learned) denoiser.

Evaluates BM3D on all 10 patients' QD slices vs FD ground truth.
Results saved to loocv_results/bm3d_results.json
"""
import sys, os, json, time
import numpy as np
from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from dataset import collect_paired_paths, load_ima, window_normalize

DATA_ROOT = os.environ.get('MAYO_DATA_ROOT', './data/Traning_Image_Data')
KERNEL = '3mm B30'
PATIENTS = ['L067','L096','L109','L143','L192','L286','L291','L310','L333','L506']
OUT_DIR = Path(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results'))


def estimate_sigma(qd_imgs, fd_imgs, n_sample=20):
    """Estimate noise σ from QD-FD residuals."""
    idxs = np.linspace(0, len(qd_imgs)-1, min(n_sample, len(qd_imgs)), dtype=int)
    stds = []
    for i in idxs:
        noise = qd_imgs[i] - fd_imgs[i]
        stds.append(np.std(noise))
    return float(np.mean(stds))


def run_bm3d_patient(patient, all_pairs):
    """Run BM3D on all slices of one patient."""
    import bm3d

    pairs = [(q, f) for q, f in all_pairs if patient in q]
    print(f"  {patient}: {len(pairs)} slices", flush=True)

    # Load all slices
    qd_imgs, fd_imgs = [], []
    for qf, ff in pairs:
        qd_imgs.append(window_normalize(load_ima(qf)))
        fd_imgs.append(window_normalize(load_ima(ff)))

    # Estimate noise level from residuals
    sigma = estimate_sigma(qd_imgs, fd_imgs)
    print(f"    Estimated σ = {sigma:.5f}", flush=True)

    # Input PSNR (QD vs FD)
    input_psnrs = []
    for qd, fd in zip(qd_imgs, fd_imgs):
        input_psnrs.append(peak_signal_noise_ratio(fd, qd, data_range=1.0))
    input_psnr = float(np.mean(input_psnrs))

    # Run BM3D on each slice
    psnrs, ssims = [], []
    t0 = time.time()
    for i, (qd, fd) in enumerate(zip(qd_imgs, fd_imgs)):
        denoised = bm3d.bm3d(qd, sigma_psd=sigma, stage_arg=bm3d.BM3DStages.ALL_STAGES)
        denoised = np.clip(denoised, 0, 1).astype(np.float32)

        p = peak_signal_noise_ratio(fd, denoised, data_range=1.0)
        s = structural_similarity(fd, denoised, data_range=1.0, win_size=7,
                                  gaussian_weights=True, sigma=1.5)
        psnrs.append(p)
        ssims.append(s)

        if (i+1) % 50 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i+1) / elapsed
            eta = (len(qd_imgs) - i - 1) / rate if rate > 0 else 0
            print(f"    [{i+1}/{len(qd_imgs)}] PSNR={np.mean(psnrs):.2f} "
                  f"SSIM={np.mean(ssims):.4f} ({rate:.1f} sl/s, ETA {eta:.0f}s)", flush=True)

    result = {
        'patient': patient,
        'n_slices': len(pairs),
        'sigma_est': sigma,
        'input_psnr': input_psnr,
        'bm3d_psnr': float(np.mean(psnrs)),
        'bm3d_psnr_std': float(np.std(psnrs)),
        'bm3d_ssim': float(np.mean(ssims)),
        'bm3d_ssim_std': float(np.std(ssims)),
        'gain_over_input': float(np.mean(psnrs)) - input_psnr,
    }
    print(f"    DONE: Input={input_psnr:.2f}  BM3D={result['bm3d_psnr']:.2f}±{result['bm3d_psnr_std']:.2f}  "
          f"SSIM={result['bm3d_ssim']:.4f}  Gain={result['gain_over_input']:+.2f} dB", flush=True)
    return result


def main():
    print("="*70)
    print("  BM3D BASELINE EVALUATION (CPU-only, no training)")
    print("="*70)

    all_pairs = collect_paired_paths(DATA_ROOT, kernel=KERNEL)
    print(f"  Total pairs: {len(all_pairs)}")

    results = {}
    for pat in PATIENTS:
        t0 = time.time()
        results[pat] = run_bm3d_patient(pat, all_pairs)
        elapsed = time.time() - t0
        print(f"    Time: {elapsed:.0f}s\n", flush=True)

    # Summary table
    print("="*70)
    print("  SUMMARY")
    print("="*70)
    print(f"  {'Patient':>8} │ {'Input':>7} │ {'BM3D':>7} │ {'±STD':>5} │ {'SSIM':>6} │ {'Gain':>6}")
    print(f"  {'─'*8} ┼ {'─'*7} ┼ {'─'*7} ┼ {'─'*5} ┼ {'─'*6} ┼ {'─'*6}")

    all_psnrs = []
    all_ssims = []
    all_inputs = []
    for pat in PATIENTS:
        r = results[pat]
        all_psnrs.append(r['bm3d_psnr'])
        all_ssims.append(r['bm3d_ssim'])
        all_inputs.append(r['input_psnr'])
        print(f"  {pat:>8} │ {r['input_psnr']:>7.2f} │ {r['bm3d_psnr']:>7.2f} │ {r['bm3d_psnr_std']:>5.2f} │ "
              f"{r['bm3d_ssim']:>6.4f} │ {r['gain_over_input']:>+6.2f}")

    print(f"  {'─'*8} ┼ {'─'*7} ┼ {'─'*7} ┼ {'─'*5} ┼ {'─'*6} ┼ {'─'*6}")
    mean_gain = np.mean(all_psnrs) - np.mean(all_inputs)
    print(f"  {'MEAN':>8} │ {np.mean(all_inputs):>7.2f} │ {np.mean(all_psnrs):>7.2f} │ "
          f"{np.std(all_psnrs):>5.2f} │ {np.mean(all_ssims):>6.4f} │ {mean_gain:>+6.2f}")

    # Save
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / 'bm3d_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {out_path}")


if __name__ == '__main__':
    main()
