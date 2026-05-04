#!/usr/bin/env python3
"""
Two-part diagnostic validation (CPU-only):

  Part 1 — τ1–τ7 on 1mm B30 (cross-protocol validation)
    Physics predicts: τ6↓ (thinner slices → more overlap), τ1↑ (fewer photons),
    τ3≈same (ramp filter invariant to slice thickness).

  Part 2 — Controlled distortion on FD images (calibration)
    (a) Inject i.i.d. AWGN       → expect τ1≈0, τ3≈0
    (b) Inject correlated noise   → expect τ3↑ proportionally
    (c) Inject non-stationary noise → expect τ1↑ proportionally

  Output: JSON + console summary comparing 3mm vs 1mm vs controlled.
"""
import sys, os, json, time
import numpy as np
from pathlib import Path
from scipy import stats
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from dataset import collect_paired_paths, load_ima
from diagnostics_v2 import compute_tau2_gaussianity, compute_tau4_heteroscedasticity

# ═════════════════════════════════════════════════════════════════════════════
DATA_ROOT = os.environ.get('MAYO_DATA_ROOT', './data/Traning_Image_Data')
ALL_PATIENTS = ['L067', 'L096', 'L109', 'L143', 'L192',
                'L286', 'L291', 'L310', 'L333', 'L506']
OUT_DIR = Path(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results'))
MAX_SLICES = 50
DOMAIN = 'hu_soft_tissue'


def _to_domain(slices_hu, domain='hu_soft_tissue'):
    if domain == 'hu_soft_tissue':
        return slices_hu.astype(np.float32)
    raise ValueError(f"unknown domain: {domain}")


def load_patient_data(patient_id, all_pairs, max_slices=MAX_SLICES):
    pairs = [(q, f) for q, f in all_pairs if patient_id in q]
    if len(pairs) > max_slices:
        idx = np.linspace(0, len(pairs)-1, max_slices, dtype=int)
        pairs = [pairs[i] for i in idx]
    qd_imgs, fd_imgs = [], []
    for qf, ff in pairs:
        qd_imgs.append(_to_domain(load_ima(qf)))
        fd_imgs.append(_to_domain(load_ima(ff)))
    return np.array(qd_imgs), np.array(fd_imgs)


def compute_tau_quick(qd_all, fd_all):
    """Compute key τ metrics (τ1, τ3, τ5, τ6, τ7) — fast version."""
    noise = qd_all - fd_all
    N, H, W = noise.shape

    # τ1: CV of local noise σ
    block = 32
    local_stds = []
    for i in range(0, N, max(1, N // 20)):
        for y in range(0, H - block, block):
            for x in range(0, W - block, block):
                patch = noise[i, y:y+block, x:x+block]
                if np.std(fd_all[i, y:y+block, x:x+block]) > 0.01:
                    local_stds.append(np.std(patch))
    local_stds = np.array(local_stds)
    tau1 = float(np.std(local_stds) / np.mean(local_stds)) if len(local_stds) > 0 else 0.0

    # τ3: Autocorrelation lag-1
    autocorrs = []
    for i in range(0, N, max(1, N // 20)):
        n_row = noise[i, H // 2, :]
        if np.std(n_row) > 1e-6:
            r = np.corrcoef(n_row[:-1], n_row[1:])[0, 1]
            autocorrs.append(r)
    tau3 = float(np.mean(autocorrs)) if autocorrs else 0.0

    # τ5: |skewness|
    skews = []
    for i in range(0, N, max(1, N // 20)):
        skews.append(float(stats.skew(noise[i].ravel())))
    tau5 = float(np.mean(np.abs(skews))) if skews else 0.0

    # τ6: Inter-slice shift (1 - SSIM)
    from skimage.metrics import structural_similarity
    adj_ssims = []
    for i in range(0, N - 1, max(1, N // 20)):
        s = structural_similarity(qd_all[i], qd_all[i + 1], data_range=1.0,
                                  win_size=7, gaussian_weights=True, sigma=1.5)
        adj_ssims.append(s)
    tau6 = float(1.0 - np.mean(adj_ssims)) if adj_ssims else 0.0

    # τ7: δ/σ
    adj_deltas = []
    for i in range(N - 1):
        adj_deltas.append(np.mean(np.abs(fd_all[i] - fd_all[i + 1])))
    delta_mean = float(np.mean(adj_deltas)) if adj_deltas else 0.0
    sigma_global = float(np.std(noise))
    tau7 = delta_mean / sigma_global if sigma_global > 1e-8 else 0.0

    return {
        'tau1_cv_local_sigma': tau1,
        'tau3_autocorrelation': tau3,
        'tau5_skewness': tau5,
        'tau6_interslice_shift': tau6,
        'tau7_delta_over_sigma': tau7,
        'sigma_global': sigma_global,
        'n_slices': int(N),
    }


# ═════════════════════════════════════════════════════════════════════════════
# PART 2: Controlled distortion
# ═════════════════════════════════════════════════════════════════════════════

def inject_awgn(fd_slices, sigma):
    """Inject i.i.d. AWGN — τ1≈0, τ3≈0 expected."""
    rng = np.random.RandomState(42)
    noise = rng.normal(0, sigma, fd_slices.shape).astype(np.float32)
    return fd_slices + noise


def inject_correlated_noise(fd_slices, sigma, correlation_sigma):
    """Inject spatially correlated noise (Gaussian-filtered AWGN).
    correlation_sigma controls the blur kernel size → higher = more τ3."""
    rng = np.random.RandomState(42)
    noise = rng.normal(0, sigma, fd_slices.shape).astype(np.float32)
    corr_noise = np.zeros_like(noise)
    for i in range(noise.shape[0]):
        filtered = gaussian_filter(noise[i], sigma=correlation_sigma)
        # Rescale to maintain same global σ
        corr_noise[i] = filtered * (sigma / (filtered.std() + 1e-10))
    return fd_slices + corr_noise


def inject_nonstationary_noise(fd_slices, sigma, modulation_strength):
    """Inject non-stationary noise (AWGN × spatial mask).
    modulation_strength controls how much σ varies spatially → higher = more τ1."""
    rng = np.random.RandomState(42)
    H, W = fd_slices.shape[1], fd_slices.shape[2]
    # Create smooth spatial modulation mask: 1 ± modulation_strength
    y_grid = np.linspace(-1, 1, H)
    x_grid = np.linspace(-1, 1, W)
    Y, X = np.meshgrid(y_grid, x_grid, indexing='ij')
    mask = 1.0 + modulation_strength * (np.sin(2 * np.pi * Y) * np.cos(2 * np.pi * X))
    mask = mask.astype(np.float32)

    noise = rng.normal(0, sigma, fd_slices.shape).astype(np.float32)
    modulated = noise * mask[np.newaxis, :, :]
    return fd_slices + modulated


def run_controlled_distortion(fd_slices, real_sigma):
    """Run all three controlled distortion experiments."""
    results = {}

    # (a) Pure AWGN — baseline
    print("\n    (a) i.i.d. AWGN (baseline)...")
    noisy = inject_awgn(fd_slices, real_sigma)
    results['awgn'] = compute_tau_quick(noisy, fd_slices)

    # (b) Correlated noise — sweep correlation widths
    for corr_s in [1.0, 2.0, 4.0, 8.0]:
        label = f'correlated_sigma{corr_s}'
        print(f"    (b) Correlated noise (blur σ={corr_s})...")
        noisy = inject_correlated_noise(fd_slices, real_sigma, corr_s)
        results[label] = compute_tau_quick(noisy, fd_slices)

    # (c) Non-stationary noise — sweep modulation
    for mod in [0.25, 0.5, 0.75, 1.0]:
        label = f'nonstationary_mod{mod}'
        print(f"    (c) Non-stationary noise (mod={mod})...")
        noisy = inject_nonstationary_noise(fd_slices, real_sigma, mod)
        results[label] = compute_tau_quick(noisy, fd_slices)

    return results


def main():
    t_global = time.perf_counter()

    # ═══════════════════════════════════════════════════════════════════════
    # PART 1: 1mm B30 diagnostics
    # ═══════════════════════════════════════════════════════════════════════
    print("=" * 70)
    print("  PART 1: τ1–τ7 DIAGNOSTICS ON 1mm B30 (cross-protocol)")
    print("=" * 70)

    pairs_1mm = collect_paired_paths(DATA_ROOT, kernel='1mm B30')
    print(f"  Total 1mm pairs found: {len(pairs_1mm)}")

    results_1mm = {}
    for pat in ALL_PATIENTS:
        t0 = time.perf_counter()
        print(f"\n  Patient {pat}...", end=" ", flush=True)
        qd, fd = load_patient_data(pat, pairs_1mm)
        tau = compute_tau_quick(qd, fd)
        results_1mm[pat] = tau
        print(f"({qd.shape[0]} slices, {time.perf_counter()-t0:.1f}s) "
              f"τ1={tau['tau1_cv_local_sigma']:.3f} τ3={tau['tau3_autocorrelation']:.3f} "
              f"τ5={tau['tau5_skewness']:.3f} τ6={tau['tau6_interslice_shift']:.3f} "
              f"τ7={tau['tau7_delta_over_sigma']:.2f}")

    # ═══════════════════════════════════════════════════════════════════════
    # PART 2: Controlled distortion validation
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  PART 2: CONTROLLED DISTORTION VALIDATION")
    print("=" * 70)

    # Use 3mm B30 FD images from 3 patients as calibration set
    pairs_3mm = collect_paired_paths(DATA_ROOT, kernel='3mm B30')
    cal_patients = ['L067', 'L096', 'L506']

    results_controlled = {}
    for pat in cal_patients:
        print(f"\n  Patient {pat} (controlled distortion on FD):")
        qd, fd = load_patient_data(pat, pairs_3mm)
        real_noise = qd - fd
        real_sigma = float(np.std(real_noise))
        print(f"    Real noise σ = {real_sigma:.2f} HU")
        results_controlled[pat] = run_controlled_distortion(fd, real_sigma)

    # ═══════════════════════════════════════════════════════════════════════
    # Load existing 3mm results for comparison
    # ═══════════════════════════════════════════════════════════════════════
    ref_3mm_path = OUT_DIR / 'diagnostics_per_patient_v2_hu_soft_tissue.json'
    results_3mm = {}
    if ref_3mm_path.exists():
        with open(ref_3mm_path) as f:
            results_3mm = json.load(f)

    # ═══════════════════════════════════════════════════════════════════════
    # SUMMARY TABLES
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  COMPARISON: 3mm B30 vs 1mm B30")
    print("=" * 70)
    print(f"  {'Patient':>8} │ {'τ1(3mm)':>8} {'τ1(1mm)':>8} │ {'τ3(3mm)':>8} {'τ3(1mm)':>8} │ "
          f"{'τ6(3mm)':>8} {'τ6(1mm)':>8} │ {'τ7(3mm)':>8} {'τ7(1mm)':>8}")
    print(f"  {'─'*8} ┼ {'─'*8} {'─'*8} ┼ {'─'*8} {'─'*8} ┼ {'─'*8} {'─'*8} ┼ {'─'*8} {'─'*8}")

    t1_3mm, t1_1mm = [], []
    t3_3mm, t3_1mm = [], []
    t6_3mm, t6_1mm = [], []
    t7_3mm, t7_1mm = [], []

    for pat in ALL_PATIENTS:
        r3 = results_3mm.get(pat, {})
        r1 = results_1mm.get(pat, {})
        v1_3 = r3.get('tau1_cv_local_sigma', float('nan'))
        v1_1 = r1.get('tau1_cv_local_sigma', float('nan'))
        v3_3 = r3.get('tau3_autocorrelation', float('nan'))
        v3_1 = r1.get('tau3_autocorrelation', float('nan'))
        v6_3 = r3.get('tau6_interslice_shift', float('nan'))
        v6_1 = r1.get('tau6_interslice_shift', float('nan'))
        v7_3 = r3.get('tau7_delta_over_sigma', float('nan'))
        v7_1 = r1.get('tau7_delta_over_sigma', float('nan'))

        t1_3mm.append(v1_3); t1_1mm.append(v1_1)
        t3_3mm.append(v3_3); t3_1mm.append(v3_1)
        t6_3mm.append(v6_3); t6_1mm.append(v6_1)
        t7_3mm.append(v7_3); t7_1mm.append(v7_1)

        print(f"  {pat:>8} │ {v1_3:8.3f} {v1_1:8.3f} │ {v3_3:8.3f} {v3_1:8.3f} │ "
              f"{v6_3:8.3f} {v6_1:8.3f} │ {v7_3:8.2f} {v7_1:8.2f}")

    print(f"  {'─'*8} ┼ {'─'*8} {'─'*8} ┼ {'─'*8} {'─'*8} ┼ {'─'*8} {'─'*8} ┼ {'─'*8} {'─'*8}")
    print(f"  {'MEAN':>8} │ {np.nanmean(t1_3mm):8.3f} {np.nanmean(t1_1mm):8.3f} │ "
          f"{np.nanmean(t3_3mm):8.3f} {np.nanmean(t3_1mm):8.3f} │ "
          f"{np.nanmean(t6_3mm):8.3f} {np.nanmean(t6_1mm):8.3f} │ "
          f"{np.nanmean(t7_3mm):8.2f} {np.nanmean(t7_1mm):8.2f}")

    # Physics predictions
    print("\n  Physics predictions:")
    d_tau1 = np.nanmean(t1_1mm) - np.nanmean(t1_3mm)
    d_tau3 = np.nanmean(t3_1mm) - np.nanmean(t3_3mm)
    d_tau6 = np.nanmean(t6_1mm) - np.nanmean(t6_3mm)
    print(f"    τ1 (non-stationarity): Δ = {d_tau1:+.3f}  (expected: ↑ more noise)")
    print(f"    τ3 (autocorrelation):  Δ = {d_tau3:+.3f}  (expected: ≈0 ramp filter invariant)")
    print(f"    τ6 (inter-slice):      Δ = {d_tau6:+.3f}  (expected: ↓ thinner slices)")

    # Controlled distortion summary
    print("\n" + "=" * 70)
    print("  CONTROLLED DISTORTION CALIBRATION")
    print("=" * 70)
    print(f"  {'Condition':>25} │ {'τ1':>6} {'τ3':>6} {'τ5':>6} │ Notes")
    print(f"  {'─'*25} ┼ {'─'*6} {'─'*6} {'─'*6} ┼ {'─'*30}")

    # Average across calibration patients
    conditions = ['awgn',
                  'correlated_sigma1.0', 'correlated_sigma2.0',
                  'correlated_sigma4.0', 'correlated_sigma8.0',
                  'nonstationary_mod0.25', 'nonstationary_mod0.5',
                  'nonstationary_mod0.75', 'nonstationary_mod1.0']

    for cond in conditions:
        t1s, t3s, t5s = [], [], []
        for pat in cal_patients:
            r = results_controlled[pat].get(cond, {})
            t1s.append(r.get('tau1_cv_local_sigma', float('nan')))
            t3s.append(r.get('tau3_autocorrelation', float('nan')))
            t5s.append(r.get('tau5_skewness', float('nan')))
        t1m, t3m, t5m = np.nanmean(t1s), np.nanmean(t3s), np.nanmean(t5s)

        if cond == 'awgn':
            note = "baseline — should be ≈0"
        elif 'correlated' in cond:
            note = f"τ3 should ↑"
        else:
            note = f"τ1 should ↑"
        print(f"  {cond:>25} │ {t1m:6.3f} {t3m:6.3f} {t5m:6.3f} │ {note}")

    # Print real Mayo 3mm for reference
    if results_3mm:
        t1r = np.nanmean([results_3mm[p]['tau1_cv_local_sigma'] for p in ALL_PATIENTS])
        t3r = np.nanmean([results_3mm[p]['tau3_autocorrelation'] for p in ALL_PATIENTS])
        t5r = np.nanmean([results_3mm[p]['tau5_skewness'] for p in ALL_PATIENTS])
        print(f"  {'─'*25} ┼ {'─'*6} {'─'*6} {'─'*6} ┼ {'─'*30}")
        print(f"  {'REAL Mayo 3mm B30':>25} │ {t1r:6.3f} {t3r:6.3f} {t5r:6.3f} │ actual clinical CT")

    # Save all results
    output = {
        '1mm_B30_diagnostics': results_1mm,
        'controlled_distortion': results_controlled,
        'comparison_3mm_vs_1mm': {
            'delta_tau1_mean': float(d_tau1),
            'delta_tau3_mean': float(d_tau3),
            'delta_tau6_mean': float(d_tau6),
        }
    }
    out_path = OUT_DIR / 'diagnostics_validation_1mm_and_controlled.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=lambda x: float(x) if hasattr(x, '__float__') else str(x))

    elapsed = time.perf_counter() - t_global
    print(f"\n  Total time: {elapsed:.1f}s ({elapsed/60:.1f} min) — CPU only")
    print(f"  Saved: {out_path}")


if __name__ == '__main__':
    main()
