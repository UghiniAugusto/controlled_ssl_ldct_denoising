#!/usr/bin/env python3
"""
Compute τ1–τ7 diagnostic tests PER PATIENT (CPU-only).
Results saved to loocv_results/diagnostics_per_patient.json

This provides W2 data: 10 diagnostic severity scores → scatter plot
with 10× more data points than the original n=3.
"""
import sys, os, json
import numpy as np
from pathlib import Path
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from dataset import collect_paired_paths, load_ima, window_normalize
from diagnostics_v2 import compute_tau2_gaussianity, compute_tau4_heteroscedasticity

# ═════════════════════════════════════════════════════════════════════════════
DATA_ROOT = os.environ.get('MAYO_DATA_ROOT', './data/Traning_Image_Data')
KERNEL = '3mm B30'
ALL_PATIENTS = ['L067', 'L096', 'L109', 'L143', 'L192',
                'L286', 'L291', 'L310', 'L333', 'L506']
OUT_DIR = Path(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results'))
MAX_SLICES = 50  # subsample for speed (~5 min/patient instead of 20)
DOMAIN = 'hu_soft_tissue'  # 'normalized_01' (deprecated), 'hu_soft_tissue' (new), 'hu_full'


def _to_domain(slices_hu: np.ndarray, domain: str) -> np.ndarray:
    """Return slices in the chosen analysis domain.

    'normalized_01'   : legacy window+clip, kept for traceability only.
    'hu_soft_tissue'  : raw HU, soft-tissue window applied as a *mask*
                         rather than a clip (windows outside [-160, 240]
                         are excluded by τ4's min/max_mean_hu instead of
                         being squashed to 0/1).
    'hu_full'         : raw HU, no mask (for sanity comparison).
    """
    if domain == 'normalized_01':
        low, high = 40.0 - 200.0, 40.0 + 200.0
        return np.clip((slices_hu - low) / (high - low), 0.0, 1.0).astype(np.float32)
    elif domain in ('hu_soft_tissue', 'hu_full'):
        return slices_hu.astype(np.float32)
    raise ValueError(f"unknown domain: {domain}")


def load_patient_data(patient_id, all_pairs, max_slices=MAX_SLICES):
    """Load QD and FD slices for one patient."""
    pairs = [(q, f) for q, f in all_pairs if patient_id in q]
    if len(pairs) > max_slices:
        idx = np.linspace(0, len(pairs)-1, max_slices, dtype=int)
        pairs = [pairs[i] for i in idx]

    qd_imgs, fd_imgs = [], []
    for qf, ff in pairs:
        qd_raw = load_ima(qf)
        fd_raw = load_ima(ff)
        qd_imgs.append(_to_domain(qd_raw, DOMAIN))
        fd_imgs.append(_to_domain(fd_raw, DOMAIN))
    return np.array(qd_imgs), np.array(fd_imgs)


def compute_tau_tests(qd_all, fd_all):
    """Compute τ1–τ7 for one patient."""
    noise = qd_all - fd_all
    N, H, W = noise.shape

    # τ1: CV of local noise σ (non-stationarity)
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

    # τ2: Gaussianity as effect size (excess kurtosis + Anderson-Darling).
    # Replaces Shapiro-Wilk p-value, which (a) is not an effect size and
    # (b) was computed on patch[:50] — 50 of 1024 samples, crippling power.
    # We now use all 1024 samples of each patch; aggregate by median across
    # patches (same robust aggregation as the old code).
    rng = np.random.RandomState(42)
    t2_k_excess, t2_kz, t2_ad_adj = [], [], []
    # deprecated — kept for rebuttal traceability (do NOT cite):
    sw_pvals = []
    for _ in range(200):
        i = rng.randint(0, N)
        y, x = rng.randint(0, H-32), rng.randint(0, W-32)
        patch = noise[i, y:y+32, x:x+32].ravel()
        if np.std(patch) > 1e-6:
            # new metrics on full patch
            try:
                t2 = compute_tau2_gaussianity(patch)
                t2_k_excess.append(t2.excess_kurtosis)
                t2_kz.append(t2.kurtosis_z)
                t2_ad_adj.append(t2.anderson_darling_adj)
            except ValueError:
                pass
            # old metric on 50-sample truncation (preserved verbatim)
            _, p = stats.shapiro(patch[:50])
            sw_pvals.append(p)
    tau2_k_excess = float(np.median(t2_k_excess)) if t2_k_excess else 0.0
    tau2_kz = float(np.median(t2_kz)) if t2_kz else 0.0
    tau2_ad_adj = float(np.median(t2_ad_adj)) if t2_ad_adj else 0.0
    tau2 = float(np.median(sw_pvals)) if sw_pvals else 0.0  # deprecated

    # τ3: Spatial autocorrelation at lag=1
    autocorrs = []
    for i in range(0, N, max(1, N // 20)):
        n_row = noise[i, H // 2, :]
        if np.std(n_row) > 1e-6:
            r = np.corrcoef(n_row[:-1], n_row[1:])[0, 1]
            autocorrs.append(r)
    tau3 = float(np.mean(autocorrs)) if autocorrs else 0.0

    # τ4: Heteroscedasticity coefficient α from σ² ∝ μ^α.
    # Replaces Pearson corr(|signal|, |noise|), which measures linear
    # dependence in the mean — orthogonal to the actual CT mechanism
    # (signal-dependent variance). Per-slice α aggregated by median.
    t4_alpha_ols, t4_alpha_ts, t4_r2, t4_bp = [], [], [], []
    # deprecated — kept for rebuttal traceability (do NOT cite):
    corrs = []
    # Domain-specific HU window for τ4
    min_mu = -160.0 if DOMAIN == 'hu_soft_tissue' else None
    max_mu = +240.0 if DOMAIN == 'hu_soft_tissue' else None
    for i in range(0, N, max(1, N // 20)):
        # new metric
        try:
            t4 = compute_tau4_heteroscedasticity(
                signal=fd_all[i],
                residual=noise[i],
                window_size=32,
                gradient_percentile=80.0,
                min_mean_hu=min_mu,
                max_mean_hu=max_mu,
            )
            t4_alpha_ols.append(t4.alpha_ols)
            t4_alpha_ts.append(t4.alpha_theilsen)
            t4_r2.append(t4.r_squared)
            t4_bp.append(t4.breusch_pagan_stat)
        except (ValueError, RuntimeError):
            pass
        # old metric (preserved verbatim)
        sig = np.abs(fd_all[i].ravel())
        noi = np.abs(noise[i].ravel())
        mask = sig > 0.01
        if mask.sum() > 100:
            r = np.corrcoef(sig[mask], noi[mask])[0, 1]
            corrs.append(r)
    tau4_alpha_ols = float(np.median(t4_alpha_ols)) if t4_alpha_ols else 0.0
    tau4_alpha_ts = float(np.median(t4_alpha_ts)) if t4_alpha_ts else 0.0
    tau4_r2 = float(np.median(t4_r2)) if t4_r2 else 0.0
    tau4_bp = float(np.median(t4_bp)) if t4_bp else 0.0
    tau4 = float(np.mean(corrs)) if corrs else 0.0  # deprecated

    # τ5: Noise symmetry (|skewness|)
    skews = []
    for i in range(0, N, max(1, N // 20)):
        s = float(stats.skew(noise[i].ravel()))
        skews.append(s)
    tau5 = float(np.mean(np.abs(skews))) if skews else 0.0

    # τ6: Inter-slice structural shift (1 - SSIM adjacent)
    from skimage.metrics import structural_similarity
    adj_ssims = []
    for i in range(0, N - 1, max(1, N // 20)):
        s = structural_similarity(qd_all[i], qd_all[i + 1], data_range=1.0,
                                  win_size=7, gaussian_weights=True, sigma=1.5)
        adj_ssims.append(s)
    tau6 = float(1.0 - np.mean(adj_ssims)) if adj_ssims else 0.0

    # τ7: Registration residual δ/σ
    # δ = mean absolute difference between adjacent FD slices (structural shift)
    # σ = global noise standard deviation
    adj_deltas = []
    for i in range(N - 1):
        d = np.mean(np.abs(fd_all[i] - fd_all[i + 1]))
        adj_deltas.append(d)
    delta_mean = float(np.mean(adj_deltas)) if adj_deltas else 0.0
    sigma_global = float(np.std(noise))
    delta_over_sigma = delta_mean / sigma_global if sigma_global > 1e-8 else 0.0

    # Composite severity score: mean normalized violations
    # Normalize each metric by a reference (approximate "severe" threshold)
    # τ1 > 0.2, τ3 > 0.1, τ5 > 0.1, τ6 > 0.2
    # Note: τ7 fixed — now uses δ = structural shift between adjacent FD slices
    v1 = (tau1 - 0.0) / 0.2  # CV of σ
    v3 = (tau3 - 0.0) / 0.1  # autocorrelation
    v5 = (tau5 - 0.0) / 0.1  # skewness
    v6 = (tau6 - 0.0) / 0.2  # inter-slice shift
    # Clip negative values to 0 (no credit for better than threshold)
    severity_composite_robust = np.mean([max(0, v1), max(0, v3), max(0, v5), max(0, v6)])

    # Composite severity score (mean of normalized violations)
    # Original uses deprecated τ2/τ4; v2 uses refactored metrics
    severity_v1 = np.mean([
        min(tau1, 1.0),       # CV: 0=stationary, 1=very non-stationary
        1.0 - min(tau2, 1.0), # Gaussianity: low p = violation (deprecated)
        min(abs(tau3), 1.0),  # Autocorrelation: 0=independent
        min(abs(tau4), 1.0),  # Signal-noise dep (deprecated)
        min(tau5, 1.0),       # Skewness
        min(tau6, 1.0),       # Inter-slice shift
        min(delta_over_sigma / 5.0, 1.0), # δ/σ normalized
    ])
    severity_v2 = np.mean([
        min(tau1, 1.0),
        min(abs(tau2_kz) / 3.0, 1.0),  # |z|/3: z>3 is strong non-Gaussian
        min(abs(tau3), 1.0),
        min(abs(tau4_alpha_ols) / 1.0, 1.0),  # α>1 is super-Poisson
        min(tau5, 1.0),
        min(tau6, 1.0),
        min(delta_over_sigma / 5.0, 1.0),
    ])

    return {
        # --- primary metrics (v2) ---
        'tau1_cv_local_sigma': float(tau1),
        'tau2_excess_kurtosis': float(tau2_k_excess),
        'tau2_kurtosis_z': float(tau2_kz),
        'tau2_anderson_darling_adj': float(tau2_ad_adj),
        'tau3_autocorrelation': float(tau3),
        'tau4_alpha_ols': float(tau4_alpha_ols),
        'tau4_alpha_theilsen': float(tau4_alpha_ts),
        'tau4_r_squared': float(tau4_r2),
        'tau4_breusch_pagan': float(tau4_bp),
        'tau5_skewness': float(tau5),
        'tau6_interslice_shift': float(tau6),
        'tau7_delta_over_sigma': float(delta_over_sigma),
        'sigma_global': float(sigma_global),
        'n_slices': int(N),
        'severity_composite': float(severity_v1),  # deprecated (uses old τ2/τ4)
        'severity_composite_v2': float(severity_v2),  # new (uses refactored τ2/τ4)
        'severity_composite_robust': float(severity_composite_robust),  # excludes τ2/τ4
        # --- deprecated (kept for rebuttal traceability; do NOT cite) ---
        '_deprecated_tau2_shapiro_p': float(tau2),
        '_deprecated_tau4_pearson_r': float(tau4),
    }


def main():
    print("="*70)
    print("  DIAGNOSTIC TESTS τ1–τ7 PER PATIENT (CPU-only)")
    print("="*70)

    all_pairs = collect_paired_paths(DATA_ROOT, kernel=KERNEL)
    print(f"  Total pairs found: {len(all_pairs)}")

    results = {}
    for pat in ALL_PATIENTS:
        print(f"\n  {'─'*50}")
        print(f"  Patient {pat}")
        print(f"  {'─'*50}")

        qd, fd = load_patient_data(pat, all_pairs)
        print(f"  Loaded {len(qd)} slices ({qd.shape})")

        tau = compute_tau_tests(qd, fd)
        results[pat] = tau

        print(f"  τ1 CV(σ)={tau['tau1_cv_local_sigma']:.3f}  "
              f"τ2 k_ex={tau['tau2_excess_kurtosis']:+.3f} z={tau['tau2_kurtosis_z']:.2f} A²={tau['tau2_anderson_darling_adj']:.2f}")
        print(f"  τ4 α_ols={tau['tau4_alpha_ols']:.3f} α_ts={tau['tau4_alpha_theilsen']:.3f} R²={tau['tau4_r_squared']:.3f} BP={tau['tau4_breusch_pagan']:.2f}")
        print(f"  τ3 Autocorr={tau['tau3_autocorrelation']:.3f}  "
              f"τ5 Skew={tau['tau5_skewness']:.3f}  "
              f"τ6 Shift={tau['tau6_interslice_shift']:.3f}  "
              f"τ7 δ/σ={tau['tau7_delta_over_sigma']:.2f}")
        print(f"  Severity v1={tau['severity_composite']:.3f}  v2={tau['severity_composite_v2']:.3f}  robust={tau['severity_composite_robust']:.3f}")
        print(f"  (deprecated: τ2 p={tau['_deprecated_tau2_shapiro_p']:.4f}  τ4 r={tau['_deprecated_tau4_pearson_r']:.3f})")

    # Summary table (robust metrics: τ1, τ3, τ5, τ6 only)
    print(f"\n{'='*70}")
    print(f"  SUMMARY TABLE (robust: τ1, τ3, τ5, τ6)")
    print(f"{'='*70}")
    print(f"  {'Patient':>8} │ {'τ1':>6} {'τ3':>6} {'τ5':>6} {'τ6':>6} │ {'SevRob':>6}")
    print(f"  {'─'*8} ┼ {'─'*6} {'─'*6} {'─'*6} {'─'*6} ┼ {'─'*6}")
    for pat in ALL_PATIENTS:
        t = results[pat]
        print(f"  {pat:>8} │ {t['tau1_cv_local_sigma']:6.3f} "
              f"{t['tau3_autocorrelation']:6.3f} "
              f"{t['tau5_skewness']:6.3f} {t['tau6_interslice_shift']:6.3f} │ {t['severity_composite_robust']:6.3f}")

    # Save with versioning based on domain
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    if DOMAIN == 'normalized_01':
        out_path = OUT_DIR / 'diagnostics_per_patient_v2_normalized.json'
    elif DOMAIN == 'hu_soft_tissue':
        out_path = OUT_DIR / 'diagnostics_per_patient_v2_hu_soft_tissue.json'
    elif DOMAIN == 'hu_full':
        out_path = OUT_DIR / 'diagnostics_per_patient_v2_hu_full.json'
    else:
        out_path = OUT_DIR / f'diagnostics_per_patient_{DOMAIN}.json'
    
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {out_path}")
    print(f"  Domain: {DOMAIN}")


if __name__ == '__main__':
    main()
