#!/usr/bin/env python3
"""
Publication figures + statistical tests for BRACIS 2026 paper.
All CPU — does NOT interfere with GPU training.

Outputs:
  figures/fig_boxplot_psnr_ssim.pdf
  figures/fig_controlled_distortion.pdf
  figures/fig_3mm_vs_1mm.pdf
  figures/fig_convergence_n2np.pdf
  stats/wilcoxon_tests.txt
"""
import json, os, sys
import numpy as np
from pathlib import Path
from scipy import stats as sp_stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
RESULTS = Path(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results'))
FIG_DIR = Path(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'figures'))
STAT_DIR = Path(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'stats'))
FIG_DIR.mkdir(exist_ok=True)
STAT_DIR.mkdir(exist_ok=True)

PATIENTS = ['L067','L096','L109','L143','L192','L286','L291','L310','L333','L506']
METHODS = ['supervised', 'n2n', 'nei2nei', 'noisier2noise']
LABELS = {'supervised': 'Supervised', 'n2n': 'Noise2Noise', 'nei2nei': 'Neighbor2Neighbor',
          'noisier2noise': 'Noisier2Noise', 'bm3d': 'BM3D', 'input': 'Input (QD)'}
COLORS = {'supervised': '#2196F3', 'n2n': '#4CAF50', 'nei2nei': '#FF9800',
          'noisier2noise': '#F44336', 'bm3d': '#9E9E9E', 'input': '#BDBDBD'}

# Publication style
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════
def load_loocv_results():
    """Load all completed LOO-CV results."""
    data = {}
    for m in METHODS:
        data[m] = {'psnr': {}, 'ssim': {}}
        for p in PATIENTS:
            rpath = RESULTS / f'fold_{p}' / m / 'result.json'
            if rpath.exists():
                with open(rpath) as f:
                    r = json.load(f)
                data[m]['psnr'][p] = r['best_psnr']
                best_entry = max(r['log'], key=lambda x: x['psnr'])
                data[m]['ssim'][p] = best_entry['ssim']
    return data


def load_bm3d():
    bm3d_path = RESULTS / 'bm3d_results.json.backup'
    if not bm3d_path.exists():
        return None
    with open(bm3d_path) as f:
        bm3d = json.load(f)
    return {p: {'psnr': bm3d[p]['bm3d_psnr'], 'ssim': bm3d[p]['bm3d_ssim'],
                'input_psnr': bm3d[p]['input_psnr']}
            for p in PATIENTS if p in bm3d}


def load_controlled_distortion():
    path = RESULTS / 'diagnostics_validation_1mm_and_controlled.json'
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_diagnostics_3mm():
    path = RESULTS / 'diagnostics_per_patient_v2_hu_soft_tissue.json'
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Boxplot PSNR + SSIM (all methods + BM3D + input)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_boxplot(data, bm3d):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.5))

    # Determine common patients (completed in all 4 methods)
    common = set(PATIENTS)
    for m in METHODS:
        common &= set(data[m]['psnr'].keys())
    common = sorted(common)
    n = len(common)

    # Build arrays: input, bm3d, noisier2noise, nei2nei, n2n, supervised
    order = ['input', 'noisier2noise', 'bm3d', 'nei2nei', 'n2n', 'supervised']
    psnr_data, ssim_data, box_labels, box_colors = [], [], [], []

    for m in order:
        if m == 'input' and bm3d:
            vals_p = [bm3d[p]['input_psnr'] for p in common]
            vals_s = [0.0] * n  # no SSIM for input in BM3D results
        elif m == 'bm3d' and bm3d:
            vals_p = [bm3d[p]['psnr'] for p in common]
            vals_s = [bm3d[p]['ssim'] for p in common]
        elif m in data:
            vals_p = [data[m]['psnr'][p] for p in common]
            vals_s = [data[m]['ssim'][p] for p in common]
        else:
            continue
        psnr_data.append(vals_p)
        ssim_data.append(vals_s)
        box_labels.append(LABELS[m])
        box_colors.append(COLORS[m])

    # PSNR boxplot
    bp1 = ax1.boxplot(psnr_data, labels=box_labels, patch_artist=True,
                       widths=0.6, showmeans=True, meanprops=dict(marker='D', markersize=4, markerfacecolor='black'))
    for patch, color in zip(bp1['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax1.set_ylabel('PSNR (dB)')
    ax1.set_title(f'(a) PSNR per fold (n={n})')
    ax1.tick_params(axis='x', rotation=30)
    ax1.grid(axis='y', alpha=0.3)

    # SSIM boxplot (skip input which has no SSIM)
    ssim_filtered = [s for s, l in zip(ssim_data, box_labels) if l != 'Input (QD)']
    labels_filtered = [l for l in box_labels if l != 'Input (QD)']
    colors_filtered = [c for c, l in zip(box_colors, box_labels) if l != 'Input (QD)']

    bp2 = ax2.boxplot(ssim_filtered, labels=labels_filtered, patch_artist=True,
                       widths=0.6, showmeans=True, meanprops=dict(marker='D', markersize=4, markerfacecolor='black'))
    for patch, color in zip(bp2['boxes'], colors_filtered):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax2.set_ylabel('SSIM')
    ax2.set_title(f'(b) SSIM per fold (n={n})')
    ax2.tick_params(axis='x', rotation=30)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out = FIG_DIR / 'fig_boxplot_psnr_ssim.pdf'
    fig.savefig(out)
    fig.savefig(out.with_suffix('.png'))
    plt.close()
    print(f'  ✓ {out}')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: Controlled distortion calibration (τ1, τ3 response curves)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_controlled_distortion(cd_data):
    if cd_data is None:
        print('  ✗ No controlled distortion data')
        return

    cal_patients = [p for p in cd_data.get('controlled_distortion', {})]
    if not cal_patients:
        print('  ✗ No controlled distortion patients')
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.2))

    # (a) Correlation sweep: τ3 as function of blur σ
    corr_sigmas = [0, 1.0, 2.0, 4.0, 8.0]
    corr_labels = ['awgn', 'correlated_sigma1.0', 'correlated_sigma2.0',
                   'correlated_sigma4.0', 'correlated_sigma8.0']

    tau3_means, tau1_means_corr = [], []
    for cond in corr_labels:
        t3s = [cd_data['controlled_distortion'][p][cond]['tau3_autocorrelation'] for p in cal_patients]
        t1s = [cd_data['controlled_distortion'][p][cond]['tau1_cv_local_sigma'] for p in cal_patients]
        tau3_means.append(np.mean(t3s))
        tau1_means_corr.append(np.mean(t1s))

    ax1.plot(corr_sigmas, tau3_means, 'o-', color='#E91E63', linewidth=2, markersize=6, label='τ₃ (autocorrelation)')
    ax1.plot(corr_sigmas, tau1_means_corr, 's--', color='#2196F3', linewidth=1.5, markersize=5, label='τ₁ (non-stationarity)')
    ax1.axhline(y=0.61, color='#E91E63', linestyle=':', alpha=0.5, label='Mayo 3mm τ₃ = 0.61')
    ax1.axhline(y=0.40, color='#2196F3', linestyle=':', alpha=0.5, label='Mayo 3mm τ₁ = 0.40')
    ax1.set_xlabel('Correlation kernel σ (pixels)')
    ax1.set_ylabel('Diagnostic value')
    ax1.set_title('(a) Spatial correlation sweep')
    ax1.legend(fontsize=7, loc='center right')
    ax1.grid(alpha=0.3)
    ax1.set_ylim(-0.05, 1.1)

    # (b) Non-stationarity sweep: τ1 as function of modulation
    mod_strengths = [0, 0.25, 0.5, 0.75, 1.0]
    mod_labels = ['awgn', 'nonstationary_mod0.25', 'nonstationary_mod0.5',
                  'nonstationary_mod0.75', 'nonstationary_mod1.0']

    tau1_means, tau3_means_ns = [], []
    for cond in mod_labels:
        t1s = [cd_data['controlled_distortion'][p][cond]['tau1_cv_local_sigma'] for p in cal_patients]
        t3s = [cd_data['controlled_distortion'][p][cond]['tau3_autocorrelation'] for p in cal_patients]
        tau1_means.append(np.mean(t1s))
        tau3_means_ns.append(np.mean(t3s))

    ax2.plot(mod_strengths, tau1_means, 's-', color='#2196F3', linewidth=2, markersize=6, label='τ₁ (non-stationarity)')
    ax2.plot(mod_strengths, tau3_means_ns, 'o--', color='#E91E63', linewidth=1.5, markersize=5, label='τ₃ (autocorrelation)')
    ax2.axhline(y=0.40, color='#2196F3', linestyle=':', alpha=0.5, label='Mayo 3mm τ₁ = 0.40')
    ax2.axhline(y=0.61, color='#E91E63', linestyle=':', alpha=0.5, label='Mayo 3mm τ₃ = 0.61')
    ax2.set_xlabel('Modulation strength m')
    ax2.set_ylabel('Diagnostic value')
    ax2.set_title('(b) Non-stationarity sweep')
    ax2.legend(fontsize=7, loc='center right')
    ax2.grid(alpha=0.3)
    ax2.set_ylim(-0.05, 1.1)

    plt.tight_layout()
    out = FIG_DIR / 'fig_controlled_distortion.pdf'
    fig.savefig(out)
    fig.savefig(out.with_suffix('.png'))
    plt.close()
    print(f'  ✓ {out}')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3: 3mm vs 1mm comparison (τ per patient)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_3mm_vs_1mm(cd_data, diag_3mm):
    if cd_data is None or diag_3mm is None:
        print('  ✗ Missing 1mm or 3mm data')
        return

    diag_1mm = cd_data.get('1mm_B30_diagnostics', {})
    if not diag_1mm:
        print('  ✗ No 1mm diagnostics')
        return

    fig, axes = plt.subplots(1, 3, figsize=(7.5, 3.0))
    common = sorted(set(diag_3mm.keys()) & set(diag_1mm.keys()))
    x = np.arange(len(common))
    w = 0.35

    metrics = [
        ('tau1_cv_local_sigma', 'τ₁ (non-stationarity)', '(a)'),
        ('tau3_autocorrelation', 'τ₃ (autocorrelation)', '(b)'),
        ('tau7_delta_over_sigma', 'τ₇ (δ/σ)', '(c)'),
    ]

    for ax, (key, label, panel) in zip(axes, metrics):
        v3mm = [diag_3mm[p][key] for p in common]
        v1mm = [diag_1mm[p][key] for p in common]

        bars1 = ax.bar(x - w/2, v3mm, w, label='3 mm B30', color='#1976D2', alpha=0.8)
        bars2 = ax.bar(x + w/2, v1mm, w, label='1 mm B30', color='#FF7043', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([p[1:] for p in common], rotation=45, fontsize=7)
        ax.set_title(f'{panel} {label}')
        ax.grid(axis='y', alpha=0.3)
        if ax == axes[0]:
            ax.legend(fontsize=7)

    plt.tight_layout()
    out = FIG_DIR / 'fig_3mm_vs_1mm.pdf'
    fig.savefig(out)
    fig.savefig(out.with_suffix('.png'))
    plt.close()
    print(f'  ✓ {out}')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 4: Per-patient PSNR lines (method comparison)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_per_patient_lines(data, bm3d):
    common = set(PATIENTS)
    for m in METHODS:
        common &= set(data[m]['psnr'].keys())
    common = sorted(common)

    fig, ax = plt.subplots(figsize=(7.5, 3.5))
    x = np.arange(len(common))

    plot_order = ['supervised', 'n2n', 'nei2nei', 'bm3d', 'noisier2noise']
    markers = {'supervised': 'o', 'n2n': 's', 'nei2nei': '^', 'bm3d': 'D', 'noisier2noise': 'v'}

    for m in plot_order:
        if m == 'bm3d' and bm3d:
            vals = [bm3d[p]['psnr'] for p in common]
        elif m in data:
            vals = [data[m]['psnr'][p] for p in common]
        else:
            continue
        ax.plot(x, vals, f'{markers[m]}-', color=COLORS[m], label=LABELS[m],
                linewidth=1.5, markersize=5, alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(common, rotation=45)
    ax.set_ylabel('PSNR (dB)')
    ax.set_title('Per-patient PSNR across methods (LOO-CV)')
    ax.legend(fontsize=8, ncol=2, loc='upper left')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out = FIG_DIR / 'fig_per_patient_psnr.pdf'
    fig.savefig(out)
    fig.savefig(out.with_suffix('.png'))
    plt.close()
    print(f'  ✓ {out}')


# ═══════════════════════════════════════════════════════════════════════════════
# STATISTICAL TESTS
# ═══════════════════════════════════════════════════════════════════════════════
def run_statistical_tests(data, bm3d):
    lines = []
    lines.append('=' * 70)
    lines.append('  WILCOXON SIGNED-RANK TESTS (paired, two-sided)')
    lines.append('  LOO-CV PSNR & SSIM comparisons')
    lines.append('=' * 70)

    # Common patients across all methods
    common_all = set(PATIENTS)
    for m in METHODS:
        common_all &= set(data[m]['psnr'].keys())
    common_all = sorted(common_all)

    all_methods = list(METHODS)
    if bm3d:
        all_methods.append('bm3d')

    lines.append(f'\n  Common patients (all 4 DL methods): {common_all} (n={len(common_all)})')
    lines.append('')

    # Pairwise Wilcoxon for PSNR
    lines.append('  ── PSNR (dB) ──')
    lines.append(f'  {"Pair":>30} │ {"ΔMean":>7} {"W-stat":>7} {"p-value":>9} {"Signif":>8}')
    lines.append(f'  {"─"*30} ┼ {"─"*7} {"─"*7} {"─"*9} {"─"*8}')

    pairs = [
        ('supervised', 'n2n'),
        ('supervised', 'nei2nei'),
        ('supervised', 'noisier2noise'),
        ('n2n', 'nei2nei'),
        ('n2n', 'noisier2noise'),
        ('nei2nei', 'noisier2noise'),
    ]
    if bm3d:
        pairs += [('bm3d', 'noisier2noise'), ('n2n', 'bm3d'), ('nei2nei', 'bm3d')]

    for m1, m2 in pairs:
        # Determine sets of available patients for each method
        if m1 == 'bm3d':
            if not bm3d: continue
            keys1 = set(bm3d.keys())
        else:
            keys1 = set(data[m1]['psnr'].keys())
        if m2 == 'bm3d':
            if not bm3d: continue
            keys2 = set(bm3d.keys())
        else:
            keys2 = set(data[m2]['psnr'].keys())

        common = sorted(keys1 & keys2)
        v1 = [bm3d[p]['psnr'] for p in common] if m1 == 'bm3d' else [data[m1]['psnr'][p] for p in common]
        v2 = [bm3d[p]['psnr'] for p in common] if m2 == 'bm3d' else [data[m2]['psnr'][p] for p in common]

        if len(common) < 6:
            lines.append(f'  {LABELS[m1]+" vs "+LABELS[m2]:>30} │ n={len(common)} (insufficient)')
            continue

        diff = np.array(v1) - np.array(v2)
        mean_diff = np.mean(diff)

        try:
            w_stat, p_val = sp_stats.wilcoxon(v1, v2, alternative='two-sided')
            sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
            lines.append(f'  {LABELS[m1]+" vs "+LABELS[m2]:>30} │ {mean_diff:+7.2f} {w_stat:7.1f} {p_val:9.5f} {sig:>8}')
        except Exception as e:
            lines.append(f'  {LABELS[m1]+" vs "+LABELS[m2]:>30} │ {mean_diff:+7.2f} ERROR: {e}')

    # Summary table
    lines.append(f'\n\n  ── SUMMARY TABLE ──')
    lines.append(f'  {"Method":>20} │ {"n":>3} {"PSNR":>8} {"±std":>6} │ {"SSIM":>8} {"±std":>8}')
    lines.append(f'  {"─"*20} ┼ {"─"*3} {"─"*8} {"─"*6} ┼ {"─"*8} {"─"*8}')

    for m in ['supervised', 'n2n', 'nei2nei', 'noisier2noise', 'bm3d']:
        if m == 'bm3d' and bm3d:
            psnrs = [bm3d[p]['psnr'] for p in PATIENTS]
            ssims = [bm3d[p]['ssim'] for p in PATIENTS]
            n = len(psnrs)
        elif m in data:
            psnrs = list(data[m]['psnr'].values())
            ssims = list(data[m]['ssim'].values())
            n = len(psnrs)
        else:
            continue
        if n > 0:
            lines.append(f'  {LABELS[m]:>20} │ {n:3d} {np.mean(psnrs):8.2f} {np.std(psnrs):6.2f} │ {np.mean(ssims):8.4f} {np.std(ssims):8.4f}')

    # Input PSNR
    if bm3d:
        input_psnrs = [bm3d[p]['input_psnr'] for p in PATIENTS]
        lines.append(f'  {"Input (QD)":>20} │ {10:3d} {np.mean(input_psnrs):8.2f} {np.std(input_psnrs):6.2f} │ {"—":>8} {"—":>8}')

    # Ordering consistency
    lines.append(f'\n\n  ── ORDERING CONSISTENCY ──')
    for p in common_all:
        s = data['supervised']['psnr'][p]
        n2 = data['n2n']['psnr'][p]
        ne = data['nei2nei']['psnr'][p]
        np_ = data['noisier2noise']['psnr'][p]
        order_ok = s > n2 > ne > np_
        lines.append(f'  {p}: {s:.1f} > {n2:.1f} > {ne:.1f} > {np_:.1f}  {"✓" if order_ok else "✗ BROKEN"}')

    text = '\n'.join(lines)
    print(text)

    out = STAT_DIR / 'wilcoxon_tests.txt'
    with open(out, 'w') as f:
        f.write(text)
    print(f'\n  ✓ Saved: {out}')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 5: Spearman correlation heatmap (τ vs PSNR gap)
# ═══════════════════════════════════════════════════════════════════════════════
def fig_tau_gap_correlation(data, diag_3mm):
    if diag_3mm is None:
        print('  ✗ No 3mm diagnostics for correlation')
        return

    common = sorted(set(data['supervised']['psnr'].keys()) &
                    set(data['n2n']['psnr'].keys()) &
                    set(data['nei2nei']['psnr'].keys()) &
                    set(data['noisier2noise']['psnr'].keys()) &
                    set(diag_3mm.keys()))
    if len(common) < 6:
        print(f'  ✗ Only {len(common)} common patients (need ≥6)')
        return

    # Compute PSNR gaps: supervised - SSL
    ssl_methods = ['n2n', 'nei2nei', 'noisier2noise']
    ssl_labels = ['N2N gap', 'Nei2Nei gap', 'N2N⁺ gap']
    gaps = {}
    for m in ssl_methods:
        gaps[m] = [data['supervised']['psnr'][p] - data[m]['psnr'][p] for p in common]

    # Tau metrics
    tau_keys = [
        ('tau1_cv_local_sigma', 'τ₁ non-stat.'),
        ('tau3_autocorrelation', 'τ₃ autocorr.'),
        ('tau5_skewness', 'τ₅ skewness'),
        ('tau6_interslice_shift', 'τ₆ shift'),
        ('tau7_delta_over_sigma', 'τ₇ δ/σ'),
    ]
    tau_vals = {}
    for key, label in tau_keys:
        tau_vals[key] = [diag_3mm[p][key] for p in common]

    # Compute Spearman correlation matrix
    n_tau = len(tau_keys)
    n_ssl = len(ssl_methods)
    rho_matrix = np.zeros((n_tau, n_ssl))
    pval_matrix = np.zeros((n_tau, n_ssl))

    lines = []
    lines.append('=' * 70)
    lines.append('  SPEARMAN RANK CORRELATION: τ diagnostics vs PSNR gap')
    lines.append(f'  (gap = Supervised PSNR − SSL PSNR, n={len(common)} patients)')
    lines.append('=' * 70)
    lines.append(f'\n  {"Diagnostic":>20} │ {"N2N gap":>12} {"Nei2Nei gap":>12} {"N2N⁺ gap":>12}')
    lines.append(f'  {"─"*20} ┼ {"─"*12} {"─"*12} {"─"*12}')

    for i, (key, label) in enumerate(tau_keys):
        row = f'  {label:>20} │'
        for j, m in enumerate(ssl_methods):
            rho, pval = sp_stats.spearmanr(tau_vals[key], gaps[m])
            rho_matrix[i, j] = rho
            pval_matrix[i, j] = pval
            sig = '*' if pval < 0.05 else ''
            row += f' {rho:+5.2f} (p={pval:.3f}){sig:1s}'
        lines.append(row)

    # Effect size interpretation
    lines.append(f'\n  Interpretation: |ρ| > 0.5 = strong, |ρ| > 0.3 = moderate')
    lines.append(f'  * = p < 0.05 (significant at α=0.05)')
    lines.append(f'  Note: n={len(common)} limits statistical power; interpret trends.')

    text = '\n'.join(lines)
    print(text)

    out_txt = STAT_DIR / 'spearman_tau_gap.txt'
    with open(out_txt, 'w') as f:
        f.write(text)
    print(f'\n  ✓ Saved: {out_txt}')

    # Heatmap figure
    fig, ax = plt.subplots(figsize=(5.0, 3.5))
    im = ax.imshow(rho_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

    ax.set_xticks(range(n_ssl))
    ax.set_xticklabels(ssl_labels, rotation=20)
    ax.set_yticks(range(n_tau))
    ax.set_yticklabels([l for _, l in tau_keys])

    # Annotate cells with ρ and significance
    for i in range(n_tau):
        for j in range(n_ssl):
            rho = rho_matrix[i, j]
            pval = pval_matrix[i, j]
            sig = '*' if pval < 0.05 else ''
            color = 'white' if abs(rho) > 0.6 else 'black'
            ax.text(j, i, f'{rho:+.2f}{sig}', ha='center', va='center',
                    fontsize=9, fontweight='bold' if pval < 0.05 else 'normal', color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Spearman ρ')
    ax.set_title(f'Diagnostic severity vs. PSNR gap (n={len(common)})')

    plt.tight_layout()
    out = FIG_DIR / 'fig_tau_gap_heatmap.pdf'
    fig.savefig(out)
    fig.savefig(out.with_suffix('.png'))
    plt.close()
    print(f'  ✓ {out}')

    # Also produce scatter plots for the 3 key pairs
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.8))
    key_pairs = [
        ('tau6_interslice_shift', 'n2n', 'τ₆ (inter-slice shift)', 'N2N gap'),
        ('tau3_autocorrelation', 'nei2nei', 'τ₃ (autocorrelation)', 'Nei2Nei gap'),
        ('tau1_cv_local_sigma', 'noisier2noise', 'τ₁ (non-stationarity)', 'N2N⁺ gap'),
    ]
    for ax, (tkey, meth, xlabel, ylabel) in zip(axes, key_pairs):
        xv = tau_vals[tkey]
        yv = gaps[meth]
        ax.scatter(xv, yv, c=COLORS[meth], s=40, edgecolors='black', linewidths=0.5, zorder=3)
        # Add patient labels
        for k, p in enumerate(common):
            ax.annotate(p[1:], (xv[k], yv[k]), fontsize=6, ha='left', va='bottom',
                       xytext=(2, 2), textcoords='offset points')
        # Trend line
        rho, pval = sp_stats.spearmanr(xv, yv)
        z = np.polyfit(xv, yv, 1)
        xline = np.linspace(min(xv), max(xv), 50)
        ax.plot(xline, np.polyval(z, xline), '--', color='gray', alpha=0.5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(f'Δ PSNR (dB)')
        ax.set_title(f'ρ={rho:+.2f}, p={pval:.3f}')
        ax.grid(alpha=0.3)

    plt.tight_layout()
    out = FIG_DIR / 'fig_tau_gap_scatter.pdf'
    fig.savefig(out)
    fig.savefig(out.with_suffix('.png'))
    plt.close()
    print(f'  ✓ {out}')


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print('Loading data...')
    data = load_loocv_results()
    bm3d = load_bm3d()
    cd_data = load_controlled_distortion()
    diag_3mm = load_diagnostics_3mm()

    for m in METHODS:
        n = len(data[m]['psnr'])
        print(f'  {m}: {n}/10 folds')
    if bm3d:
        print(f'  bm3d: {len(bm3d)}/10 patients')

    print('\nGenerating figures...')
    fig_boxplot(data, bm3d)
    fig_controlled_distortion(cd_data)
    fig_3mm_vs_1mm(cd_data, diag_3mm)
    fig_per_patient_lines(data, bm3d)

    print('\nRunning statistical tests...')
    run_statistical_tests(data, bm3d)

    print('\nRunning τ vs PSNR gap correlation...')
    fig_tau_gap_correlation(data, diag_3mm)

    print('\nDone!')


if __name__ == '__main__':
    main()
