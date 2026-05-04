#!/usr/bin/env python3
"""
Paper 4 — Generate all figures for BRACIS 2026 submission.

Produces:
  fig1_architecture.pdf  — REDCNN-SE schematic (placeholder text diagram)
  fig3_convergence.pdf   — Convergence curves (3 from-scratch + supervised ref)
  fig4_visual.pdf        — Visual comparison (QD, N2N, Nei2Nei, Sup, FD + error maps)
  fig5_diagnostics.pdf   — Diagnostic severity vs PSNR gap scatter

Usage:
    python generate_paper4_figures.py
"""

import os, json, re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path

# ── Shared Style ──
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 9,
    'legend.fontsize': 7.5,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
})

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'paper4_results')
LOGS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')

# Colors (colorblind-safe palette)
C_SUP  = '#2166ac'   # blue — supervised
C_N2N  = '#b2182b'   # red — N2N
C_NEI  = '#1b7837'   # green — Nei2Nei  
C_F2N  = '#e08214'   # orange — N2N+
C_QD   = '#636363'   # gray — QD baseline


def parse_psnr_from_log(logfile, max_epochs=200):
    """Extract epoch → PSNR from training log."""
    epochs, psnrs = [], []
    pattern = re.compile(r'E\s*(\d+)\s.*PSNR:\s*([\d.]+)')
    with open(logfile) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                e, p = int(m.group(1)), float(m.group(2))
                if e <= max_epochs:
                    epochs.append(e)
                    psnrs.append(p)
    return np.array(epochs), np.array(psnrs)


def fig3_convergence():
    """Fig 3: Convergence curves for from-scratch methods + F2N collapse inset."""
    print("  Generating Fig 3: Convergence curves...")
    
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    
    # N2N from scratch (v4 — longest from-scratch run)
    e, p = parse_psnr_from_log(os.path.join(LOGS_DIR, 'n2n_v4_console.log'))
    if len(e) > 0:
        ax.plot(e, p, color=C_N2N, linewidth=1.2, label='N2N (from scratch)', zorder=3)
    
    # N2N v8 from scratch (run 2 = no ortho, starts at second "FROM SCRATCH" block)
    e8, p8 = parse_psnr_from_log(os.path.join(LOGS_DIR, 'n2n_v8_noiseortho.log'))
    if len(e8) > 0:
        # v8 has multiple runs concatenated; find the second from-scratch block
        # (first 20 epochs have ortho, second 20 don't)
        # Take only the no-ortho block starting around line 20
        idx_split = len(e8) // 2 if len(e8) > 30 else 0
        if idx_split > 0:
            e8b, p8b = e8[idx_split:], p8[idx_split:]
            ax.plot(e8b, p8b, color=C_N2N, linewidth=0.8, linestyle='--', 
                    label='N2N v8 (no ortho)', alpha=0.6, zorder=2)
    
    # Nei2Nei from scratch (v7)
    e, p = parse_psnr_from_log(os.path.join(LOGS_DIR, 'n2v_v7_nei2nei.log'))
    if len(e) > 0:
        ax.plot(e, p, color=C_NEI, linewidth=1.2, label='Nei2Nei (from scratch)', zorder=3)
    
    # N2N+ from scratch (v3)
    e, p = parse_psnr_from_log(os.path.join(LOGS_DIR, 'f2n_v3_noisier2noise.log'))
    if len(e) > 0:
        ax.plot(e, p, color=C_F2N, linewidth=1.2, label='Noisier2Noise (from scratch)', zorder=3)
    
    # Supervised reference line
    ax.axhline(y=33.65, color=C_SUP, linewidth=0.8, linestyle=':', alpha=0.7, 
               label='Supervised (33.65 dB)')
    
    # QD baseline
    ax.axhline(y=29.25, color=C_QD, linewidth=0.6, linestyle='--', alpha=0.5,
               label='QD input (29.25 dB)')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('PSNR (dB)')
    ax.set_title('Convergence: From-Scratch Self-Supervised Methods')
    ax.legend(loc='lower right', framealpha=0.9, edgecolor='none')
    ax.set_ylim(16, 35)
    ax.grid(True, alpha=0.15, linewidth=0.5)
    
    # Inset: F2N collapse (f2n.log, first 30 epochs)
    ax_inset = fig.add_axes([0.22, 0.55, 0.35, 0.3])  # [left, bottom, width, height]
    e_f2n, p_f2n = parse_psnr_from_log(os.path.join(LOGS_DIR, 'f2n.log'), max_epochs=30)
    if len(e_f2n) > 0:
        ax_inset.plot(e_f2n, p_f2n, color='#d7191c', linewidth=1.0)
        ax_inset.set_xlabel('Epoch', fontsize=6)
        ax_inset.set_ylabel('PSNR (dB)', fontsize=6)
        ax_inset.set_title('F2N Collapse', fontsize=7, fontweight='bold')
        ax_inset.tick_params(labelsize=5)
        ax_inset.axhline(y=29.25, color=C_QD, linewidth=0.4, linestyle='--', alpha=0.4)
        ax_inset.grid(True, alpha=0.1, linewidth=0.3)
        # Annotate the collapse
        if len(p_f2n) > 8:
            collapse_min = np.min(p_f2n[:10])
            ax_inset.annotate(f'{collapse_min:.1f} dB\n(collapse)', 
                            xy=(np.argmin(p_f2n[:10])+1, collapse_min),
                            fontsize=5, color='red', ha='center',
                            xytext=(10, collapse_min + 3),
                            arrowprops=dict(arrowstyle='->', color='red', lw=0.5))
    
    fig.tight_layout()
    path = os.path.join(OUT_DIR, 'fig3_convergence.pdf')
    fig.savefig(path)
    fig.savefig(path.replace('.pdf', '.png'))
    plt.close(fig)
    print(f"    Saved: {path}")


def fig4_visual():
    """Fig 4: Visual comparison — QD, N2N, Nei2Nei, Supervised, FD + error maps."""
    print("  Generating Fig 4: Visual comparison...")
    
    vis_path = os.path.join(OUT_DIR, 'visual_slices.npz')
    if not os.path.exists(vis_path):
        print("    ERROR: visual_slices.npz not found. Run eval first.")
        return
    
    data = np.load(vis_path)
    available_keys = list(data.keys())
    print(f"    Available keys: {available_keys[:10]}...")
    
    # Find a slice index that has all methods
    slice_idx = None
    for idx in [100, 50, 150]:
        needed = [f'qd_{idx}', f'fd_{idx}', f'Supervised_{idx}', f'N2N_scratch_{idx}', f'Nei2Nei_scratch_{idx}']
        if all(k in available_keys for k in needed):
            slice_idx = idx
            break
    
    if slice_idx is None:
        print("    ERROR: No complete slice found with all methods")
        return
    
    qd = data[f'qd_{slice_idx}']
    fd = data[f'fd_{slice_idx}']
    sup = data[f'Supervised_{slice_idx}']
    n2n = data[f'N2N_scratch_{slice_idx}']
    nei = data[f'Nei2Nei_scratch_{slice_idx}']
    
    # Crop to ROI for detail (center 200x200)
    H, W = qd.shape
    cy, cx = H//2, W//2
    s = 100  # half-size
    roi = (slice(cy-s, cy+s), slice(cx-s, cx+s))
    
    methods = [
        ('Quarter-Dose', qd, C_QD),
        ('N2N', n2n, C_N2N),
        ('Nei2Nei', nei, C_NEI),
        ('Supervised', sup, C_SUP),
        ('Full-Dose (GT)', fd, 'black'),
    ]
    
    fig, axes = plt.subplots(2, 5, figsize=(7.0, 3.2))
    
    for j, (title, img, color) in enumerate(methods):
        # Top row: full slice (windowed)
        axes[0, j].imshow(img, cmap='gray', vmin=0, vmax=1, interpolation='none')
        axes[0, j].set_title(title, fontsize=7, color=color, fontweight='bold')
        axes[0, j].axis('off')
        # Draw ROI box
        rect = plt.Rectangle((cx-s, cy-s), 2*s, 2*s, linewidth=0.6, 
                             edgecolor='yellow', facecolor='none')
        axes[0, j].add_patch(rect)
        
        # Bottom row: error map (|pred - FD|), except for FD itself
        if title == 'Full-Dose (GT)':
            # Show ROI zoom instead
            axes[1, j].imshow(img[roi], cmap='gray', vmin=0, vmax=1, interpolation='none')
            axes[1, j].set_title('ROI zoom', fontsize=6)
        else:
            err = np.abs(img - fd)
            # Show error in ROI
            im = axes[1, j].imshow(err[roi], cmap='hot', vmin=0, vmax=0.05, interpolation='none')
            from dataset import compute_metrics
            p, s_val = compute_metrics(img, fd)
            axes[1, j].set_title(f'{p:.2f} dB / {s_val:.4f}', fontsize=5.5)
        axes[1, j].axis('off')
    
    # Colorbar for error maps
    cax = fig.add_axes([0.92, 0.08, 0.01, 0.35])
    fig.colorbar(im, cax=cax, label='|Error|')
    
    fig.suptitle(f'L506, slice {slice_idx}', fontsize=8, y=0.98)
    fig.tight_layout(rect=[0, 0, 0.91, 0.95])
    
    path = os.path.join(OUT_DIR, 'fig4_visual.pdf')
    fig.savefig(path)
    fig.savefig(path.replace('.pdf', '.png'))
    plt.close(fig)
    print(f"    Saved: {path}")


def fig5_diagnostics():
    """Fig 5: Diagnostic test severity vs PSNR gap scatter."""
    print("  Generating Fig 5: Diagnostics vs gap...")
    
    json_path = os.path.join(OUT_DIR, 'eval_results.json')
    with open(json_path) as f:
        results = json.load(f)
    
    diag = results['diagnostics']
    sup_psnr = results['methods']['Supervised']['psnr_mean']
    
    # Map each method's primary diagnostic violation severity to its PSNR gap
    # N2N: affected by τ6 (inter-slice shift) and τ3 (autocorrelation)
    # Nei2Nei: affected by τ1 (non-stationarity) and τ3 (autocorrelation)
    # N2N+: affected by ALL (especially τ1, τ3, τ4, τ7)
    
    methods_data = {
        'N2N': {
            'gap': sup_psnr - results['methods']['N2N_scratch']['psnr_mean'],
            'primary_severity': diag['tau6_interslice_shift'],  # structural shift
            'label': 'N2N\n(τ6=inter-slice)',
            'color': C_N2N,
        },
        'Nei2Nei': {
            'gap': sup_psnr - results['methods']['Nei2Nei_scratch']['psnr_mean'],
            'primary_severity': diag['tau1_cv_local_sigma'],  # non-stationarity
            'label': 'Nei2Nei\n(τ1=non-stat.)',
            'color': C_NEI,
        },
        'N2N+': {
            'gap': sup_psnr - results['methods']['N2N+_scratch']['psnr_mean'],
            'primary_severity': diag['tau3_autocorrelation'] + diag['tau1_cv_local_sigma'],
            'label': 'Noisier2Noise\n(τ1+τ3)',
            'color': C_F2N,
        },
    }
    
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    
    xs, ys = [], []
    for name, d in methods_data.items():
        ax.scatter(d['primary_severity'], d['gap'], color=d['color'], 
                  s=80, zorder=3, edgecolors='black', linewidths=0.5)
        ax.annotate(d['label'], (d['primary_severity'], d['gap']),
                   fontsize=6, ha='center', va='bottom',
                   xytext=(0, 8), textcoords='offset points')
        xs.append(d['primary_severity'])
        ys.append(d['gap'])
    
    # Fit line
    xs, ys = np.array(xs), np.array(ys)
    if len(xs) >= 2:
        z = np.polyfit(xs, ys, 1)
        x_fit = np.linspace(min(xs) * 0.8, max(xs) * 1.2, 100)
        y_fit = np.polyval(z, x_fit)
        ax.plot(x_fit, y_fit, color='gray', linewidth=0.8, linestyle='--', alpha=0.5)
        
        # R² (note: only 3 points, so this is illustrative)
        ss_res = np.sum((ys - np.polyval(z, xs))**2)
        ss_tot = np.sum((ys - np.mean(ys))**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        ax.text(0.95, 0.05, f'$R^2$={r2:.2f}\n(n=3)', transform=ax.transAxes,
               fontsize=7, ha='right', va='bottom', style='italic', alpha=0.7)
    
    ax.set_xlabel('Diagnostic Severity (primary τ)')
    ax.set_ylabel('PSNR Gap vs Supervised (dB)')
    ax.set_title('Assumption Violation Severity vs Performance Gap')
    ax.grid(True, alpha=0.15, linewidth=0.5)
    
    fig.tight_layout()
    path = os.path.join(OUT_DIR, 'fig5_diagnostics.pdf')
    fig.savefig(path)
    fig.savefig(path.replace('.pdf', '.png'))
    plt.close(fig)
    print(f"    Saved: {path}")


def fig_diagnostics_table():
    """Generate a diagnostic summary table as a figure (Table 1 in paper)."""
    print("  Generating diagnostic table figure...")
    
    json_path = os.path.join(OUT_DIR, 'eval_results.json')
    with open(json_path) as f:
        results = json.load(f)
    
    diag = results['diagnostics']
    
    fig, ax = plt.subplots(figsize=(3.5, 2.0))
    ax.axis('off')
    
    table_data = [
        ['τ1', 'CV(local σ)', f"{diag['tau1_cv_local_sigma']:.3f}", 'Non-stationary'],
        ['τ2', 'Shapiro-Wilk p', f"{diag['tau2_gaussianity_p']:.3f}", 'Gaussian ✓'],
        ['τ3', 'Autocorr(lag=1)', f"{diag['tau3_autocorrelation']:.3f}", 'Correlated'],
        ['τ4', 'Signal-noise |r|', f"{abs(diag['tau4_signal_noise_r']):.3f}", 'Weak dep.'],
        ['τ5', '|Skewness|', f"{diag['tau5_skewness']:.3f}", 'Symmetric ✓'],
        ['τ6', '1−SSIM(adj)', f"{diag['tau6_interslice_shift']:.3f}", 'Shift exists'],
        ['τ7', 'δ/σ', f"{diag['tau7_delta_over_sigma']:.3f}", 'Moderate'],
    ]
    
    table = ax.table(cellText=table_data,
                    colLabels=['Test', 'Description', 'Value', 'Interpretation'],
                    loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.3)
    
    # Style header
    for j in range(4):
        table[0, j].set_facecolor('#4472C4')
        table[0, j].set_text_props(color='white', fontweight='bold')
    
    fig.tight_layout()
    path = os.path.join(OUT_DIR, 'fig_diagnostics_table.pdf')
    fig.savefig(path)
    fig.savefig(path.replace('.pdf', '.png'))
    plt.close(fig)
    print(f"    Saved: {path}")


def fig1_architecture():
    """Fig 1: REDCNN-SE architecture schematic."""
    print("  Generating Fig 1: Architecture schematic...")
    
    fig, ax = plt.subplots(figsize=(7.0, 2.0))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 3)
    ax.axis('off')
    
    # Encoder blocks
    enc_labels = ['Conv 5×5\nc→74', 'Conv 5×5\n74→74', 'Conv 5×5\n74→74', 
                  'Conv 5×5\n74→74', 'Conv 5×5\n74→74']
    dec_labels = ['TConv 5×5\n74→74', 'TConv 5×5\n74→74', 'TConv 5×5\n74→74',
                  'TConv 5×5\n74→74', 'TConv 5×5\n74→1']
    
    x_pos = np.linspace(0.5, 6.5, 5)
    dx_pos = np.linspace(7.5, 13.5, 5)
    
    for i, (x, label) in enumerate(zip(x_pos, enc_labels)):
        color = '#4472C4' if i < 4 else '#2F5496'
        rect = plt.Rectangle((x-0.45, 0.8), 0.9, 1.4, facecolor=color, 
                            edgecolor='black', linewidth=0.5, alpha=0.8)
        ax.add_patch(rect)
        ax.text(x, 1.5, f'e{i+1}\n{label}', ha='center', va='center', 
               fontsize=5, color='white', fontweight='bold')
        if i < 4:
            ax.annotate('', xy=(x+0.5, 1.5), xytext=(x+0.45, 1.5),
                       arrowprops=dict(arrowstyle='->', color='black', lw=0.5))
    
    for i, (x, label) in enumerate(zip(dx_pos, dec_labels)):
        color = '#C0504D' if i > 0 else '#943634'
        rect = plt.Rectangle((x-0.45, 0.8), 0.9, 1.4, facecolor=color,
                            edgecolor='black', linewidth=0.5, alpha=0.8)
        ax.add_patch(rect)
        ax.text(x, 1.5, f'd{5-i}\n{label}', ha='center', va='center',
               fontsize=5, color='white', fontweight='bold')
        if i < 4:
            ax.annotate('', xy=(x+0.5, 1.5), xytext=(x+0.45, 1.5),
                       arrowprops=dict(arrowstyle='->', color='black', lw=0.5))
    
    # SE blocks on skip connections
    for i in range(4):
        ex = x_pos[i]
        dx = dx_pos[3-i]
        ax.annotate('', xy=(dx, 2.3), xytext=(ex, 2.3),
                   arrowprops=dict(arrowstyle='->', color='#70AD47', lw=1.0,
                                  connectionstyle='arc3,rad=0.2'))
        mid_x = (ex + dx) / 2
        ax.text(mid_x, 2.7, f'SE{4-i}', fontsize=5, ha='center', va='center',
               bbox=dict(boxstyle='round,pad=0.15', facecolor='#70AD47', 
                        edgecolor='black', linewidth=0.3, alpha=0.8),
               color='white', fontweight='bold')
    
    # Global skip connection
    ax.annotate('', xy=(13.5, 0.7), xytext=(0.5, 0.7),
               arrowprops=dict(arrowstyle='->', color='gray', lw=0.8,
                              connectionstyle='arc3,rad=-0.15', linestyle='--'))
    ax.text(7.0, 0.3, 'Global residual (+x)', fontsize=6, ha='center', 
           style='italic', color='gray')
    
    # Input/Output labels
    ax.text(0.0, 1.5, 'x\n(QD)', fontsize=7, ha='center', va='center', fontweight='bold')
    ax.text(14.0, 1.5, 'x̂\n(denoised)', fontsize=7, ha='center', va='center', fontweight='bold')
    
    ax.set_title('REDCNN-SE Architecture (1.11M parameters, c=74)', fontsize=9, fontweight='bold')
    
    fig.tight_layout()
    path = os.path.join(OUT_DIR, 'fig1_architecture.pdf')
    fig.savefig(path)
    fig.savefig(path.replace('.pdf', '.png'))
    plt.close(fig)
    print(f"    Saved: {path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Paper 4 — Figure Generation")
    print(f"Output: {OUT_DIR}/\n")
    
    fig1_architecture()
    fig3_convergence()
    fig4_visual()
    fig5_diagnostics()
    fig_diagnostics_table()
    
    print("\nAll figures generated.")


if __name__ == '__main__':
    main()
