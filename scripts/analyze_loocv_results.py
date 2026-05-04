#!/usr/bin/env python3
"""
Post-processing script for LOO-CV results.
Run AFTER all 40 jobs complete.

Generates:
  1. PSNR/SSIM table (10 patients × 4 methods) with mean±std
  2. Wilcoxon signed-rank tests (paired comparisons)
  3. Scatter plot: diagnostic severity vs PSNR gap (W2 response)
  4. Box plots per method
  5. LaTeX table for paper
  6. JSON summary

Usage:
  python3 analyze_loocv_results.py [--partial]  # --partial to run with incomplete results
"""
import sys, os, json, argparse
import numpy as np
from pathlib import Path
from scipy import stats

# ═════════════════════════════════════════════════════════════════════════════
RESULTS_BASE = Path(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results'))
ALL_PATIENTS = ['L067', 'L096', 'L109', 'L143', 'L192',
                'L286', 'L291', 'L310', 'L333', 'L506']
ALL_METHODS = ['supervised', 'n2n', 'nei2nei', 'noisier2noise']
METHOD_LABELS = {
    'supervised': 'Supervised',
    'n2n': 'N2N (adj-slice)',
    'nei2nei': 'Nei2Nei (8-conn)',
    'noisier2noise': 'Noisier2Noise',
}


def load_results(partial=False):
    """Load all result.json files."""
    data = {}
    missing = []
    for pat in ALL_PATIENTS:
        data[pat] = {}
        for method in ALL_METHODS:
            rpath = RESULTS_BASE / f'fold_{pat}' / method / 'result.json'
            if rpath.exists():
                with open(rpath) as f:
                    data[pat][method] = json.load(f)
            else:
                missing.append(f'{method}/{pat}')
                data[pat][method] = None

    if missing and not partial:
        print(f"ERROR: {len(missing)} results missing:")
        for m in missing:
            print(f"  - {m}")
        print("Use --partial to analyze incomplete results.")
        sys.exit(1)
    elif missing:
        print(f"WARNING: {len(missing)} results missing, proceeding with partial data.")

    return data


def build_tables(data):
    """Build PSNR and SSIM matrices."""
    psnr = {m: [] for m in ALL_METHODS}
    ssim = {m: [] for m in ALL_METHODS}
    patients_with_data = []

    for pat in ALL_PATIENTS:
        has_all = all(data[pat][m] is not None for m in ALL_METHODS)
        has_any = any(data[pat][m] is not None for m in ALL_METHODS)
        if has_any:
            patients_with_data.append(pat)
        for m in ALL_METHODS:
            r = data[pat][m]
            if r is not None:
                psnr[m].append(r.get('best_psnr', 0))
                # Extract SSIM from log
                log = r.get('log', [])
                best_entry = max(log, key=lambda x: x.get('psnr', 0)) if log else {}
                ssim[m].append(best_entry.get('ssim', 0))
            else:
                psnr[m].append(np.nan)
                ssim[m].append(np.nan)

    return psnr, ssim, patients_with_data


def print_psnr_table(psnr, ssim):
    """Print formatted PSNR table."""
    print("\n" + "="*90)
    print("  PSNR RESULTS (dB) — LOO-CV 10 Patients × 4 Methods")
    print("="*90)

    header = f"  {'Patient':>8} │"
    for m in ALL_METHODS:
        header += f" {METHOD_LABELS[m]:>17s} │"
    header += f" {'N2N/Sup%':>8} │"
    print(header)
    print(f"  {'─'*8} ┼{'─'*19}┼{'─'*19}┼{'─'*19}┼{'─'*19}┼{'─'*10}┤")

    for i, pat in enumerate(ALL_PATIENTS):
        row = f"  {pat:>8} │"
        for m in ALL_METHODS:
            v = psnr[m][i]
            if np.isnan(v):
                row += f"       ---       │"
            else:
                row += f" {v:>15.2f}  │"
        # %Sup
        sup_v = psnr['supervised'][i]
        n2n_v = psnr['n2n'][i]
        if not np.isnan(sup_v) and not np.isnan(n2n_v) and sup_v > 0:
            row += f" {n2n_v/sup_v*100:>7.1f}% │"
        else:
            row += f"     ---  │"
        print(row)

    # Mean ± std
    print(f"  {'─'*8} ┼{'─'*19}┼{'─'*19}┼{'─'*19}┼{'─'*19}┼{'─'*10}┤")
    avg_row = f"  {'Mean':>8} │"
    for m in ALL_METHODS:
        vals = [v for v in psnr[m] if not np.isnan(v)]
        if vals:
            avg_row += f" {np.mean(vals):>7.2f}±{np.std(vals):<6.2f}  │"
        else:
            avg_row += f"       ---       │"
    # Mean %Sup
    pcts = []
    for i in range(len(ALL_PATIENTS)):
        s, n = psnr['supervised'][i], psnr['n2n'][i]
        if not np.isnan(s) and not np.isnan(n) and s > 0:
            pcts.append(n / s * 100)
    if pcts:
        avg_row += f" {np.mean(pcts):>7.1f}% │"
    else:
        avg_row += f"     ---  │"
    print(avg_row)


def wilcoxon_tests(psnr):
    """Run Wilcoxon signed-rank tests between all method pairs."""
    print("\n" + "="*90)
    print("  WILCOXON SIGNED-RANK TESTS (paired, two-sided)")
    print("="*90)

    comparisons = [
        ('supervised', 'n2n'),
        ('supervised', 'nei2nei'),
        ('supervised', 'noisier2noise'),
        ('n2n', 'nei2nei'),
        ('n2n', 'noisier2noise'),
        ('nei2nei', 'noisier2noise'),
    ]

    results = {}
    for m1, m2 in comparisons:
        v1 = np.array(psnr[m1])
        v2 = np.array(psnr[m2])
        mask = ~np.isnan(v1) & ~np.isnan(v2)
        if mask.sum() < 3:
            print(f"  {METHOD_LABELS[m1]:>17s} vs {METHOD_LABELS[m2]:<17s}: insufficient data")
            continue

        diff = v1[mask] - v2[mask]
        mean_diff = np.mean(diff)
        try:
            stat, p = stats.wilcoxon(v1[mask], v2[mask])
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        except ValueError:
            stat, p, sig = 0, 1.0, 'ns'

        print(f"  {METHOD_LABELS[m1]:>17s} vs {METHOD_LABELS[m2]:<17s}: "
              f"Δ={mean_diff:+.2f} dB  p={p:.4f} {sig}  (n={mask.sum()})")

        results[f'{m1}_vs_{m2}'] = {
            'delta_mean': float(mean_diff),
            'p_value': float(p),
            'significant': sig != 'ns',
            'n': int(mask.sum()),
        }

    return results


def generate_latex_table(psnr, ssim):
    """Generate LaTeX table for paper."""
    print("\n" + "="*90)
    print("  LATEX TABLE")
    print("="*90)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Leave-one-out cross-validation PSNR (dB) and SSIM across 10 patients.}",
        r"\label{tab:loocv}",
        r"\begin{tabular}{l" + "c" * len(ALL_METHODS) + "}",
        r"\toprule",
        r"Patient & " + " & ".join(METHOD_LABELS[m] for m in ALL_METHODS) + r" \\",
        r"\midrule",
    ]

    for i, pat in enumerate(ALL_PATIENTS):
        row_parts = [pat]
        for m in ALL_METHODS:
            v = psnr[m][i]
            s = ssim[m][i]
            if not np.isnan(v):
                row_parts.append(f"{v:.2f}")
            else:
                row_parts.append("---")
        lines.append(" & ".join(row_parts) + r" \\")

    lines.append(r"\midrule")

    # Mean row
    row_parts = [r"\textbf{Mean$\pm$Std}"]
    for m in ALL_METHODS:
        vals = [v for v in psnr[m] if not np.isnan(v)]
        if vals:
            row_parts.append(f"\\textbf{{{np.mean(vals):.2f}$\\pm${np.std(vals):.2f}}}")
        else:
            row_parts.append("---")
    lines.append(" & ".join(row_parts) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    latex = "\n".join(lines)
    print(latex)

    out_path = RESULTS_BASE / 'loocv_table.tex'
    with open(out_path, 'w') as f:
        f.write(latex)
    print(f"\n  Saved: {out_path}")
    return latex


def generate_plots(psnr, ssim):
    """Generate box plots and scatter plots."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping plots.")
        return

    # 1. Box plot of PSNR per method
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    bp_data = []
    bp_labels = []
    for m in ALL_METHODS:
        vals = [v for v in psnr[m] if not np.isnan(v)]
        if vals:
            bp_data.append(vals)
            bp_labels.append(METHOD_LABELS[m])

    if bp_data:
        bp = ax1.boxplot(bp_data, labels=bp_labels, patch_artist=True)
        colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
        for patch, color in zip(bp['boxes'], colors[:len(bp_data)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax1.set_ylabel('PSNR (dB)')
        ax1.set_title('LOO-CV PSNR Distribution (10 patients)')
        ax1.grid(axis='y', alpha=0.3)

    # 2. Per-patient comparison (supervised vs N2N)
    sup_vals = [v for v in psnr['supervised'] if not np.isnan(v)]
    n2n_vals = [v for v in psnr['n2n'] if not np.isnan(v)]
    pats_with = [p for i, p in enumerate(ALL_PATIENTS)
                 if not np.isnan(psnr['supervised'][i]) and not np.isnan(psnr['n2n'][i])]

    if sup_vals and n2n_vals and len(sup_vals) == len(n2n_vals):
        x = np.arange(len(pats_with))
        w = 0.35
        ax2.bar(x - w/2, sup_vals, w, label='Supervised', color='#2ecc71', alpha=0.8)
        ax2.bar(x + w/2, n2n_vals, w, label='N2N', color='#3498db', alpha=0.8)
        ax2.set_xticks(x)
        ax2.set_xticklabels(pats_with, rotation=45)
        ax2.set_ylabel('PSNR (dB)')
        ax2.set_title('Supervised vs N2N per Patient')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    fig.savefig(RESULTS_BASE / 'loocv_boxplots.png', dpi=150)
    fig.savefig(RESULTS_BASE / 'loocv_boxplots.pdf')
    plt.close(fig)
    print(f"  Saved: {RESULTS_BASE / 'loocv_boxplots.png'}")


def generate_scatter_w2(psnr):
    """Generate W2 scatter plot: diagnostic severity vs PSNR gap."""
    diag_path = RESULTS_BASE / 'diagnostics_per_patient.json'
    if not diag_path.exists():
        print("  diagnostics_per_patient.json not found, skipping W2 scatter.")
        print("  Run: python3 compute_diagnostics_per_patient.py first.")
        return

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping.")
        return

    with open(diag_path) as f:
        diag = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    method_pairs = [
        ('n2n', 'N2N Gap'),
        ('nei2nei', 'Nei2Nei Gap'),
        ('noisier2noise', 'Noisier2Noise Gap'),
    ]

    for ax, (method, title) in zip(axes, method_pairs):
        severities = []
        gaps = []
        labels = []
        for i, pat in enumerate(ALL_PATIENTS):
            if pat not in diag:
                continue
            sup_v = psnr['supervised'][i]
            met_v = psnr[method][i]
            if np.isnan(sup_v) or np.isnan(met_v):
                continue

            sev = diag[pat]['severity_composite']
            gap = sup_v - met_v
            severities.append(sev)
            gaps.append(gap)
            labels.append(pat)

        if len(severities) < 3:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title(title)
            continue

        severities = np.array(severities)
        gaps = np.array(gaps)

        ax.scatter(severities, gaps, s=80, c='steelblue', edgecolor='black', zorder=3)
        for s, g, l in zip(severities, gaps, labels):
            ax.annotate(l, (s, g), fontsize=8, ha='left', va='bottom',
                       xytext=(3, 3), textcoords='offset points')

        # Regression line
        if len(severities) >= 3:
            slope, intercept, r, p, se = stats.linregress(severities, gaps)
            x_fit = np.linspace(severities.min(), severities.max(), 100)
            ax.plot(x_fit, slope * x_fit + intercept, 'r--', alpha=0.7,
                    label=f'R²={r**2:.2f}, p={p:.3f}')

        ax.set_xlabel('Diagnostic Severity (composite)')
        ax.set_ylabel('PSNR Gap: Supervised − Method (dB)')
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    fig.suptitle('W2 Response: Diagnostic Severity vs Performance Gap (n=10 patients)',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(RESULTS_BASE / 'w2_scatter_severity_vs_gap.png', dpi=150)
    fig.savefig(RESULTS_BASE / 'w2_scatter_severity_vs_gap.pdf')
    plt.close(fig)
    print(f"  Saved: {RESULTS_BASE / 'w2_scatter_severity_vs_gap.png'}")


def save_summary(data, psnr, ssim, wilcoxon_results):
    """Save JSON summary."""
    summary = {
        'n_patients': len(ALL_PATIENTS),
        'n_methods': len(ALL_METHODS),
        'methods': ALL_METHODS,
        'patients': ALL_PATIENTS,
        'psnr': {},
        'ssim': {},
        'wilcoxon': wilcoxon_results,
    }

    for m in ALL_METHODS:
        vals = [v for v in psnr[m] if not np.isnan(v)]
        summary['psnr'][m] = {
            'values': [float(v) for v in psnr[m] if not np.isnan(v)],
            'mean': float(np.mean(vals)) if vals else None,
            'std': float(np.std(vals)) if vals else None,
            'n': len(vals),
        }
        svals = [v for v in ssim[m] if not np.isnan(v)]
        summary['ssim'][m] = {
            'values': [float(v) for v in ssim[m] if not np.isnan(v)],
            'mean': float(np.mean(svals)) if svals else None,
            'std': float(np.std(svals)) if svals else None,
            'n': len(svals),
        }

    out_path = RESULTS_BASE / 'loocv_summary.json'
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--partial', action='store_true',
                       help='Allow analysis with incomplete results')
    args = parser.parse_args()

    print("="*90)
    print("  LOO-CV RESULTS ANALYSIS")
    print("="*90)

    data = load_results(partial=args.partial)
    psnr, ssim, patients = build_tables(data)

    done = sum(1 for pat in ALL_PATIENTS for m in ALL_METHODS
               if data[pat][m] is not None)
    print(f"\n  Results loaded: {done}/{len(ALL_PATIENTS)*len(ALL_METHODS)}")

    print_psnr_table(psnr, ssim)
    wilcoxon_results = wilcoxon_tests(psnr)
    generate_latex_table(psnr, ssim)
    generate_plots(psnr, ssim)
    generate_scatter_w2(psnr)
    save_summary(data, psnr, ssim, wilcoxon_results)

    print(f"\n{'='*90}")
    print(f"  DONE. All outputs in: {RESULTS_BASE}")
    print(f"{'='*90}")


if __name__ == '__main__':
    main()
