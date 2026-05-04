"""
Physics-informed diagnostics v2 — refactored τ2 and τ4.

Addresses reviewer-level methodological issues in the original τ1–τ7 framework:

    τ2  Shapiro-Wilk p-value       →  excess kurtosis + Anderson-Darling statistic
    τ4  Pearson corr(signal,noise) →  heteroscedasticity coefficient (log-log σ² vs μ)

Rationale
---------
τ2 (old): p-value from Shapiro-Wilk is not an effect size and is n-dependent.
    On large patches any trivial deviation rejects H0; on small patches no
    deviation is detected. Reporting "p = 0.38 → non-Gaussian" is formally
    incorrect (p > α means "fail to reject", not "reject").

τ2 (new): Excess kurtosis (Fisher's definition: k₄ - 3) is a distribution-free
    effect size. For Gaussian it converges to 0; standard error under H₀ is
    sqrt(24/n). Anderson-Darling A² is reported as a statistic (not a p-value)
    and is tail-weighted, which matters for CT where noise deviations are
    primarily in the tails.

τ4 (old): Pearson corr ≈ 0 between signal and noise is independence in the
    mean — but CT noise is signal-dependent via the *variance* (Poisson-like
    pre-log, filtered and mixed post-FBP). Reporting corr=-0.02 as evidence
    of "signal-noise dependence" is the opposite of what the number says.

τ4 (new): Heteroscedasticity coefficient α from the power law σ² ∝ μ^α,
    estimated by log-log regression of local variance on local mean across
    flat-region windows. α = 0 → homoscedastic; α = 1 → Poisson-like;
    α ∈ (0,1) typical for post-FBP CT. Reported with OLS, Theil-Sen (robust),
    R², and the Breusch-Pagan test statistic as a formal heteroscedasticity
    check.

Author: implementation for BRACIS 2026 rebuttal (Reviewer W2/W4 response).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
from scipy import stats
from scipy.ndimage import sobel


# ---------------------------------------------------------------------------
# τ2 — Gaussianity as effect size
# ---------------------------------------------------------------------------

@dataclass
class Tau2Result:
    """Result of τ2 (Gaussianity) diagnostic.

    Attributes
    ----------
    excess_kurtosis : float
        Fisher excess kurtosis (k₄ - 3). 0 for Gaussian, >0 heavy-tailed,
        <0 light-tailed / platykurtic. This is the primary effect size.
    kurtosis_z : float
        |excess_kurtosis| / SE, where SE = sqrt(24/n). Scale-free magnitude.
        Under H₀ (Gaussian), |z| > 2 is ~5% tail; > 3 is strong evidence of
        non-Gaussianity *independent* of sample size in the sense that it
        normalizes by the sampling variance.
    anderson_darling : float
        A² statistic. Tail-weighted goodness-of-fit to Gaussian. Critical
        values (from scipy.stats.anderson) at 5% level ≈ 0.787 for unknown
        μ/σ; A² > 1 is decisive deviation.
    anderson_darling_adj : float
        Adjusted A² per Stephens (1986): A²·(1 + 0.75/n + 2.25/n²).
        Slightly more conservative for finite n.
    n_samples : int
        Sample size used.

    Notes
    -----
    Report `excess_kurtosis` and `anderson_darling_adj` as the two numbers.
    Drop Shapiro p-values entirely from the paper.
    """
    excess_kurtosis: float
    kurtosis_z: float
    anderson_darling: float
    anderson_darling_adj: float
    n_samples: int


def compute_tau2_gaussianity(noise: np.ndarray) -> Tau2Result:
    """Compute τ2 (Gaussianity of noise) as effect sizes, not p-values.

    Parameters
    ----------
    noise : np.ndarray
        1D or flattenable array of noise samples. Typically the residual
        r = QD - FD sampled in flat anatomical regions (mask-based) so that
        structural content does not contaminate the noise estimate.

    Returns
    -------
    Tau2Result

    Raises
    ------
    ValueError
        If fewer than 8 samples are provided (Anderson-Darling minimum).

    Examples
    --------
    Good — operate on flat-region residual samples:
    >>> flat_mask = extract_flat_regions(full_dose_image)
    >>> noise = (quarter_dose - full_dose)[flat_mask]
    >>> r = compute_tau2_gaussianity(noise)

    Bad — passing the full residual including anatomy:
    >>> r = compute_tau2_gaussianity(quarter_dose - full_dose)  # contaminated
    """
    x = np.asarray(noise, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    n = x.size

    if n < 8:
        raise ValueError(
            f"τ2 requires n≥8 samples for Anderson-Darling; got n={n}."
        )

    # Fisher excess kurtosis: 0 under Gaussian, distribution-free effect size.
    # bias=False uses the unbiased estimator (Fisher's k-statistic).
    k_excess = float(stats.kurtosis(x, fisher=True, bias=False))

    # Normalize by the SE of sample excess kurtosis under H₀ (Gaussian):
    # Var(k₄ - 3) ≈ 24/n for large n. Gives a unit-free "z-like" magnitude.
    se_k = np.sqrt(24.0 / n)
    k_z = abs(k_excess) / se_k

    # Anderson-Darling for Gaussian with estimated μ, σ. Pass method when
    # available (scipy >=1.13) to silence the 1.17 FutureWarning; fall back
    # to the default signature otherwise.
    try:
        ad = stats.anderson(x, dist="norm", method="interpolate")  # type: ignore[call-arg]
    except TypeError:
        ad = stats.anderson(x, dist="norm")
    a2 = float(ad.statistic)

    # Stephens (1986) small-sample adjustment.
    a2_adj = a2 * (1.0 + 0.75 / n + 2.25 / (n ** 2))

    return Tau2Result(
        excess_kurtosis=k_excess,
        kurtosis_z=k_z,
        anderson_darling=a2,
        anderson_darling_adj=a2_adj,
        n_samples=n,
    )


# ---------------------------------------------------------------------------
# τ4 — Signal-dependent noise as heteroscedasticity
# ---------------------------------------------------------------------------

@dataclass
class Tau4Result:
    """Result of τ4 (signal-dependent noise) diagnostic.

    Attributes
    ----------
    alpha_ols : float
        Power-law exponent from OLS log-log regression: log σ² = α·log μ + c.
        α = 0  → homoscedastic (i.i.d. assumption satisfied)
        α = 1  → Poisson-like (pre-log attenuation domain)
        α ∈ (0,1) typical for post-FBP reconstructed CT in HU.
    alpha_theilsen : float
        Robust Theil-Sen estimate of α. Use this if OLS is sensitive to
        high-leverage windows (e.g., near-zero μ).
    r_squared : float
        Coefficient of determination of the OLS log-log fit. High R² with
        α≠0 is the strongest evidence of signal-dependent noise.
    breusch_pagan_stat : float
        Breusch-Pagan test statistic for heteroscedasticity in the residuals
        of a mean-level regression. Chi² distributed under H₀ (homoscedastic),
        df=1. Values > 3.84 indicate heteroscedasticity at 5% level.
    n_windows : int
        Number of windows retained after flat-region screening.
    window_size : int
        Side length of the square window used.
    """
    alpha_ols: float
    alpha_theilsen: float
    r_squared: float
    breusch_pagan_stat: float
    n_windows: int
    window_size: int


def _tile_windows(img: np.ndarray, size: int) -> np.ndarray:
    """Non-overlapping square tiling. Returns array of shape (N, size, size)."""
    h, w = img.shape
    h_trim = h - (h % size)
    w_trim = w - (w % size)
    img = img[:h_trim, :w_trim]
    tiles = img.reshape(h_trim // size, size, w_trim // size, size)
    # (rows, size, cols, size) → (rows*cols, size, size)
    return tiles.transpose(0, 2, 1, 3).reshape(-1, size, size)


def _gradient_magnitude(img: np.ndarray) -> np.ndarray:
    """Per-pixel Sobel gradient magnitude. Used for flat-region screening."""
    gx = sobel(img, axis=0, mode="reflect")
    gy = sobel(img, axis=1, mode="reflect")
    return np.hypot(gx, gy)


def compute_tau4_heteroscedasticity(
    signal: np.ndarray,
    residual: np.ndarray,
    window_size: int = 32,
    gradient_percentile: float = 80.0,
    min_mean_hu: Optional[float] = -900.0,
    max_mean_hu: Optional[float] = None,
) -> Tau4Result:
    """Estimate signal-noise dependence as heteroscedasticity coefficient α.

    Procedure
    ---------
    1. Tile both `signal` (e.g., full-dose reference) and `residual` 
       (e.g., quarter-dose − full-dose) into non-overlapping windows.
    2. For each window, compute local mean μ from `signal` and local
       variance σ² from `residual`.
    3. Flat-region screening: discard windows whose mean gradient magnitude
       (from `signal`) is above `gradient_percentile`. Structural content
       inflates local variance artificially and biases the fit.
    4. Optional: discard windows with μ < `min_mean_hu` (air) — these are
       outside the body and have no clinically meaningful attenuation.
    5. Fit log σ² = α · log μ + c via both OLS and Theil-Sen (robust).
       Uses μ - μ_min shift to keep logs defined if μ can be ≤ 0 (HU).
    6. Compute Breusch-Pagan statistic on linear σ² ~ μ for a formal
       heteroscedasticity test.

    Parameters
    ----------
    signal : np.ndarray
        2D signal map (full-dose image, HU). Same shape as `residual`.
    residual : np.ndarray
        2D noise realization map. Typically QD - FD.
    window_size : int
        Side of the square window. Default 32 (standard in CT noise
        characterization literature).
    gradient_percentile : float
        Windows with mean gradient above this percentile are discarded.
        Default 80 → retains bottom 80% flattest windows.
    min_mean_hu : float or None
        Discard windows with local mean below this threshold. Defaults to
        -900 HU (rough air cutoff). Pass None to keep all.
    max_mean_hu : float or None
        Discard windows with local mean above this threshold. Useful for
        excluding bone (>+240 HU) which has different noise physics post-FBP.
        Defaults to None (no upper cutoff).

    Returns
    -------
    Tau4Result

    Notes
    -----
    Scale matters. Working in HU, mean values near -1000 (air) force a
    shift before log. We shift by (μ_min - 1) per image so all shifted
    means are strictly positive. α is invariant to additive shifts only
    in the variance, not in the mean, so this is a modelling choice;
    report the shift in the paper.
    """
    if signal.shape != residual.shape:
        raise ValueError(
            f"signal and residual must share shape; got {signal.shape} vs "
            f"{residual.shape}."
        )
    if signal.ndim != 2:
        raise ValueError(f"expected 2D arrays; got ndim={signal.ndim}.")

    sig = np.asarray(signal, dtype=np.float64)
    res = np.asarray(residual, dtype=np.float64)

    # Per-pixel gradient of signal → per-window mean gradient for screening.
    grad = _gradient_magnitude(sig)

    sig_tiles = _tile_windows(sig, window_size)
    res_tiles = _tile_windows(res, window_size)
    grad_tiles = _tile_windows(grad, window_size)

    mu = sig_tiles.mean(axis=(1, 2))
    var = res_tiles.var(axis=(1, 2), ddof=1)
    grad_mean = grad_tiles.mean(axis=(1, 2))

    # --- flat-region screening -------------------------------------------
    grad_cutoff = np.percentile(grad_mean, gradient_percentile)
    keep = grad_mean <= grad_cutoff

    # --- air / out-of-body screening -------------------------------------
    if min_mean_hu is not None:
        keep &= mu >= min_mean_hu
    # --- bone screening -------------------------------------------------
    if max_mean_hu is not None:
        keep &= mu <= max_mean_hu

    # --- numerical guards -------------------------------------------------
    keep &= var > 0  # log-undefined on zero variance

    mu_k = mu[keep]
    var_k = var[keep]

    if mu_k.size < 20:
        raise RuntimeError(
            f"τ4 retained only {mu_k.size} windows after screening; "
            "need ≥20 for a meaningful fit. Check gradient_percentile and "
            "min_mean_hu, or use a smaller window_size."
        )

    # Shift μ so logs are defined. Only apply if μ can be ≤ 0 (HU can be
    # negative). For strictly positive μ, any shift distorts the power-law
    # slope asymmetrically (compresses small values, leaves large values
    # nearly unchanged), biasing α toward 0.
    if mu_k.min() <= 0.0:
        mu_for_log = mu_k - mu_k.min() + 1.0
    else:
        mu_for_log = mu_k
    log_mu = np.log(mu_for_log)
    log_var = np.log(var_k)

    # --- OLS in log-log ---------------------------------------------------
    # σ² ∝ μ^α  ⇔  log σ² = α · log μ + c
    ols = stats.linregress(log_mu, log_var)
    alpha_ols = float(ols.slope)
    r2 = float(ols.rvalue ** 2)

    # --- Theil-Sen (robust to outlier windows) ----------------------------
    ts = stats.theilslopes(log_var, log_mu)
    alpha_ts = float(ts.slope)

    # --- Breusch-Pagan test -----------------------------------------------
    # H₀: residual² is uncorrelated with μ (homoscedasticity).
    # Auxiliary regression: e_i² = a + b·μ_i; test statistic n·R²_aux ~ χ²(1).
    lin = stats.linregress(mu_k, var_k)
    e = var_k - (lin.intercept + lin.slope * mu_k)
    aux = stats.linregress(mu_k, e ** 2)
    bp_stat = float(mu_k.size * aux.rvalue ** 2)

    return Tau4Result(
        alpha_ols=alpha_ols,
        alpha_theilsen=alpha_ts,
        r_squared=r2,
        breusch_pagan_stat=bp_stat,
        n_windows=int(mu_k.size),
        window_size=window_size,
    )


# ---------------------------------------------------------------------------
# Smoke test with controlled synthetic noise
# ---------------------------------------------------------------------------

def _sanity_check(seed: int = 0) -> None:
    """Run the two diagnostics on synthetic data with known properties.

    Expected behavior:
    * Pure AWGN                 → excess_kurtosis ≈ 0, α ≈ 0
    * Laplace noise             → excess_kurtosis ≈ 3, α ≈ 0
    * Signal-dependent Gaussian → excess_kurtosis ≈ 0, α ≈ 1 (Poisson-like)
    """
    rng = np.random.default_rng(seed)

    print("=" * 66)
    print("τ2 sanity checks")
    print("=" * 66)

    awgn = rng.normal(0.0, 1.0, size=20_000)
    r = compute_tau2_gaussianity(awgn)
    print(f"  AWGN      : k_excess={r.excess_kurtosis:+.3f} "
          f"(z={r.kurtosis_z:.2f})  A²_adj={r.anderson_darling_adj:.3f}")

    laplace = rng.laplace(0.0, 1.0, size=20_000)
    r = compute_tau2_gaussianity(laplace)
    print(f"  Laplace   : k_excess={r.excess_kurtosis:+.3f} "
          f"(z={r.kurtosis_z:.2f})  A²_adj={r.anderson_darling_adj:.3f}")

    uniform = rng.uniform(-1.0, 1.0, size=20_000)
    r = compute_tau2_gaussianity(uniform)
    print(f"  Uniform   : k_excess={r.excess_kurtosis:+.3f} "
          f"(z={r.kurtosis_z:.2f})  A²_adj={r.anderson_darling_adj:.3f}")

    print()
    print("=" * 66)
    print("τ4 sanity checks")
    print("=" * 66)

    H, W = 512, 512
    # Gradient signal: from 0 to 1000 HU across the image.
    sig = np.tile(np.linspace(0.0, 1000.0, W), (H, 1))

    # (a) Homoscedastic Gaussian noise → α should be ≈ 0.
    res_homo = rng.normal(0.0, 25.0, size=(H, W))
    r = compute_tau4_heteroscedasticity(sig, res_homo, min_mean_hu=None)
    print(f"  Homoscedastic : α_ols={r.alpha_ols:+.3f} "
          f"α_ts={r.alpha_theilsen:+.3f}  R²={r.r_squared:.3f}  "
          f"BP={r.breusch_pagan_stat:.2f}  n={r.n_windows}")

    # (b) Poisson-like: σ² ∝ μ  →  α should be ≈ 1.
    mu_pos = sig - sig.min() + 1.0
    res_poisson = rng.normal(0.0, np.sqrt(mu_pos))
    r = compute_tau4_heteroscedasticity(sig, res_poisson, min_mean_hu=None)
    print(f"  Poisson-like  : α_ols={r.alpha_ols:+.3f} "
          f"α_ts={r.alpha_theilsen:+.3f}  R²={r.r_squared:.3f}  "
          f"BP={r.breusch_pagan_stat:.2f}  n={r.n_windows}")

    # (c) Sub-linear heteroscedasticity: σ² ∝ μ^0.5  →  α should be ≈ 0.5.
    res_sub = rng.normal(0.0, mu_pos ** 0.25)  # std ∝ μ^0.25 → var ∝ μ^0.5
    r = compute_tau4_heteroscedasticity(sig, res_sub, min_mean_hu=None)
    print(f"  Sub-linear    : α_ols={r.alpha_ols:+.3f} "
          f"α_ts={r.alpha_theilsen:+.3f}  R²={r.r_squared:.3f}  "
          f"BP={r.breusch_pagan_stat:.2f}  n={r.n_windows}")


if __name__ == "__main__":
    _sanity_check(seed=42)
