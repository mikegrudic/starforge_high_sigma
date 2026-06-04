"""Single-dataset chabrier_smooth_bounds vs chabrier_smooth_bounds_lognormal fit.

Fits both models on the M2e4_R1_Z0.01 IMF, prints model-comparison statistics
(LR, AIC, BIC, Wilks LRT) and a gap-statistic check, and saves a histogram +
fitted-IMF plot to ``imf_plots/lognormal.pdf``. Style and color scheme match
``imf_plots.py`` so the dataset is identifiable across figures.

Notebook-equivalent script — see ``lognormal.ipynb`` for the interactive
version. The parametric-bootstrap cell is intentionally not ported here; it
runs in a few minutes and the χ²(3) Wilks reference is plenty for the report.
"""

import os

import matplotlib

matplotlib.use("Agg")

import jax.numpy as jnp
import numpy as np
import salpyter
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D  # noqa: F401  (kept for parity with imf_plots.py)
from palettable.colorbrewer.qualitative import Dark2_3
from scipy.stats import chi2


IMF_DATA = (
    "imf_data/STARFORGE_RT/STARFORGE_v1.2/M2e4_R1/"
    "M2e4_R1_Z0.01_S0_A2_B0.1_I10000_Res271_n2_sol0.5_42/"
    "output_turbsphere_driving1/IMF.dat"
)
OUTDIR = "imf_plots"
OUT_PDF = os.path.join(OUTDIR, "lognormal.pdf")

# Match imf_plots.py's color assignment: this dataset is first in the sorted
# M2e4_R1 glob there, so it picks DARK2_3[0] (teal).
DATASET_COLOR = Dark2_3.mpl_colors[0]


def main() -> None:
    raw_masses = np.loadtxt(IMF_DATA)[:, 1]
    masses = raw_masses[raw_masses > 0.1]
    N = len(masses)
    lm_lo = float(np.log10(masses.min()))
    lm_hi = float(np.log10(masses.max()))

    p0 = salpyter.imf_mostlikely_params(masses, model="chabrier_smooth_bounds")
    p1 = salpyter.imf_mostlikely_params(masses, model="chabrier_smooth_bounds_lognormal")

    print(f"dataset: {IMF_DATA}")
    print(f"N = {N},  log10(m) range = [{lm_lo:.3f}, {lm_hi:.3f}]")
    print()
    print("MAP results")
    print("-----------")
    print(f"chabrier_smooth_bounds            -lnL = {p0.fun:.4f}  ({len(p0.x)} params)")
    for n, v in p0.params.items():
        print(f"    {n:<10s} = {v:.4f}")
    print()
    print(f"chabrier_smooth_bounds_lognormal  -lnL = {p1.fun:.4f}  ({len(p1.x)} params)")
    for n, v in p1.params.items():
        print(f"    {n:<10s} = {v:.4f}")
    print()

    # ---- model comparison ------------------------------------------------
    k0, k1 = len(p0.x), len(p1.x)
    delta_lnL = p0.fun - p1.fun                       # ln(L_mix / L_base)
    delta_AIC = 2 * (k1 - k0) - 2 * delta_lnL
    delta_BIC = np.log(N) * (k1 - k0) - 2 * delta_lnL

    print("model comparison")
    print("----------------")
    print(f"            base (k={k0})    mix (k={k1})     mix - base")
    print(f"-lnL        {p0.fun:>10.2f}    {p1.fun:>10.2f}    {p1.fun - p0.fun:>+10.2f}")
    print(f"AIC         {2*k0 + 2*p0.fun:>10.2f}    {2*k1 + 2*p1.fun:>10.2f}    {delta_AIC:>+10.2f}")
    print(f"BIC         {k0*np.log(N) + 2*p0.fun:>10.2f}    {k1*np.log(N) + 2*p1.fun:>10.2f}    {delta_BIC:>+10.2f}")
    print(f"\nL_mix / L_base = {np.exp(delta_lnL):.3g}")

    # Wilks' LRT — χ²(df=k1-k0) is conservative because logit_w → ∞ puts the
    # null on the boundary and makes the lognormal's (logm0, logsigma)
    # unidentifiable. True p is smaller, not larger, so the conclusion stands.
    D = 2 * delta_lnL
    df = k1 - k0
    print(f"LRT D = 2·ΔlnL = {D:.2f}  ~  χ²(df={df})   p = {chi2.sf(D, df):.3g}")
    print()

    # ---- gap statistic ---------------------------------------------------
    m_sorted = np.sort(masses)
    m_4, m_5 = m_sorted[-4], m_sorted[-5]
    logm_grid = np.linspace(np.log10(m_5), np.log10(m_4), 2001)
    imf_at_gap = np.asarray(
        salpyter.get_imf_function("chabrier_smooth_bounds")(
            jnp.asarray(logm_grid), p0.x, lm_lo, lm_hi
        )
    )
    p_gap = float(np.trapezoid(imf_at_gap, logm_grid))
    n_remaining = N - 2
    lam = n_remaining * p_gap

    print("gap statistic (under pure-chabrier MAP)")
    print("---------------------------------------")
    print(f"m_4 = {m_4:.3f} Msun  (log10 = {np.log10(m_4):.3f})")
    print(f"m_5 = {m_5:.3f} Msun  (log10 = {np.log10(m_5):.3f})")
    print(f"gap width = {np.log10(m_4) - np.log10(m_5):.3f} dex")
    print(f"per-star probability of landing in gap: p = {p_gap:.4g}")
    print(f"expected count in gap: λ = (N-2)·p = {lam:.3g}")
    print(f"P(0 stars in gap)  ≈ exp(-λ)    = {np.exp(-lam):.3g}")
    print(f"                  = (1-p)^(N-2) = {(1-p_gap)**n_remaining:.3g}")

    # ---- plot ------------------------------------------------------------
    BINS_PER_DECADE = 4
    LOG_M_LO, LOG_M_HI = -3, 4
    NUM_HIST_BINS = (LOG_M_HI - LOG_M_LO) * BINS_PER_DECADE
    mbins = np.logspace(LOG_M_LO, LOG_M_HI, 1 + NUM_HIST_BINS)
    log_bin_width = np.log10(mbins.max() / mbins.min()) / NUM_HIST_BINS

    logm = np.linspace(-1, 4, 501)
    mgrid = 10 ** logm
    imf_base = np.asarray(
        salpyter.get_imf_function("chabrier_smooth_bounds")(logm, p0.x, lm_lo, lm_hi)
    )
    imf_mix = np.asarray(
        salpyter.get_imf_function("chabrier_smooth_bounds_lognormal")(logm, p1.x, lm_lo, lm_hi)
    )
    imf_to_bins = log_bin_width * N

    counts, _ = np.histogram(raw_masses, mbins)
    counts = counts.astype(float)
    nonzero = np.flatnonzero(counts > 0)
    if len(nonzero) > 0:
        first, last = nonzero[0], nonzero[-1]
        counts[:first] = np.nan
        counts[last + 1:] = np.nan

    fig, ax = plt.subplots(1, 1, figsize=(4.5, 4.5))
    ax.stairs(counts, mbins, color=DATASET_COLOR, linewidth=1.2, alpha=0.75, baseline=None)

    base_c = np.where(((imf_base > 1e-6) | (mgrid >10)), imf_base * imf_to_bins, np.nan)
    mix_c = np.where(((imf_base > 1e-6) | (mgrid >10)), imf_mix * imf_to_bins, np.nan)
    ax.plot(
        mgrid, base_c, color=DATASET_COLOR, ls="-", lw=1.4,
        label=rf"Chabrier  ($\ln \mathcal{{L}}={-p0.fun:.1f}$)",
    )
    ax.plot(
        mgrid, mix_c, color=DATASET_COLOR, ls="--", lw=1.4,
        label=rf"Chabrier + Lognormal  ($\ln \mathcal{{L}}={-p1.fun:.1f}$)",
    )

    ax.set(
        xscale="log",
        yscale="log",
        xlim=[0.03, 4e3],
        ylim=[0.5, 1e3],
        xlabel=r"Stellar Mass $(M_\odot)$",
        ylabel="count per bin",
    )
    leg = ax.legend(loc="upper right", fontsize=8, borderaxespad=0, edgecolor="black")
    leg.get_frame().set_linewidth(0.8)
    fig.tight_layout()

    os.makedirs(OUTDIR, exist_ok=True)
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {OUT_PDF}")


if __name__ == "__main__":
    main()
