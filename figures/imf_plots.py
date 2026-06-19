"""IMF comparison plot: all 4 datasets on one panel, color-coded.

For each STARFORGE directory in ``imfs_to_plot`` that has both ``IMF.dat`` and
the JAX driver's ``imf_samples_jax/samples_<model>.npy``, evaluate the IMF at
every posterior sample on a shared log-mass grid, then plot:
  * the input-mass histogram in the dataset's color
  * the 16-84 percentile envelope of the IMF posterior (fill_between)
  * the posterior median IMF (solid line, doubles as the legend handle)

Output goes to ``imf_plots/comparison_<model>.pdf``.
"""

import glob
import os

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

import jax
import jax.numpy as jnp
import numpy as np
import salpyter

from sim_paths import IMF_RUNS

model = "chabrier_smooth_bounds"
# If True, plot the histograms as fractions of total counts and the IMF curves
# as normalized PDFs (fraction per bin, integrating to 1). If False, use raw
# counts and the count-per-bin scaled IMF.
for NORMALIZED in True, False:
    imf_func = salpyter.get_imf_function(model)

    # 4 bins per decade, aligned so every decade boundary is a bin edge.
    BINS_PER_DECADE = 4
    LOG_M_LO, LOG_M_HI = -3, 4
    NUM_HIST_BINS = (LOG_M_HI - LOG_M_LO) * BINS_PER_DECADE
    mbins = np.logspace(LOG_M_LO, LOG_M_HI, 1 + NUM_HIST_BINS)
    # Curve grid extends down to log10(M)=-2 (0.01 Msun) so the Chabrier and
    # Kroupa reference lines remain visible into the brown-dwarf regime. Their
    # normalization range stays [log10(MMIN), log10(MMAX)] (set per-dataset in
    # the ref_imf calls below), so values >0.1 Msun match the data and posterior
    # curves; values below 0.1 Msun are extrapolated references, not predictions.
    logm = np.linspace(-2, 4, 601)
    mgrid = 10**logm
    logm_jax = jnp.asarray(logm)
    log_bin_width = np.log10(mbins.max() / mbins.min()) / NUM_HIST_BINS

    OUTDIR = "imf_plots"
    TABLES_DIR = "../tables"
    os.makedirs(OUTDIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)


    @jax.jit
    def _batched_imf(samples_jax, lmin, lmax):
        """Evaluate the IMF on logm_jax for every row of samples_jax."""
        return jax.vmap(lambda s: imf_func(logm_jax, s, lmin, lmax))(samples_jax)


    fig, ax = plt.subplots(1, 1, figsize=(4.5, 4.5))
    # Dark2_3 colors via sim_paths.logz_to_color — metallicity sets the color
    # uniformly, R_cloud doesn't.
    from sim_paths import logz_to_color

    def _color_for(run):
        return logz_to_color(run.logZ)

    labeled_chabrier = False
    labeled_kroupa = False
    for run in IMF_RUNS:
        imf_data_path = run.path + "/IMF.dat"
        samples_path = run.path + f"/imf_samples_jax/samples_{model}.npy"
        if not (os.path.isfile(imf_data_path) and os.path.isfile(samples_path)):
            print(f"skip (missing inputs): {run}")
            continue

        masses = np.loadtxt(imf_data_path)[:, 1]
        MMIN, MMAX = max(0.1, masses.min()), masses.max()
        fit_masses = masses[(masses > MMIN) & (masses < MMAX)]
        if len(fit_masses) == 0:
            print(f"skip (empty fit_masses): {run}")
            continue

        samples = np.load(samples_path)
        if len(samples) == 0:
            print(f"skip (empty samples): {run}")
            continue

        # Batch-evaluate the IMF for all posterior samples on the shared grid.
        imf_vals = np.asarray(
            _batched_imf(jnp.asarray(samples), float(np.log10(MMIN)), float(np.log10(MMAX)))
        )  # shape (n_samples, len(logm))

        lo, mid, hi = np.percentile(imf_vals, [16, 50, 84], axis=0)
        # imf(logm) integrates to 1 over log10(m); multiplying by log_bin_width
        # gives expected fraction-per-bin (a pdf in the bin sense). In NORMALIZED
        # mode that's the final scale; otherwise multiply through by N to get
        # expected count-per-bin matching the un-normalized histogram.
        imf_to_bins = log_bin_width * (1.0 if NORMALIZED else len(fit_masses))

        color = _color_for(run)
        label = run.short_label

        # Histogram the full mass array (including stars below the fit cutoff
        # so the cut-off tail is visible). In NORMALIZED mode we divide by
        # len(fit_masses) instead of counts.sum() so that bins inside the fit
        # range line up with the IMF curve (the IMF integrates to 1 over the
        # fit range). Bins below the cut show the cut-off tail with the same
        # denominator, so the total exceeds 1 — that's the intended
        # visualization of the cut-off stars.
        counts, _ = np.histogram(masses, mbins)
        counts = counts.astype(float)
        if NORMALIZED and len(fit_masses) > 0:
            counts = counts / len(fit_masses)
        # Mask zeros *outside* the populated range so the log-y axis doesn't
        # draw drop lines at the data extremes. Zeros *between* populated bins
        # are left in place — on the log axis they show as vertical bars,
        # which is the intended visualization of empty bins in the data body.
        # nonzero = np.flatnonzero(counts > 0)
        # if len(nonzero) > 0:
        #     first, last = nonzero[0], nonzero[-1]
        #     counts[:first] = np.nan
        #     counts[last + 1:] = np.nan
        # Use ax.stairs which takes the bin edges directly, so step boundaries
        # land *exactly* on mbins (including the decade boundaries). step()
        # with linear-midpoint alignment would shift the visible boundaries
        # away from the bin edges on a log x-axis.
        # Layering: references at zorder=1 (background), histogram at 2,
        # posterior fill_between + median line at 3 (foreground). Models on
        # top, histogram beneath, references in back.
        # Convention: M2e4_R10 entries get dotted lines so they're visually
        # distinguishable from the M2e4_R1 series of the same logZ color.
        ls = "dotted" if run.R_cloud == 10.0 else "solid"
        ax.stairs(counts, mbins, color=color, linewidth=1.2, alpha=0.75,
                  baseline=None, linestyle=ls, zorder=2)

        lo_c = np.where(lo > 0, lo * imf_to_bins, np.nan)
        hi_c = np.where(hi > 0, hi * imf_to_bins, np.nan)
        mid_c = np.where(mid > 0, mid * imf_to_bins, np.nan)
        ax.fill_between(mgrid, lo_c, hi_c, color=color, alpha=0.25, zorder=3)
        ax.plot(mgrid, mid_c, color=color, lw=1.4, ls=ls, label=label, zorder=3)

        # Reference IMFs (Chabrier 2005, Kroupa 2001) are only meaningful when
        # the y-axis is normalized — count-mode plots have a per-dataset N
        # scaling that doesn't line up with a "literature reference" overlay.
        if NORMALIZED:
            # Chabrier (2005) reference, scaled to this dataset's bins. Same form
            # as IMF_analysis_jax.py: chabrier_smooth at the published defaults.
            ref_imf = np.asarray(
                salpyter.get_imf_function("chabrier_smooth")(
                    logm,
                    np.asarray(salpyter.CHABRIER_SMOOTH_DEFAULT_PARAMS),
                    logmmin=float(np.log10(MMIN)),
                    logmmax=float(np.log10(MMAX)),
                )
            )
            ref_c = np.where(ref_imf > 0, ref_imf * imf_to_bins, np.nan)
            # Don't label here — we want "Chabrier 2005" at the end of the legend,
            # which we handle with a proxy handle after the loop.
            ax.plot(mgrid, ref_c, color="black", ls="dashed", lw=0.9, zorder=1)
            labeled_chabrier = True

            # Kroupa (2001) reference: salpyter's piecewise model with canonical
            # slopes and breaks. Same per-dataset scaling as Chabrier; legend
            # entry added via proxy handle after the loop.
            kroupa_params = np.array([
                0.7, -0.3, -1.3,
                float(np.log10(0.08)), float(np.log10(0.5)),
            ])
            kroupa_imf = np.asarray(
                salpyter.kroupa(
                    logm, kroupa_params,
                    float(np.log10(MMIN)), float(np.log10(MMAX)),
                )
            )
            kroupa_c = np.where(kroupa_imf > 0, kroupa_imf * imf_to_bins, np.nan)
            ax.plot(mgrid, kroupa_c, color="red", ls="dashed", lw=0.9, zorder=1)
            labeled_kroupa = True

    ax.set(
        xscale="log",
        yscale="log",
        xlim=[0.01, 4e3],
        ylim=[3e-4, 1] if NORMALIZED else [0.5, 1e4],
        xlabel=r"Stellar Mass $(M_\odot)$",
        ylabel="fraction per bin" if NORMALIZED else "count per bin",
    #    title=f"{model}  ·  16-84% posterior envelope per dataset",
    )
    handles, labels = ax.get_legend_handles_labels()
    from matplotlib.lines import Line2D
    # Prepend Chabrier and Kroupa so the reference IMFs lead the legend.
    ref_handles, ref_labels = [], []
    if labeled_chabrier:
        ref_handles.append(Line2D([], [], color="black", ls="dotted", lw=0.9))
        ref_labels.append("Chabrier 2005")
    if labeled_kroupa:
        ref_handles.append(Line2D([], [], color="red", ls="dotted", lw=0.9))
        ref_labels.append("Kroupa 2001")
    handles = ref_handles + handles
    labels = ref_labels + labels
    leg = ax.legend(handles, labels, loc="lower left", fontsize=8,
                    borderaxespad=0, edgecolor="black",labelspacing=0,frameon=True)
    leg.get_frame().set_linewidth(0.8)
    fig.tight_layout()
    out_path = os.path.join(OUTDIR, f"comparison_{model}{'_normalized' if NORMALIZED else ''}.pdf")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def _fmt_uncertain(values):
    """Format $median^{+up}_{-down}$ with 2 sig figs of the larger error."""
    lo, mid, hi = np.percentile(values, [16, 50, 84])
    up, dn = hi - mid, mid - lo
    err = max(up, dn, 1e-12)
    n = max(0, -int(np.floor(np.log10(err))) + 1)
    return rf"${mid:.{n}f}^{{+{up:.{n}f}}}_{{-{dn:.{n}f}}}$"


def _table_label(run):
    """`M2e4_RX_ZY`-style identifier extracted from the run directory name."""
    name = os.path.basename(os.path.dirname(run.path)).split("_S")[0]
    return r"\texttt{" + name.replace("_", r"\_") + "}"


def _sfe_values(run):
    """Return array of final SFE values.

    For combined output_all runs (n_realizations > 0) there is no
    global_statistics.fits; glob the individual driving-realization directories
    from the parent instead.
    """
    from astropy.table import Table

    def _sfe_from_path(path):
        stats = Table.read(path + "/global_statistics.fits")
        return float(np.max(stats["stellar_mass_sum"])) / run.M_cloud

    if run.n_realizations == 0:
        return np.array([_sfe_from_path(run.path)])
    parent = os.path.dirname(run.path)
    dirs = sorted(glob.glob(os.path.join(parent, "output_turbsphere_driving*")))
    return np.array([
        _sfe_from_path(d) for d in dirs
        if os.path.isfile(d + "/global_statistics.fits")
    ])


def write_imf_params_table():
    """Two tables: integrated properties (SFE, L/M, Q/M) and IMF parameters.

    Physics experiments are included in both tables. Chabrier 2003/2005 reference
    rows are appended to the IMF parameter table.

    Writes:
      ../tables/integrated_props_table_<model>.tex
      ../tables/imf_params_table_<model>.tex
      ../tables/imf_params_table_<model>.txt  (ASCII summary, both combined)
    """
    import math
    from starforge_tools.star_properties import Q_ionizing, luminosity_MS as lum_MS
    from sim_paths import PHYSICS_EXPERIMENTS

    NODATA = r"\ldots"

    # ---- collect rows for all runs (fiducial + physics experiments) ----
    # Each entry: (lbl, phys_label, sfe_arr, L_spec|None, QH_spec|None,
    #              m_peak|None, sigma_imf|None, gamma|None, m_max|None)
    table_rows = []
    for run in list(IMF_RUNS) + list(PHYSICS_EXPERIMENTS):
        phys_label = "Standard" if not run.is_physics else run.label
        lbl = _table_label(run)
        sfe = _sfe_values(run)

        samples_path = run.path + f"/imf_samples_jax/samples_{model}.npy"
        imf_data_path = run.path + "/IMF.dat"
        has_all = os.path.isfile(samples_path) and os.path.isfile(imf_data_path)

        if has_all:
            samples = np.load(samples_path)
            if len(samples) > 0:
                # chabrier_smooth_bounds params: [logm0, logsigma, alpha, logmmin, logmmax]
                m_peak    = 10 ** samples[:, 0]
                sigma_imf = np.exp(samples[:, 1])
                gamma     = samples[:, 2]
                m_max     = 10 ** samples[:, 4]
                masses    = np.loadtxt(imf_data_path)[:, 1]
                Mtot      = masses.sum()
                QH_spec   = np.sum(Q_ionizing(masses)) / Mtot
                L_spec    = np.sum(lum_MS(masses)) / Mtot
                table_rows.append(
                    (lbl, phys_label, sfe, L_spec, QH_spec, m_peak, sigma_imf, gamma, m_max)
                )
                continue
        # Missing IMF data: SFE is available from global_statistics, rest is not.
        table_rows.append((lbl, phys_label, sfe, None, None, None, None, None, None))

    def _fmt_sfe(sfe_arr):
        if len(sfe_arr) == 1:
            return f"${sfe_arr[0]:.2f}$"
        return rf"${np.mean(sfe_arr):.2f} \pm {np.std(sfe_arr, ddof=1):.2f}$"

    def _fmt_sci(val, sig=2):
        exp = int(math.floor(math.log10(abs(val))))
        m = val / 10**exp
        return rf"${m:.{sig-1}f}\times10^{{{exp}}}$"

    # ---- Table 1: Integrated Properties (SFE, L/M*, Q/M*) ----
    int_rows = []
    for lbl, phys, sfe, Lsp, QH, *_ in table_rows:
        lm_qm = (
            f"{_fmt_sci(Lsp)} & {_fmt_sci(QH)}"
            if Lsp is not None else f"{NODATA} & {NODATA}"
        )
        int_rows.append(f"    {lbl} & {phys} & {_fmt_sfe(sfe)} & {lm_qm} \\\\")

    int_tex = (
        "% Auto-generated by figures/imf_plots.py -- do not edit by hand.\n"
        "\\begin{table*}[t!]\n"
        "\\centering\n"
        "\\begin{tabular}{llccc}\n"
        "\\toprule\n"
        "Simulation & Physics & SFE & "
        "$L/M_\\star\\,(L_\\odot\\,M_\\odot^{-1})$ & "
        "$\\mathcal{Q}(H_0)/M_\\star\\,(\\mathrm{s}^{-1}\\,M_\\odot^{-1})$ \\\\\n"
        "\\midrule\n"
        + "\n".join(int_rows) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\caption{Basic properties of the simulated star clusters. "
        "SFE is the final star formation efficiency (fraction of initial cloud mass converted to stars), "
        "$L/M_\\star$ is the specific bolometric luminosity, and "
        "$\\mathcal{Q}(H_0)/M_\\star$ the specific H-ionizing photon production rate.}\n"
        "\\label{table:integrated_props}\n"
        "\\end{table*}\n"
    )
    int_path = os.path.join(TABLES_DIR, f"integrated_props_table_{model}.tex")
    with open(int_path, "w") as f:
        f.write(int_tex)
    print(f"wrote {int_path}")

    # ---- Table 2: IMF Parameters (m_c, sigma, Gamma, Mmax) ----
    imf_rows = []
    for lbl, phys, sfe, Lsp, QH, mp, sig_imf, g, mm in table_rows:
        if mp is not None:
            imf_rows.append(
                f"    {lbl} & {phys} & "
                f"{_fmt_uncertain(mp)} & {_fmt_uncertain(sig_imf)} & "
                f"{_fmt_uncertain(g)} & {_fmt_uncertain(mm)} \\\\"
            )
        else:
            imf_rows.append(
                rf"    {lbl} & {phys} & \multicolumn{{4}}{{c}}{{{NODATA}}} \\"
            )

    # Canonical published values from Chabrier 2005.
    chabrier_rows = [
        r"    \midrule",
        r"    \citet{chabrier_imf}         & --- & $0.25$ & $0.55$ & $-1.35$ & --- \\",
    ]

    imf_tex = (
        "% Auto-generated by figures/imf_plots.py -- do not edit by hand.\n"
        "\\begin{table*}[t!]\n"
        "\\centering\n"
        "\\begin{tabular}{llcccc}\n"
        "\\toprule\n"
        "Simulation & Physics & $m_c\\,(M_\\odot)$ & $\\sigma\\,(\\mathrm{dex})$ & $\\Gamma$ & $M_{\\max}\\,(M_\\odot)$ \\\\\n"
        "\\midrule\n"
        + "\n".join(imf_rows) + "\n"
        + "\n".join(chabrier_rows) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\caption{Posterior-median IMF parameters of different simulations, for the modified Chabrier model given in Appendix \\ref{appendix:mcmc}: peak mass $m_c$, log-normal variance in dex $\\sigma$, high-mass slope $\\Gamma$, and hard upper cutoff mass $M_{\\rm max}$.}\n"
        "\\label{table:imf_params}\n"
        "\\end{table*}\n"
    )
    imf_path = os.path.join(TABLES_DIR, f"imf_params_table_{model}.tex")
    with open(imf_path, "w") as f:
        f.write(imf_tex)
    print(f"wrote {imf_path}")

    # ---- ASCII summary (both tables combined) ----
    def _name(lbl):
        return lbl.replace(r"\texttt{", "").replace("}", "").replace(r"\_", "_")

    def _phys_str(p):
        return p.replace(r"$3\times$", "3x").replace("$", "").strip()

    col1w = max(len("Simulation"), max(len(_name(lbl)) for lbl, *_ in table_rows))
    phys_w = max(len("Physics"), max(len(_phys_str(p)) for _, p, *_ in table_rows))

    def _ascii_val(values):
        lo, mid, hi = np.percentile(values, [16, 50, 84])
        return f"{mid:10.3g} +{hi-mid:.3g} -{mid-lo:.3g}"

    def _ascii_sfe(sfe_arr):
        if len(sfe_arr) == 1:
            return f"{sfe_arr[0]:.3f}"
        return f"{np.mean(sfe_arr):.3f} +/-{np.std(sfe_arr, ddof=1):.3f}"

    header = (
        f"{'Simulation':<{col1w}}  {'Physics':<{phys_w}}  {'SFE':>8}"
        f"  {'m_c (Msun)':>24}  {'sigma':>24}  {'Gamma':>24}  {'M_max (Msun)':>24}"
        f"  {'L/M* (Lsun/Msun)':>18}  {'QH/M* (s^-1 Msun^-1)':>22}"
    )
    sep = "-" * len(header)
    lines = [sep, header, sep]
    for lbl, phys, sfe, Lsp, QH, mp, sig_imf, g, mm in table_rows:
        sfe_str = _ascii_sfe(sfe)
        if mp is not None:
            lines.append(
                f"{_name(lbl):<{col1w}}  {_phys_str(phys):<{phys_w}}  {sfe_str:>8}"
                f"  {_ascii_val(mp):>24}  {_ascii_val(sig_imf):>24}"
                f"  {_ascii_val(g):>24}  {_ascii_val(mm):>24}"
                f"  {Lsp:>18.0f}  {QH:>22.3e}"
            )
        else:
            lines.append(
                f"{_name(lbl):<{col1w}}  {_phys_str(phys):<{phys_w}}  {sfe_str:>8}"
                f"  {'...':>24}  {'...':>24}  {'...':>24}  {'...':>24}"
                f"  {'...':>18}  {'...':>22}"
            )
    lines.append(sep)
    ascii_text = "\n".join(lines) + "\n"
    txt_path = os.path.join(TABLES_DIR, f"imf_params_table_{model}.txt")
    with open(txt_path, "w") as f:
        f.write(ascii_text)
    print(f"wrote {txt_path}")
    print(ascii_text)


write_imf_params_table()
