"""IMF comparison plot: all 4 datasets on one panel, color-coded.

For each STARFORGE directory in ``imfs_to_plot`` that has both ``IMF.dat`` and
the JAX driver's ``imf_samples_jax/samples_<model>.npy``, evaluate the IMF at
every posterior sample on a shared log-mass grid, then plot:
  * the input-mass histogram in the dataset's color
  * the 16-84 percentile envelope of the IMF posterior (fill_between)
  * the posterior median IMF (solid line, doubles as the legend handle)

Output goes to ``imf_plots/comparison_<model>.pdf``.
"""

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
    os.makedirs(OUTDIR, exist_ok=True)


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
