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
import re
from glob import glob

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

import jax
import jax.numpy as jnp
import numpy as np
import salpyter


imfs_to_plot = (
    glob(
        "/Users/mgrudic/code/starforge_high_sigma/figures/imf_data/STARFORGE_RT/"
        "STARFORGE_v1.2/M2e4_R1/*/output*"
    )
    + [
        "/Users/mgrudic/code/starforge_high_sigma/figures/imf_data/STARFORGE_RT/"
        "STARFORGE_v1.2/M2e4_R10/M2e4_R10_Z1_S0_A2_B0.1_I1_Res271_n2_sol0.5_42/output_all"
    ]
)

model = "chabrier_smooth_bounds"
# If True, plot the histograms as fractions of total counts and the IMF curves
# as normalized PDFs (fraction per bin, integrating to 1). If False, use raw
# counts and the count-per-bin scaled IMF.
for NORMALIZED in True, False:
    imf_func = salpyter.get_imf_function(model)

    NUM_HIST_BINS = 31
    mbins = np.logspace(-3, 4, 1 + NUM_HIST_BINS)
    logm = np.linspace(-1, 4, 501)
    mgrid = 10**logm
    logm_jax = jnp.asarray(logm)
    log_bin_width = np.log10(mbins.max() / mbins.min()) / NUM_HIST_BINS

    OUTDIR = "imf_plots"
    os.makedirs(OUTDIR, exist_ok=True)


    @jax.jit
    def _batched_imf(samples_jax, lmin, lmax):
        """Evaluate the IMF on logm_jax for every row of samples_jax."""
        return jax.vmap(lambda s: imf_func(logm_jax, s, lmin, lmax))(samples_jax)


    def _short_label(run):
        """Extract a short, human-friendly legend label from the sim-dir path.

        The simulation directory encodes the cloud mass (``M<x>e<y>``), radius
        (``R<value>`` in pc), and metallicity (``Z<value>`` in solar units). The
        mean surface density is ``Sigma = M / (pi * R^2)`` in Msun/pc^2.
        """
        parts = os.path.normpath(run).split(os.sep)
        sim_dir = parts[-2] if len(parts) >= 2 else parts[-1]
        out_dir = parts[-1]
        m_m = re.search(r"M([\d.]+(?:e[+-]?\d+)?)_", sim_dir)
        m_r = re.search(r"_R([\d.]+)_", sim_dir)
        m_z = re.search(r"_Z([\d.]+?)_", sim_dir)
        if m_m and m_r:
            sigma = float(m_m.group(1)) / (np.pi * float(m_r.group(1)) ** 2)
            # Round to 2 significant figures.
            import math
            exp = math.floor(math.log10(abs(sigma)))
            sigma_2sf = round(sigma, -int(exp) + 1)
            sigma_str = f"{sigma_2sf:.0f}"
        else:
            sigma_str = "?"
        z_val = m_z.group(1) if m_z else "?"

        # Per-output-subdir suffix: drop "turbsphere_driving1", relabel "all".
        if out_dir == "output_all":
            suffix = "  (10 realizations)"
        elif out_dir == "output_turbsphere_driving1":
            suffix = ""
        else:
            suffix = f"  ({out_dir.replace('output_', '')})"

        return (
            rf"${sigma_str}\,M_\odot\,\mathrm{{pc}}^{{-2}}$, "
            rf"${z_val}\,Z_\odot${suffix}"
        )


    fig, ax = plt.subplots(1, 1, figsize=(4.5, 4.5))
    # ColorBrewer Dark2_3 (qualitative, 3-class) for the M2e4_R1 datasets;
    # black for the M2e4_R10 (low-Sigma) dataset.
    from palettable.colorbrewer.qualitative import Dark2_3
    DARK2_3 = Dark2_3.mpl_colors

    def _color_for(run, dark2_idx_holder=[0]):
        sim_dir = os.path.normpath(run).split(os.sep)[-2]
        if "M2e4_R10" in sim_dir:
            return "black"
        c = DARK2_3[dark2_idx_holder[0] % len(DARK2_3)]
        dark2_idx_holder[0] += 1
        return c

    labeled_chabrier = False
    for run in imfs_to_plot:
        imf_data_path = run + "/IMF.dat"
        samples_path = run + f"/imf_samples_jax/samples_{model}.npy"
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
        label = _short_label(run)

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
        nonzero = np.flatnonzero(counts > 0)
        if len(nonzero) > 0:
            first, last = nonzero[0], nonzero[-1]
            counts[:first] = np.nan
            counts[last + 1:] = np.nan
        bin_centers = 0.5 * (mbins[:-1] + mbins[1:])
        ax.step(bin_centers, counts, where="mid", color=color, linewidth=1.2, alpha=0.75)

        lo_c = np.where(lo > 0, lo * imf_to_bins, np.nan)
        hi_c = np.where(hi > 0, hi * imf_to_bins, np.nan)
        mid_c = np.where(mid > 0, mid * imf_to_bins, np.nan)
        ax.fill_between(mgrid, lo_c, hi_c, color=color, alpha=0.25)
        ax.plot(mgrid, mid_c, color=color, lw=1.4, label=label)

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
        ax.plot(mgrid, ref_c, color="black", ls="dotted", lw=0.9)
        labeled_chabrier = True

    ax.set(
        xscale="log",
        yscale="log",
        xlim=[5e-3, 1e4],
        ylim=[1e-4, 1] if NORMALIZED else [0.5, 1e4],
        xlabel=r"$M\ (M_\odot)$",
        ylabel="fraction per bin" if NORMALIZED else "count per bin",
    #    title=f"{model}  ·  16-84% posterior envelope per dataset",
    )
    handles, labels = ax.get_legend_handles_labels()
    if labeled_chabrier:
        from matplotlib.lines import Line2D
        handles.append(Line2D([], [], color="black", ls="dotted", lw=0.9))
        labels.append("Chabrier 2005")
    ax.legend(handles, labels, loc="lower left", fontsize=8, frameon=True)
    fig.tight_layout()
    out_path = os.path.join(OUTDIR, f"comparison_{model}{"_normalized" if NORMALIZED else ""}.pdf")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")
