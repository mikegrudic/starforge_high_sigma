"""Alpha plot: posterior-percentile envelope of the IMF slope Gamma vs M.

Generates two PDFs in one run:
  * ``alphaplot_<model>.pdf``      — bare plot (no observational overlay)
  * ``alphaplot_obs_<model>.pdf``  — same plot + literature compilation from
                                     ``~/code/alphaplot/alphaplot.csv``

Switching between them is a ``WITH_OBS`` toggle inside the loop; all the
shared infrastructure (dataset envelopes, Chabrier/Kroupa references,
axes/legend styling) lives in one place so changes affect both outputs.
"""

import os

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import salpyter

from sim_paths import IMF_RUNS, LOGZ_COLORMAP, logz_to_color


# ---------------- inputs ---------------- #

ALPHAPLOT_CSV = os.path.expanduser("./alphaplot.csv")

model = "chabrier_smooth_bounds"

logm_alpha = np.linspace(-2, 3.5, 601)
mgrid_alpha = 10**logm_alpha
logm_alpha_jax = jnp.asarray(logm_alpha)

OUTDIR = "imf_plots"
os.makedirs(OUTDIR, exist_ok=True)


# ---------------- helpers ---------------- #

def _color_for(run):
    return logz_to_color(run.logZ)


@jax.jit
def _gamma_batch(samples_jax):
    def one(p):
        return salpyter.imf_log_slope(logm_alpha_jax, p, model, -3.0, 4.0)
    return jax.vmap(one)(samples_jax)


# Load the observational compilation once so both passes share it. CSV slope
# convention is dN/dm (Salpeter=2.35); we convert to dN/dlogm via Gamma = -s + 1.
_obs_data = pd.read_csv(ALPHAPLOT_CSV, skip_blank_lines=True)
_obs_data = _obs_data[~np.isnan(_obs_data["Slope (Salpeter=2.35)"])].reset_index(drop=True)
_obs_slope = -np.asarray(_obs_data["Slope (Salpeter=2.35)"]) + 1
_obs_mlow = np.asarray(_obs_data["Lower mass (Msun)"])
_obs_mhi = np.asarray(_obs_data["Upper mass (Msun)"])
_obs_mmed = np.sqrt(_obs_mlow * _obs_mhi)
_obs_slope_err = np.asarray(_obs_data["Slope uncertainty"])
_obs_slope_upper = -np.asarray(_obs_data["Upper limit (1 sigma)"]) + 1
_obs_slope_lower = -np.asarray(_obs_data["Lower limit (1 sigma)"]) + 1
_obs_types = np.asarray(_obs_data["Class"])
# Metallicity [Z/H] is logZ in dex; rows without a measurement are NaN.
_obs_logz = np.asarray(_obs_data["Metallicity [Z/H]"], dtype=float)
_obs_references = _obs_data["Reference"].astype(str).values
_obs_urls = [
    f"https://ui.adsabs.harvard.edu/abs/{r.split(';')[0].strip()}"
    for r in _obs_references
]
_obs_markerdict = {
    "OB Association": "d",
    "Young Cluster": "o",
    "MW Field": ">",
    "MW Bulge": "v",
    "Globular Cluster": "X",
    "MW Nuclear Cluster": "^",
    "UFD": "<",
}
_obs_markers = np.array([_obs_markerdict.get(t, "o") for t in _obs_types])


def _draw_observations(ax, color_by_z=False):
    """Overlay the alphaplot literature compilation (markers + error bars +
    clickable ADS hyperlinks).

    When ``color_by_z`` is True, each marker is colored by its ``[Z/H]``
    metallicity using the simulation's ``logz_to_color`` (a ListedColormap of
    three Dark2 bins anchored at logZ ∈ {-2, -1, 0}; values outside [-2, 0]
    are clamped). Rows with no metallicity measurement render in grey.
    """
    ebar_lw = 0.3

    if color_by_z:
        _missing = ~np.isfinite(_obs_logz)
        _pt_colors = np.array([
            (0.6, 0.6, 0.6, 1.0) if missing else logz_to_color(z)
            for z, missing in zip(_obs_logz, _missing)
        ])
    else:
        _pt_colors = None

    def _scatter(sel, mk, zorder):
        if not sel.any():
            return
        kwargs = dict(s=10, marker=mk, lw=0, zorder=zorder)
        if color_by_z:
            ax.scatter(_obs_mmed[sel], _obs_slope[sel],
                       c=_pt_colors[sel], **kwargs)
        else:
            ax.scatter(_obs_mmed[sel], _obs_slope[sel], c="black", **kwargs)

    # 1) asymmetric errors where both lower/upper bounds are given
    m_asym = np.isfinite(_obs_slope_lower) & np.isfinite(_obs_slope_upper)
    if m_asym.any():
        ax.errorbar(
            _obs_mmed[m_asym], _obs_slope[m_asym],
            xerr=np.array([
                _obs_mmed[m_asym] - _obs_mlow[m_asym],
                _obs_mhi[m_asym] - _obs_mmed[m_asym],
            ]),
            yerr=np.array([
                np.abs(_obs_slope - _obs_slope_lower)[m_asym],
                np.abs(_obs_slope_upper - _obs_slope)[m_asym],
            ]),
            ls="", capsize=0, lw=ebar_lw, marker=None, ecolor="grey",
        )
        for mk in np.unique(_obs_markers):
            _scatter(m_asym & (_obs_markers == mk), mk, zorder=20)

    # 2) symmetric errors otherwise
    m_sym = np.isfinite(_obs_slope_err) & ~m_asym
    if m_sym.any():
        ax.errorbar(
            _obs_mmed[m_sym], _obs_slope[m_sym],
            xerr=np.array([
                _obs_mmed[m_sym] - _obs_mlow[m_sym],
                _obs_mhi[m_sym] - _obs_mmed[m_sym],
            ]),
            yerr=_obs_slope_err[m_sym], ls="", capsize=0, lw=ebar_lw,
            marker=None, ecolor="grey",
        )
        for mk in np.unique(_obs_markers):
            _scatter(m_sym & (_obs_markers == mk), mk, zorder=10)

    # 3) no error info — just point with x-extent
    m_none = ~m_asym & ~m_sym
    if m_none.any():
        ax.errorbar(
            _obs_mmed[m_none], _obs_slope[m_none],
            xerr=np.array([
                _obs_mmed[m_none] - _obs_mlow[m_none],
                _obs_mhi[m_none] - _obs_mmed[m_none],
            ]),
            ls="", capsize=0, lw=ebar_lw, ecolor="grey",
        )
        for mk in np.unique(_obs_markers):
            _scatter(m_none & (_obs_markers == mk), mk, zorder=2)

    # Empty-data proxy markers so the system-class entries appear in the
    # legend without adding off-axis artists that would inflate the figure's
    # tight-bbox calculation. Proxy color stays black so the markers stay
    # readable on the legend regardless of color_by_z.
    for cls, mk in _obs_markerdict.items():
        ax.scatter([], [], marker=mk, lw=0, color="black", s=10, label=cls)

    # Clickable ADS hyperlinks per measurement. PDF viewers like Preview only
    # honor link annotations attached to text artists (not PathCollection
    # URLs), so we add an invisible-but-present text per point: ``color=
    # "none"`` makes the glyph render-transparent, ``clip_on=True`` stops the
    # rendered bbox from inflating the figure's tight bbox if a point lies
    # off the visible axes range.
    for x, y, u in zip(_obs_mmed, _obs_slope, _obs_urls):
        ax.text(
            x, y, "o", url=u, color="none", fontsize=4,
            bbox=dict(boxstyle="circle", url=u, facecolor="none", edgecolor="none"),
            ha="center", va="center", zorder=10000, clip_on=True,
        )


# ---------------- main loop ---------------- #

_VARIANTS = (
    ("none",      False, False),
    ("obs",       True,  False),
    ("obs_zcolor", True, True),
)

for _variant_name, WITH_OBS, COLOR_OBS_BY_Z in _VARIANTS:
    fig, ax = plt.subplots(1, 1, figsize=(4, 4) if WITH_OBS else (4.5, 4.5))

    # ---- dataset envelopes ----
    for run in IMF_RUNS:
        imf_data_path = run.path + "/IMF.dat"
        samples_path = run.path + f"/imf_samples_jax/samples_{model}.npy"
        if not (os.path.isfile(imf_data_path) and os.path.isfile(samples_path)):
            print(f"skip (missing inputs): {run.path}")
            continue

        masses = np.loadtxt(imf_data_path)[:, 1]
        MMIN, MMAX = max(0.1, masses.min()), masses.max()
        fit_masses = masses[(masses > MMIN) & (masses < MMAX)]
        if len(fit_masses) == 0:
            continue

        samples = np.load(samples_path)
        if len(samples) == 0:
            continue

        samples_extrap = np.asarray(samples).copy()
        if samples_extrap.shape[1] >= 5:
            samples_extrap[:, 3] = -3.0
            samples_extrap[:, 4] = 4.0

        gamma = np.asarray(_gamma_batch(jnp.asarray(samples_extrap)))
        mass_mask = logm_alpha[None, :] <= float(np.log10(MMAX))
        gamma = np.where(mass_mask, gamma, np.nan)
        lo_g, mid_g, hi_g = np.nanpercentile(gamma, [16, 50, 84], axis=0)

        color = _color_for(run)
        label = run.short_label
        # Convention: M2e4_R10 entries get dotted lines so they're visually
        # distinguishable from the M2e4_R1 series of the same logZ color.
        ls = "dotted" if run.R_cloud == 10.0 else "solid"
        # On the obs overlay, disambiguate the simulation series from the
        # literature points in the legend.
#        if WITH_OBS:
#            label = "STARFORGE: " + label
        ax.fill_between(mgrid_alpha, lo_g, hi_g, color=color, alpha=0.25)
        ax.plot(mgrid_alpha, mid_g, color=color, lw=1.4, ls=ls, label=label)

    # ---- analytic references ----
    ax.axhline(-1.35, color="black", ls="-.", lw=0.5, zorder=-100)
    ax.text(0.012, -1.55, "Salpeter slope (-1.35)", fontsize=6, color="black")

    chabrier_params = np.asarray(salpyter.CHABRIER_SMOOTH_DEFAULT_PARAMS)
    chabrier_gamma = np.asarray(
        salpyter.imf_log_slope(logm_alpha_jax, chabrier_params, "chabrier_smooth", -3.0, 4.0)
    )
    kroupa_params = np.array([
        0.7, -0.3, -1.3,
        float(np.log10(0.08)), float(np.log10(0.5)),
    ])
    kroupa_gamma = np.asarray(
        salpyter.imf_log_slope(logm_alpha_jax, kroupa_params, "kroupa", -3.0, 4.0)
    )
    mlimit = mgrid_alpha < 150
    ax.plot(mgrid_alpha[mlimit], chabrier_gamma[mlimit], color="black",
            ls="dashed", lw=1.0, label="Chabrier 2005")
    ax.plot(mgrid_alpha[mlimit], kroupa_gamma[mlimit], color="red",
            ls="dashed", lw=1.0, label="Kroupa 2001")

    # ---- observational overlay (only on the _obs variants) ----
    if WITH_OBS:
        _draw_observations(ax, color_by_z=COLOR_OBS_BY_Z)

    # ---- finalize ----
    ax.set(
        xscale="log",
        xlim=[1e-2, 3e3],
        ylim=[-3, 3.5],
        xlabel=r"Stellar Mass $\ (M_\odot)$",
        ylabel=r"IMF slope $\Gamma_{\rm IMF}$",
    )
    if WITH_OBS:
        # Reorder so that all observation-class entries land in the same
        # legend column. With column-major fill and ncol=2, listing the
        # obs first sends them all into col1 and the sims+references into
        # col2 (instead of splitting one obs entry across columns).
        handles, labels = ax.get_legend_handles_labels()
        obs_classes = set(_obs_markerdict.keys())
        obs_idx = [i for i, l in enumerate(labels) if l in obs_classes]
        other_idx = [i for i in range(len(labels)) if i not in obs_idx]
        ordered = obs_idx + other_idx
        leg = ax.legend(
            [handles[i] for i in ordered], [labels[i] for i in ordered],
            loc="upper right", fontsize=6, frameon=False,
            borderaxespad=0, labelspacing=0.2, ncol=1, edgecolor="black",
        )
    else:
        leg = ax.legend(
            loc="upper right", fontsize=8, frameon=False,
            borderaxespad=0, edgecolor="black",
        )
    # Inset logZ colorbar so the reader can decode the obs-point colors.
    # Placed in the bottom-right of the axes (sparse region of the plot) with
    # ticks + label on top so the label doesn't run into the x-axis label.
    if COLOR_OBS_BY_Z:
        _cb_proxy = np.array([[-2.5, 0.5], [-2, 0.5]])
        _img = ax.pcolormesh(_cb_proxy, cmap=LOGZ_COLORMAP)
        _img.set_visible(False)
        _cax = ax.inset_axes([0.05, 0.07, 0.35, 0.03])
        _cb = plt.colorbar(
            _img, cax=_cax,
            label=r"$\log Z/Z_\odot$",
            orientation="horizontal", ticks=[-2, -1, 0],
        )
        _cb.ax.xaxis.set_ticks_position("top")
        _cb.ax.xaxis.set_label_position("top")
        _cb.ax.tick_params(labelsize=6, pad=1)
        _cb.ax.xaxis.label.set_size(6)
        _cb.minorticks_off()

    fig.tight_layout()
    _suffix = ""
    if WITH_OBS:
        _suffix = "_obs" + ("_zcolor" if COLOR_OBS_BY_Z else "")
    out_path = os.path.join(OUTDIR, f"alphaplot{_suffix}_{model}.pdf")

    # For the obs variant, compute each clickable point's normalized
    # coordinate in the saved-PDF frame BEFORE savefig (renderer must still
    # be available). pad_inches=0.1 is matplotlib's default for
    # bbox_inches="tight".
    overlays: list[tuple[float, float, str]] = []
    if WITH_OBS:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        tb = fig.get_tightbbox(renderer)  # inches
        pad = 0.1
        sx0, sy0 = tb.x0 - pad, tb.y0 - pad
        sx1, sy1 = tb.x1 + pad, tb.y1 + pad
        dpi = fig.dpi
        for x, y, url in zip(_obs_mmed, _obs_slope, _obs_urls):
            if not np.isfinite(x) or not np.isfinite(y):
                continue
            px, py = ax.transData.transform((x, y))
            ix, iy = px / dpi, py / dpi
            nx = (ix - sx0) / (sx1 - sx0)
            ny = (iy - sy0) / (sy1 - sy0)
            if 0.0 <= nx <= 1.0 and 0.0 <= ny <= 1.0:
                overlays.append((nx, ny, url))

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")

    # Companion .tex overlay: \includegraphics{...} + TikZ \href hotspots so
    # the in-figure ADS links survive being embedded in manuscript.tex (PDF
    # link annotations are dropped by \includegraphics, but native LaTeX
    # hyperlinks are not). \input this instead of the bare \includegraphics.
    if WITH_OBS and overlays:
        tex_path = out_path[:-4] + ".tex"
        # The overlay is \input'd from manuscript.tex sitting one level up
        # (../manuscript.tex). pdflatex resolves \includegraphics relative to
        # its CWD (the manuscript directory), so the included PDF must be
        # addressed via the same "figures/imf_plots/<name>.pdf" prefix the
        # user already types for \input. If your manuscript lives elsewhere,
        # edit the path here or set \graphicspath in the preamble.
        manuscript_relative_pdf = os.path.join("figures", out_path)
        input_relative = os.path.join("figures", out_path[:-4])
        with open(tex_path, "w") as f:
            f.write(f"% Auto-generated overlay for {os.path.basename(out_path)}.\n")
            f.write("% Requires: hyperref, tikz, graphicx in the preamble.\n")
            f.write(f"% Use as: \\input{{{input_relative}}}\n")
            f.write("\\begin{tikzpicture}\n")
            f.write("  \\node[anchor=south west,inner sep=0pt] (img) at (0,0) {%\n")
            f.write(f"    \\includegraphics[width=\\linewidth]{{{manuscript_relative_pdf}}}%\n")
            f.write("  };\n")
            f.write("  \\begin{scope}[x={(img.south east)},y={(img.north west)}]\n")
            for nx, ny, url in overlays:
                # & is the only common LaTeX-fatal char in ADS bibcodes;
                # percent-encode it so the URL still resolves in browsers.
                safe_url = url.replace("&", "%26").replace("#", "%23")
                f.write(
                    f"    \\node[inner sep=2pt] at ({nx:.4f},{ny:.4f}) "
                    f"{{\\href{{{safe_url}}}{{\\phantom{{x}}}}}};\n"
                )
            f.write("  \\end{scope}\n")
            f.write("\\end{tikzpicture}\n")
        print(f"wrote {tex_path}")
