"""Alpha plot: posterior-percentile envelope of the IMF slope Gamma vs M.

Generates two PDFs in one run:
  * ``alphaplot_<model>.pdf``      — bare plot (no observational overlay)
  * ``alphaplot_obs_<model>.pdf``  — same plot + literature compilation from
                                     ``~/code/alphaplot/alphaplot.csv``

Switching between them is a ``WITH_OBS`` toggle inside the loop; all the
shared infrastructure (dataset envelopes, Chabrier/Kroupa references,
axes/legend styling) lives in one place so changes affect both outputs.
"""

import math
import os
import re
from glob import glob

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import salpyter
from palettable.colorbrewer.qualitative import Dark2_3


# ---------------- inputs ---------------- #

ALPHAPLOT_CSV = os.path.expanduser("./alphaplot.csv")
imfs_to_plot = (
    [
        "./imf_data/STARFORGE_RT/"
        "STARFORGE_v1.2/M2e4_R10/M2e4_R10_Z1_S0_A2_B0.1_I1_Res271_n2_sol0.5_42/output_all"
    ]
    + sorted(
        glob(
            "./imf_data/STARFORGE_RT/"
            "STARFORGE_v1.2/M2e4_R1/*/output*"
        )
    )
)

model = "chabrier_smooth_bounds"

logm_alpha = np.linspace(-2, 3.5, 601)
mgrid_alpha = 10**logm_alpha
logm_alpha_jax = jnp.asarray(logm_alpha)

OUTDIR = "imf_plots"
os.makedirs(OUTDIR, exist_ok=True)


# ---------------- helpers ---------------- #

DARK2_3 = Dark2_3.mpl_colors


def _color_for(run, dark2_idx_holder=[0]):
    sim_dir = os.path.normpath(run).split(os.sep)[-2]
    if "M2e4_R10" in sim_dir:
        return "black"
    c = DARK2_3[dark2_idx_holder[0] % len(DARK2_3)]
    dark2_idx_holder[0] += 1
    return c


def _short_label(run):
    parts = os.path.normpath(run).split(os.sep)
    sim_dir = parts[-2] if len(parts) >= 2 else parts[-1]
    out_dir = parts[-1]
    m_m = re.search(r"M([\d.]+(?:e[+-]?\d+)?)_", sim_dir)
    m_r = re.search(r"_R([\d.]+)_", sim_dir)
    m_z = re.search(r"_Z([\d.]+?)_", sim_dir)
    if m_m and m_r:
        sigma = float(m_m.group(1)) / (np.pi * float(m_r.group(1)) ** 2)
        exp = math.floor(math.log10(abs(sigma)))
        sigma_2sf = round(sigma, -int(exp) + 1)
        sigma_str = f"{sigma_2sf:.0f}"
    else:
        sigma_str = "?"
    z_val = m_z.group(1) if m_z else "?"
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


def _draw_observations(ax):
    """Overlay the alphaplot literature compilation (markers + error bars +
    clickable ADS hyperlinks)."""
    ebar_lw = 0.3

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
            sel = m_asym & (_obs_markers == mk)
            if sel.any():
                ax.scatter(_obs_mmed[sel], _obs_slope[sel],
                           c="black", s=10, marker=mk, lw=0, zorder=20)

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
            sel = m_sym & (_obs_markers == mk)
            if sel.any():
                ax.scatter(_obs_mmed[sel], _obs_slope[sel],
                           c="black", s=10, marker=mk, lw=0, zorder=10)

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
            sel = m_none & (_obs_markers == mk)
            if sel.any():
                ax.scatter(_obs_mmed[sel], _obs_slope[sel],
                           c="black", s=10, marker=mk, lw=0, zorder=2)

    # Empty-data proxy markers so the system-class entries appear in the
    # legend without adding off-axis artists that would inflate the figure's
    # tight-bbox calculation.
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

for WITH_OBS in (False, True):
    # Reset the Dark2 color index so the per-dataset colors match across PDFs.
    _color_for.__defaults__ = (([0],))  # type: ignore[attr-defined]

    fig, ax = plt.subplots(1, 1, figsize=(4, 4) if WITH_OBS else (4.5, 4.5))

    # ---- dataset envelopes ----
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
        label = _short_label(run)
        # On the obs overlay, disambiguate the simulation series from the
        # literature points in the legend.
#        if WITH_OBS:
#            label = "STARFORGE: " + label
        ax.fill_between(mgrid_alpha, lo_g, hi_g, color=color, alpha=0.25)
        ax.plot(mgrid_alpha, mid_g, color=color, lw=1.4, label=label)

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
    ax.plot(mgrid_alpha[mlimit], chabrier_gamma[mlimit], color="darkblue",
            ls="dotted", lw=1.0, label="Chabrier 2005")
    ax.plot(mgrid_alpha[mlimit], kroupa_gamma[mlimit], color="red",
            ls="dotted", lw=1.0, label="Kroupa 2001")

    # ---- observational overlay (only on the _obs variant) ----
    if WITH_OBS:
        _draw_observations(ax)

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
    fig.tight_layout()
    out_path = os.path.join(
        OUTDIR, f"alphaplot{'_obs' if WITH_OBS else ''}_{model}.pdf"
    )

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
