"""Plots total cluster mass vs. maximum stellar mass, producing an output Mcluster_vs_Mmax_paramspace.pdf and Mcluster_vs_Mmax_physics.pdf"""

import matplotlib
# Force the non-interactive PDF backend before importing pyplot. The default
# macosx backend silently drops the per-point `url=` link annotations we
# attach to the Yan+2023 observation points; Agg preserves them on save.
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import palettable
from palettable.colorbrewer.qualitative import Dark2_3
import numpy as np
from astropy.table import Table
from matplotlib.colors import ListedColormap

OVERWRITE = True
LOGZ_COLORMAP = ListedColormap(Dark2_3.mpl_colors)


def logz_to_color(logz):
    cmap = LOGZ_COLORMAP  # plt.get_cmap("rainbow")

    return cmap((logz - (-2)) / (np.log10(1) - (-2)))


runs_paramspace = {
    #   r"$Z_\odot$ (MW GMC)": "imf_data/STARFORGE_RT/STARFORGE_v1.2/M2e5_R3/M2e5_R30_Z1_S0_A2_B0.1_I1_Res271_n2_sol0.5_42/output",
    # r"$Z_\odot$ Fiducial": "M2e4_R1_Z1_S0_A2_B0.1_I10000_Res271_n2_sol0.5_42/output",
    r"M2e4_R1_Z1": "imf_data/STARFORGE_RT/STARFORGE_v1.2/M2e4_R1/M2e4_R1_Z1_S0_A2_B0.1_I10000_Res271_n2_sol0.5_42/output_turbsphere_driving1",
    r"M2e4_R1_Z0.1": "imf_data/STARFORGE_RT/STARFORGE_v1.2/M2e4_R1/M2e4_R1_Z0.1_S0_A2_B0.1_I10000_Res271_n2_sol0.5_42/output_turbsphere_driving1",
    r"M2e4_R1_Z0.01": "imf_data/STARFORGE_RT/STARFORGE_v1.2/M2e4_R1/M2e4_R1_Z0.01_S0_A2_B0.1_I10000_Res271_n2_sol0.5_42/output_turbsphere_driving1",
    # r"M2e5_R3_Z1": "imf_data/STARFORGE_RT/STARFORGE_v1.2/M2e5_R3/M2e5_R3_Z1_S0_A2_B0.1_I10000_Res271_n2_sol0.5_42/output_turbsphere_driving1",
    # r"M2e5_R3_Z0.1": "imf_data/STARFORGE_RT/STARFORGE_v1.2/M2e5_R3/M2e5_R3_Z0.1_S0_A2_B0.1_I10000_Res271_n2_sol0.5_42/output_turbsphere_driving1",
    # r"M2e5_R3_Z0.01": "imf_data/STARFORGE_RT/STARFORGE_v1.2/M2e5_R3/M2e5_R3_Z0.01_S0_A2_B0.1_I10000_Res271_n2_sol0.5_42/output_turbsphere_driving1",
#    r"M2e4_R10_Z1": "imf_data/STARFORGE_RT/STARFORGE_v1.2/M2e4_R10/M2e4_R10_Z1_S0_A2_B0.1_I1_Res271_n2_sol0.5_42/output",
    r"M2e4_R10_Z1_1": "imf_data/STARFORGE_RT/STARFORGE_v1.2/M2e4_R10/M2e4_R10_Z1_S0_A2_B0.1_I1_Res271_n2_sol0.5_42/output_turbsphere_driving1",
    r"M2e4_R10_Z1_2": "imf_data/STARFORGE_RT/STARFORGE_v1.2/M2e4_R10/M2e4_R10_Z1_S0_A2_B0.1_I1_Res271_n2_sol0.5_42/output_turbsphere_driving2",
    r"M2e4_R10_Z1_3": "imf_data/STARFORGE_RT/STARFORGE_v1.2/M2e4_R10/M2e4_R10_Z1_S0_A2_B0.1_I1_Res271_n2_sol0.5_42/output_turbsphere_driving3",
    r"M2e4_R10_Z1_4": "imf_data/STARFORGE_RT/STARFORGE_v1.2/M2e4_R10/M2e4_R10_Z1_S0_A2_B0.1_I1_Res271_n2_sol0.5_42/output_turbsphere_driving4",
    r"M2e4_R10_Z1_5": "imf_data/STARFORGE_RT/STARFORGE_v1.2/M2e4_R10/M2e4_R10_Z1_S0_A2_B0.1_I1_Res271_n2_sol0.5_42/output_turbsphere_driving5",
    r"M2e4_R10_Z1_6": "imf_data/STARFORGE_RT/STARFORGE_v1.2/M2e4_R10/M2e4_R10_Z1_S0_A2_B0.1_I1_Res271_n2_sol0.5_42/output_turbsphere_driving6",
    r"M2e4_R10_Z1_7": "imf_data/STARFORGE_RT/STARFORGE_v1.2/M2e4_R10/M2e4_R10_Z1_S0_A2_B0.1_I1_Res271_n2_sol0.5_42/output_turbsphere_driving7",
    r"M2e4_R10_Z1_8": "imf_data/STARFORGE_RT/STARFORGE_v1.2/M2e4_R10/M2e4_R10_Z1_S0_A2_B0.1_I1_Res271_n2_sol0.5_42/output_turbsphere_driving8",
#    r"M2e4_R10_Z0.01_1": "imf_data/STARFORGE_RT/STARFORGE_v1.2/M2e4_R10/M2e4_R10_Z0.01_S0_A2_B0.1_I1_Res271_n2_sol0.5_42/output_turbsphere_driving1",
#    r"M2e4_R10_Z0.01_2": "imf_data/STARFORGE_RT/STARFORGE_v1.2/M2e4_R10/M2e4_R10_Z0.01_S0_A2_B0.1_I1_Res271_n2_sol0.5_42/output_turbsphere_driving2",
#    r"M2e4_R10_Z0.01_3": "imf_data/STARFORGE_RT/STARFORGE_v1.2/M2e4_R10/M2e4_R10_Z0.01_S0_A2_B0.1_I1_Res271_n2_sol0.5_42/output_turbsphere_driving3",
#    r"M2e4_R10_Z0.01": "imf_data/STARFORGE_RT/STARFORGE_v1.2/M2e4_R10/M2e4_R10_Z0.01_S0_A2_B0.1_I1_Res271_n2_sol0.5_42/output",
    # r"M2e4_R10_Z0.1": "imf_data/STARFORGE_RT/STARFORGE_v1.2/M2e4_R10/M2e4_R10_Z0.1_S0_A2_B0.1_I1_Res271_n2_sol0.5_42/output",
#    r"M2e4_R3_Z1": "imf_data/STARFORGE_RT/STARFORGE_v1.2/M2e4_R3/M2e4_R3_Z1_S0_A2_B0.1_I1_Res271_n2_sol0.5_42/output",
    # r"M2e4_R10_Z0.01_2": "imf_data/STARFORGE_RT/STARFORGE_v1.2/M2e4_R10/M2e4_R10_Z0.01_S0_A2_B0.1_I1_Res271_n2_sol0.5_42/output_turbsphere_driving2",
    # r"M2e4_R10_Z0.01_3": "imf_data/STARFORGE_RT/STARFORGE_v1.2/M2e4_R10/M2e4_R10_Z0.01_S0_A2_B0.1_I1_Res271_n2_sol0.5_42/output_turbsphere_driving3",
    # r"$Z_\odot$ M2e5_R3": "M2e5_R3_Z1_S0_A2_B0.1_I10000_Res271_n2_sol0.5_42/output",
    # r"$1\% Z_\odot$ M2e5_R3": "M2e5_R3_Z0.01_S0_A2_B0.1_I10000_Res271_n2_sol0.5_42/output",
    # r"$Z_\odot$ No Winds": "M2e4_R1_Z1_S0_A2_B0.1_I10000_Res271_n2_sol0.5_42/output_nowind",
    # r"$Z_\odot$ No IR Rad. Pressure": "M2e4_R1_Z1_S0_A2_B0.1_I10000_Res271_n2_sol0.5_42/output_noIR",
    # r"$Z_\odot$ Stronger Winds": "M2e4_R1_Z1_S0_A2_B0.1_I10000_Res271_n2_sol0.5_42/output_vink"
    #    r"$1\% Z_\odot$": "M2e4_R1_Z0.01_S0_A2_B0.1_I10000_Res271_n2_sol0.5_42/output",
    #    r"$Z_\odot$ (Milky Way GMC, $2e5M_\odot$, No Rad. Pressure)": "/home/mgrudic/code/hiz/STARFORGE_v1.1/M2e5_R3/M2e5_R30/M2e5_R3/M2e5_R30_S0_T1_B0.01_Res271_n2_sol0.5_42/output/",
}

# Physics experiments: same M2e4_R1 Z=1 cloud rerun with feedback channels
# toggled. Single source of truth for legend label, simulation path, and
# scatter marker — the plotting loop and the legend proxies both read this.
_PHYS = "imf_data/STARFORGE_RT/STARFORGE_v1.2/M2e4_R1/M2e4_R1_Z1_S0_A2_B0.1_I10000_Res271_n2_sol0.5_42"
runs_physics = {
    # label                           : (path,                  marker)
    r"No Winds":                        (f"{_PHYS}/output_nowind", "s"),
    r"No IR Rad. Pressure":             (f"{_PHYS}/output_noIR",   "D"),
    r"$3\times$ Stronger Winds":        (f"{_PHYS}/output_vink",   "^"),
}


fig, ax = plt.subplots(figsize=(4, 4))
ax.set_prop_cycle("color", palettable.colorbrewer.qualitative.Dark2_8.mpl_colors)
for run, path in runs_paramspace.items():
    logz = 0
    if "Z0.1" in path:
        logz = -1
    elif "Z0.01" in path:
        logz = -2

    # print(run, path)
    # continue

    # stats_to_compute = ("stellar_mass_sum", "stellar_mass_max")
    # stats = get_globalstats_of_simulation(path + "/stars", overwrite=OVERWRITE, stats_to_compute=stats_to_compute)
    label = run
    if "M2e4_R10" in run:
        if run == "M2e4_R10_Z1_1":
            label = "M2e4_R10_Z1"
        elif run == "M2e4_R10_Z0.01_1":
            label = "M2e4_R10_Z0.01"
        else:
            label = ""

    ls = "solid"
    if "M2e4_R10" in run:
        ls = "dashdot"
    elif "M2e5" in run:
        ls = "dashed"

    stats = Table.read(path + "/global_statistics.fits")
    t = stats["time"]
    stats = stats[t.argsort()]
    # np.savetxt(f"{run}_mcluster_vs_mmax.dat", np.c_[stats["stellar_mass_sum"], stats["stellar_mass_max"]])
    # np.savetxt("m.dat", np.c_[stats["stellar_mass_sum"], stats["stellar_mass_max"]])
    color = logz_to_color(logz)

    s = 20
    # if "M2e4_R10" in run:
    #     # alpha = 0.5
    #     s *= 1
    # elif "M2e4_R3" in run:
    #     # alpha = 0.75
    #     s *= 2
    # else:
    #     s *= 4
    # alpha = 1.0

    print(run)
    # Circles always mark the fiducial physics model. Physics-experiment
    # variants get distinct markers in the loop below.
    marker = "o"

    if "M2e4_R1_" in run:
        ax.loglog(
            stats["stellar_mass_sum"],
            stats["stellar_mass_max"],
            #   label=label,
            ls="dotted",
            color=color,
            lw=1,
        )
    ax.scatter(
        [stats["stellar_mass_sum"].max()],
        [stats["stellar_mass_max"].max()],
        color=color,
        # label=(label if "M2e4_R10" in run else None),
        s=s,
        marker=marker,  # ("s" if "M2e4_R10" in run else None),
        edgecolor="black",
        lw=0.3,
        zorder=1000,
        # alpha=alpha,
    )
    # print(run, stats["stellar_mass_sum"][stats["stellar_mass_max"].argmax()], stats["stellar_mass_max"].max())

# Physics-experiment overlay: all variants share the Z=1 color from the
# colormap; the marker (carried in runs_physics) disambiguates which feedback
# channel was toggled.
for run, (path, marker) in runs_physics.items():
    stats = Table.read(path + "/global_statistics.fits")
    stats = stats[stats["time"].argsort()]
    color = logz_to_color(0)
    ax.loglog(
        stats["stellar_mass_sum"], stats["stellar_mass_max"],
        ls="dashed", color=color, lw=1, alpha=0.6,
    )
    ax.scatter(
        [stats["stellar_mass_sum"].max()],
        [stats["stellar_mass_max"].max()],
        color=color, s=30, marker=marker,
        edgecolor="black", lw=0.3, zorder=1000,
    )

# --------------------------------------------------------------------------- #
# Observational compilation: Yan, Jerabkova, Kroupa 2023 (A&A 670 A151) Table 2.
# Data live in yan23_table2.fits next to this script (100 young embedded clusters
# with mmax and Mecl, asymmetric log-space uncertainties). All points share the
# same source paper, so each clickable hotspot links to that ADS bibcode.
# --------------------------------------------------------------------------- #
obs_full = Table.read("yan23_table2.fits")
# Drop the Yan/Massey & Hunter R136 entry — it gets plotted in the unified
# red-R136 overlay below alongside the alternative measurements.
_is_r136 = obs_full["designation"] == "No.139"
obs = obs_full[~_is_r136]
obs_Mecl = 10.0 ** obs["log_Mecl"]
obs_mmax = 10.0 ** obs["log_mmax"]
obs_xerr = np.vstack([
    obs_Mecl - 10.0 ** (obs["log_Mecl"] - obs["log_Mecl_lo"]),
    10.0 ** (obs["log_Mecl"] + obs["log_Mecl_up"]) - obs_Mecl,
])
obs_yerr = np.vstack([
    obs_mmax - 10.0 ** (obs["log_mmax"] - obs["log_mmax_lo"]),
    10.0 ** (obs["log_mmax"] + obs["log_mmax_up"]) - obs_mmax,
])
# Render observations behind the simulation points (low zorder).
ax.errorbar(
    obs_Mecl, obs_mmax, xerr=obs_xerr, yerr=obs_yerr,
    fmt="x", color="gray", ecolor="lightgray",
    markersize=4, mew=0.5, elinewidth=0.4, capsize=0, alpha=0.8,
    zorder=0, label="Yan+2023 (obs)",
)
# Clickable ADS hyperlink per data point. WKP13 is itself a compilation paper,
# so the FITS carries a `primary_bibcode` column resolved one level deeper:
# each row links to the actual original observation paper (Carpenter+1993,
# Borissova+2005, Testi+1999, etc.). Where WKP13's primary reference is a
# book chapter or conference abstract with no journal bibcode, we fall back
# to the WKP13 paper itself (4 of 100 rows). PDF viewers honor link
# annotations attached to text artists, so we drop an invisible-but-present
# glyph at each point. The arxiv-friendly .tex overlay generated below
# re-creates the same hotspots in native LaTeX for the manuscript build.
def _ads_url(bibcode):
    return "https://ui.adsabs.harvard.edu/abs/{}/abstract".format(
        bibcode.replace("&", "%26")
    )
obs_urls = [_ads_url(b) for b in obs["primary_bibcode"]]
for x, y, u in zip(obs_Mecl, obs_mmax, obs_urls):
    ax.text(
        x, y, "o", url=u, color="none", fontsize=4,
        bbox=dict(boxstyle="circle", url=u,
                  facecolor="none", edgecolor="none"),
        ha="center", va="center", zorder=10000, clip_on=True,
    )

# R136 measurements (all in red, single legend entry). Massey & Hunter 1998
# is the value Yan+2023 inherits via WKP13; the other three are post-2010
# re-analyses that put the most massive R136 stars well above the canonical
# 150 Msun limit: Crowther+2010 (R136a1 initial mass 320 +100/-40 Msun),
# Schneider+2018 (IMF densely sampled up to 200 Msun in the surrounding 30
# Doradus region), Higgins+2022 (evolutionary models span 250-1000 Msun for
# R136 WNh stars). All four points share the R136 cluster mass.
# R136 cluster mass is itself uncertain by a factor of ~4 across the
# literature. The lower bound 5.5e4 Msun is from Hunter et al. 1995 (also
# quoted in Crowther+2010 Table 7) and is the inner-cluster (R136a/b) mass.
# The upper bound 2.2e5 Msun is Yan+2023 / WKP13 No.139's value, which
# refers to a broader region (NGC 2070 inner core / 30 Doradus central
# stellar content). We plot the markers at the geometric center of that
# range with a symmetric ±0.30 dex x-error bar spanning the full factor-
# of-4 — making the systematic on the cluster-mass axis explicit.
_R136_MECL_LO = 5.5e4   # Hunter+1995 / Crowther+2010, inner-cluster
_R136_MECL_HI = 2.2e5   # Yan+2023 / WKP13, broader region
R136_LOG_MECL = 0.5 * (np.log10(_R136_MECL_LO) + np.log10(_R136_MECL_HI))
R136_LOG_MECL_ERR = 0.5 * (np.log10(_R136_MECL_HI) - np.log10(_R136_MECL_LO))
# All values below are mass estimates each paper actually publishes for the
# most massive star in the R136 region (initial mass unless noted):
#   - M&H 1998:    125 +25/-45 Msun, the value Yan+2023 inherits via WKP13 —
#                  asymmetric ± from WKP13's "±0.5 spectral subclass" rule
#                  (their Appendix A), linearized from log_mmax = 2.098
#                  +0.078/-0.195 in Yan's Table 2.
#   - Crowther+2010: ZAMS mass of R136a1 = 320 +100/-40 Msun (their abstract).
#   - Schneider+2018: VFTS 1025 (R136c), initial mass 203 +40/-44 Msun (their
#                     Sect. S7.4 / Table S3). The R136 core itself was excluded
#                     from the VFTS sample.
#   - Higgins+2022: their hydrogen-clock analysis can't distinguish initial
#                   masses across 250-1000 Msun for R136 WNh stars; plotted
#                   as a range bar (no central marker) over that span.
#   - Keszthelyi+2025: R136a1 initial mass 346 ± 41 Msun (their abstract).
R136_POINTS = [
    # (label,                mmax,  +up,   -lo,    bibcode)
    ("Massey&Hunter+1998",   125,    25,   45,   "1998ApJ...493..180M"),
    ("Crowther+2010",        320,   100,   40,   "2010MNRAS.408..731C"),
#    ("Schneider+2018",       203,    40,   44,   "2018Sci...359...69S"),
    # Bestenlehner+2020 MNRAS 499 1918: current spectroscopic mass from
    # HST/STIS + CMFGEN modeling (Paper II of the R136 STIS series).
    ("Bestenlehner+2020",    251,    48,   31,   "2020MNRAS.499.1918B"),
    # Brands+2022 A&A 663 A36 (Paper III): initial mass 273 +25/-36 from
    # HST/STIS UV+optical fits with FASTWIND + Kiwi-GA genetic algorithm.
    ("Brands+2022",          273,    25,   36,   "2022A&A...663A..36B"),
    ("Keszthelyi+2025",      346,    41,   41,   "2025A&A...700A.186K"),
]
# Higgins+2022 range bar: 250-1000 Msun, no specific central value.
# HIGGINS22 = ("Higgins+2022", 250, 1000, "2022MNRAS.516.4052H")
_r136_mecl = 10.0 ** R136_LOG_MECL
_r136_xerr = np.array([
    [_r136_mecl - 10.0 ** (R136_LOG_MECL - R136_LOG_MECL_ERR)],
    [10.0 ** (R136_LOG_MECL + R136_LOG_MECL_ERR) - _r136_mecl],
])
for _i, (label, mmax, dup, dlo, bib) in enumerate(R136_POINTS):
    ax.errorbar(
        [_r136_mecl], [mmax],
        xerr=_r136_xerr, yerr=[[dlo], [dup]],
        fmt="*", color="red", ecolor="red",
        markersize=8, mew=0.3, elinewidth=0.5, capsize=0,
        markeredgecolor="black", zorder=5,
        label="R136a1 (obs)" if _i == 0 else None,
    )
    u = _ads_url(bib)
    ax.text(
        _r136_mecl, mmax, "o", url=u, color="none", fontsize=4,
        bbox=dict(boxstyle="circle", url=u, facecolor="none", edgecolor="none"),
        ha="center", va="center", zorder=10000, clip_on=True,
    )

# Higgins+2022: range bar (no central marker) spanning 250-1000 Msun, the
# initial-mass span their hydrogen-clock analysis can't tighten for R136 WNh
# stars. Hyperlink hotspot at the geometric midpoint.
# _, _h_lo, _h_hi, _h_bib = HIGGINS22
# ax.vlines(_r136_mecl, _h_lo, _h_hi, color="red", lw=0.8, zorder=5)
# _h_mid = (_h_lo * _h_hi) ** 0.5
# _u = _ads_url(_h_bib)
# ax.text(
#     _r136_mecl, _h_mid, "o", url=_u, color="none", fontsize=4,
#     bbox=dict(boxstyle="circle", url=_u, facecolor="none", edgecolor="none"),
#     ha="center", va="center", zorder=10000, clip_on=True,
# )

# Legend proxies for the marker scheme: circle = fiducial, the rest pick out
# individual physics experiments. Metallicity is color-encoded so it stays
# on the colorbar, not in the legend.
ax.scatter([], [], marker="o", color="black", s=30, edgecolor="black",
           lw=0.3, label="Fiducial Model")
for _label, (_, _mk) in runs_physics.items():
    ax.scatter([], [], marker=_mk, color="black", s=30, edgecolor="black",
               lw=0.3, label=_label)

ax.legend(loc="lower right", labelspacing=0.2, frameon=True, fontsize=7,
          edgecolor="black")
ax.set(
    ylabel=r"$M_{\rm max}\,\left(M_\odot\right)$",
    xlabel=r"$M_{\rm cluster}\,\left(M_\odot\right)$",
    xlim=[100, 3e5],
    # ymin reaches the Yan+2023 low-mass tail (~2 Msun within xlim, plus error
    # bars). ymax holds the simulation maxima.
    ylim=[3, 2400],
    xscale="log",
    yscale="log",
)

a = np.array([[-2.5, 0.5], [-2, 0.5]])
img = ax.pcolormesh(a, cmap=LOGZ_COLORMAP)
img.set_visible(False)
c = plt.colorbar(
    # img, cax=fig.add_axes([0.15, 0.8, 0.3, 0.05]), label=r"$\log Z$", orientation="horizontal", ticks=[-2, -1, 0]
    img,
    cax=fig.add_axes([0.15, 0.8, 0.3, 0.05]),  # fig.add_axes([0.15, 0.5, 0.3, 0.05]),
    label=r"$\log Z/Z_\odot$",
    orientation="horizontal",
    ticks=[-2, -1, 0],
)
c.minorticks_off()

# Compute the obs-point screen positions in the saved-PDF frame BEFORE savefig
# (renderer must still be live). Pad matches matplotlib's default for
# bbox_inches="tight". Same trick used in imf_alphaplot.py.
fig.canvas.draw()
_renderer = fig.canvas.get_renderer()
_tb = fig.get_tightbbox(_renderer)
_pad = 0.1
_sx0, _sy0 = _tb.x0 - _pad, _tb.y0 - _pad
_sx1, _sy1 = _tb.x1 + _pad, _tb.y1 + _pad
_dpi = fig.dpi
# Walk every clickable point. The Yan compilation lives in (obs_Mecl,
# obs_mmax, obs_urls); the R136 markers are added separately above and were
# previously missing from the overlay (which broke their clickability when
# the figure is \input'd into the manuscript).
_clickables = list(zip(obs_Mecl, obs_mmax, obs_urls))
_clickables.extend(
    (_r136_mecl, mmax, _ads_url(bib)) for _, mmax, _, _, bib in R136_POINTS
)
overlays = []
for x, y, u in _clickables:
    px, py = ax.transData.transform((x, y))
    nx = (px / _dpi - _sx0) / (_sx1 - _sx0)
    ny = (py / _dpi - _sy0) / (_sy1 - _sy0)
    if 0.0 <= nx <= 1.0 and 0.0 <= ny <= 1.0:
        overlays.append((nx, ny, u))

def _write_tex_overlay(out_pdf, overlays):
    """Write the TikZ \\href hotspot overlay companion file for the saved PDF."""
    if not overlays: return
    tex_path = out_pdf.replace(".pdf", ".tex")
    manuscript_relative_pdf = f"figures/{out_pdf}"
    with open(tex_path, "w") as _f:
        _f.write(f"% Auto-generated overlay for {out_pdf}.\n")
        _f.write("% Requires hyperref, tikz, graphicx in the preamble.\n")
        _f.write(f"% Use as: \\input{{figures/{out_pdf[:-4]}}}\n")
        _f.write("\\begin{tikzpicture}\n")
        _f.write("  \\node[anchor=south west,inner sep=0pt] (img) at (0,0) {%\n")
        _f.write(f"    \\includegraphics[width=\\linewidth]{{{manuscript_relative_pdf}}}%\n")
        _f.write("  };\n")
        _f.write("  \\begin{scope}[x={(img.south east)},y={(img.north west)}]\n")
        for nx, ny, u in overlays:
            safe_url = u.replace("#", "%23")
            _f.write(
                f"    \\node[inner sep=2pt] at ({nx:.4f},{ny:.4f}) "
                f"{{\\href{{{safe_url}}}{{\\phantom{{x}}}}}};\n"
            )
        _f.write("  \\end{scope}\n")
        _f.write("\\end{tikzpicture}\n")
    print(f"wrote {tex_path}")


def _normalize_overlays(clickables, ax, fig, sx0, sy0, sx1, sy1, dpi):
    """Convert (x, y, url) data tuples to normalized figure-coord overlays for the .tex hotspots."""
    out = []
    for x, y, u in clickables:
        px, py = ax.transData.transform((x, y))
        nx = (px / dpi - sx0) / (sx1 - sx0)
        ny = (py / dpi - sy0) / (sy1 - sy0)
        if 0.0 <= nx <= 1.0 and 0.0 <= ny <= 1.0:
            out.append((nx, ny, u))
    return out


# ---- (1) Save the baseline (Yan compilation + R136) PDF first.
out_pdf = "Mcluster_vs_Mmax.pdf"
plt.savefig(out_pdf, bbox_inches="tight")
_write_tex_overlay(out_pdf, overlays)

# ---- (2) Now overlay the modern-compilation alternative cluster-mass measurements
# (Hosek+2019, Wright+2015, Bonatto+2006, etc. from yan23_alt_measurements.fits),
# re-save as Mcluster_vs_Mmax_alt.pdf with its own .tex companion. Each modern
# measurement is plotted as a dark-blue open square at its Mecl value; mmax uses
# the paper's quoted value when available, else falls back to the Yan-row mmax
# for that cluster (so the marker sits on the same y as the gray X).
ALT_FITS = "yan23_alt_measurements.fits"
alt_obs = Table.read(ALT_FITS)
# Yan-row mmax per cluster, for rows where the modern paper doesn't quote mmax.
YAN_MMAX = {  # Msun, linear
    "R136": 125.31, "Arches": 110.92, "Cyg OB2": 92.04, "NGC 6611": 61.66,
    "Trumpler 14": 99.77, "NGC 6357": 65.77, "IC 1805": 57.02,
}
alt_clickables = []
for row in alt_obs:
    cname = str(row["cluster_name"])
    if cname == "R136":
        continue  # already plotted as red stars above; don't double-mark
    # Only plot well-determined "value" measurements. Upper/lower bound rows
    # (Figer+2002's velocity-dispersion upper limit on Arches; Lima+2014's
    # saturation-affected direct counts in NGC 6357) live in the FITS for the
    # record but are excluded from the visual overlay to avoid misleading the
    # eye — they're not apples-to-apples with the IMF-extrapolated values.
    if str(row["Mecl_type"]) != "value":
        continue
    Mecl = float(row["Mecl"])
    if not np.isfinite(Mecl):
        continue
    # asymmetric x error bars in linear Msun from the stored log-dex offsets
    up_dex = float(row["Mecl_up_dex"]) if np.isfinite(row["Mecl_up_dex"]) else 0.0
    lo_dex = float(row["Mecl_lo_dex"]) if np.isfinite(row["Mecl_lo_dex"]) else 0.0
    xerr_up = Mecl * (10.0 ** up_dex - 1.0) if up_dex > 0 else 0.0
    xerr_lo = Mecl * (1.0 - 10.0 ** (-lo_dex)) if lo_dex > 0 else 0.0
    mmax = float(row["mmax"]) if np.isfinite(row["mmax"]) else YAN_MMAX.get(cname, np.nan)
    if not np.isfinite(mmax):
        continue
    ax.errorbar(
        [Mecl], [mmax],
        xerr=[[xerr_lo], [xerr_up]],
        fmt="s", mfc="none", mec="navy", color="navy",
        markersize=5, mew=0.8, elinewidth=0.5, capsize=0, zorder=6,
    )
    u = _ads_url(str(row["bibcode"]))
    ax.text(
        Mecl, mmax, "o", url=u, color="none", fontsize=4,
        bbox=dict(boxstyle="circle", url=u, facecolor="none", edgecolor="none"),
        ha="center", va="center", zorder=10000, clip_on=True,
    )
    alt_clickables.append((Mecl, mmax, u))
# legend proxy for the modern-compilation overlay
ax.scatter([], [], marker="s", facecolor="none", edgecolor="navy", s=25,
           lw=0.8, label="Modern reanalyses")
# Rebuild the legend so the new "Modern reanalyses" proxy appears.
_old_legend = ax.get_legend()
if _old_legend is not None:
    _old_legend.remove()
ax.legend(loc="lower right", labelspacing=0.2, frameon=True, fontsize=7,
          edgecolor="black")
# Recompute screen positions (legend changed = tight bbox shifted) and append
# the alt clickables to the overlay list.
fig.canvas.draw()
_tb2 = fig.get_tightbbox(fig.canvas.get_renderer())
_sx0a, _sy0a = _tb2.x0 - _pad, _tb2.y0 - _pad
_sx1a, _sy1a = _tb2.x1 + _pad, _tb2.y1 + _pad
overlays_alt = _normalize_overlays(
    _clickables + alt_clickables, ax, fig,
    _sx0a, _sy0a, _sx1a, _sy1a, fig.dpi,
)
out_pdf_alt = "Mcluster_vs_Mmax_alt.pdf"
plt.savefig(out_pdf_alt, bbox_inches="tight")
_write_tex_overlay(out_pdf_alt, overlays_alt)
