"""SFE vs t/t_ff for the same sims plotted in imf_plots.py / mclusters_vs_mmax.py.

SFE = stellar_mass_sum / M_cloud_initial, where M_cloud_initial is read off the
directory name (``M<x>e<y>``). This is equivalent to M* / (M* + M_gas) under
mass conservation in the box. Per-sim, t is shifted so that the earliest
snapshot is t=0 and normalized by the cloud freefall time
t_ff = sqrt(3*pi/(32*G*rho)) with rho = 3*M/(4*pi*R^3).

Colors: M2e4_R10 → black; M2e4_R1 Z={0.01, 0.1, 1} → Dark2_3[0..2] via
logz_to_color (same assignment as the Mmax / SFE-vs-Sigma plots).

Run list comes from sim_paths.FIDUCIAL_RUNS.
"""

import os

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

import numpy as np
from astropy.table import Table

from sim_paths import FIDUCIAL_RUNS, logz_to_color


# G in (pc^3 Msun^-1 Myr^-2). Times in global_statistics.fits are in Myr.
G_PC_MSUN_MYR = 4.49849e-3


def freefall_time(M_msun, R_pc):
    rho = 3.0 * M_msun / (4.0 * np.pi * R_pc**3)
    return np.sqrt(3.0 * np.pi / (32.0 * G_PC_MSUN_MYR * rho))


fig, ax = plt.subplots(figsize=(4, 4))


def _z_str(logZ):
    if logZ == 0:
        return r"Z_\odot"
    return rf"{10.0 ** logZ:g}\,Z_\odot"


# Count R10 ensemble sizes per metallicity so the first entry of each group
# can be labeled e.g. "M2e4_R10, Z=Z_sun (8 realizations)".
r10_counts = {}
for r in FIDUCIAL_RUNS:
    if r.R_cloud == 10.0:
        r10_counts[r.logZ] = r10_counts.get(r.logZ, 0) + 1

# Color by logZ uniformly. Distinguish R1 vs R10 by linestyle/linewidth so
# the metallicity color stays meaningful across cloud sizes.
seen_groups = set()
for run in FIDUCIAL_RUNS:
    stats_path = f"{run.path}/global_statistics.fits"
    if not os.path.isfile(stats_path):
        print(f"skip (no global_statistics.fits): {run.path}")
        continue

    stats = Table.read(stats_path)
    stats.sort("time")
    t = np.asarray(stats["time"])
    mstar = np.maximum.accumulate(np.asarray(stats["stellar_mass_sum"]))

    t_ff = freefall_time(run.M_cloud, run.R_cloud)
    x = (t - t[0]) / t_ff
    sfe = mstar / run.M_cloud

    is_r10 = run.R_cloud == 10.0
    color = logz_to_color(run.logZ)
    lw, ls = (1, "dotted") if is_r10 else (2, "solid")

    key = (run.R_cloud, run.logZ)
    if key in seen_groups:
        label = None
    elif is_r10:
        label = rf"M2e4_R10, $Z={_z_str(run.logZ)}$ ({r10_counts[run.logZ]} realizations)"
        seen_groups.add(key)
    else:
        label = run.label
        seen_groups.add(key)

    ax.plot(x, sfe, color=color, lw=lw, ls=ls, label=label)

# Power-law reference lines anchored at a common point, fanning out.
_x0, _y0 = 1.5, 1e-3
_x_ref = np.array([_x0, _x0 * (0.7 / 0.15) ** 0.5])  # half log-span of original
for _n, _lab in [(1, r"$\propto t$"), (2, r"$\propto t^2$"), (3, r"$\propto t^3$")]:
    _y_ref = _y0 * (_x_ref / _x0) ** _n
    ax.plot(_x_ref, _y_ref, color="black", lw=0.8, ls="--", zorder=0)
    ax.text(_x_ref[-1] * 1.05, _y_ref[-1], _lab, fontsize=7, color="black", va="center")

ax.set(
    xlabel=r"$t / t_{\rm ff}$",
    ylabel=r"$\mathrm{SFE} = M_\star / (M_\star + M_{\rm gas})$",
    xscale="log",
    yscale="log",
    xlim=(1e-1, 5),
    ylim=(1e-4, 1),
)
# M2e4_R10 entries first (low-Sigma cloud), then M2e4_R1.
_handles, _labels = ax.get_legend_handles_labels()
_r10 = [(h, lab) for h, lab in zip(_handles, _labels) if "M2e4_R10" in lab]
_r1 = [(h, lab) for h, lab in zip(_handles, _labels) if "M2e4_R10" not in lab]
_ordered = _r10 + _r1
leg = ax.legend(
    [h for h, _ in _ordered], [lab for _, lab in _ordered],
    loc="upper left", fontsize=8, edgecolor="black",
    borderaxespad=0.4, labelspacing=0,
)
leg.get_frame().set_linewidth(0.8)
fig.tight_layout()

out_pdf = "sfe_vs_time.pdf"
fig.savefig(out_pdf, bbox_inches="tight")
plt.close(fig)
print(f"wrote {out_pdf}")
