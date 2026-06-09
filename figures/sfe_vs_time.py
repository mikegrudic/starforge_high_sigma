"""SFE vs t/t_ff for the same sims plotted in imf_plots.py / mclusters_vs_mmax.py.

SFE = stellar_mass_sum / M_cloud_initial, where M_cloud_initial is read off the
directory name (``M<x>e<y>``). This is equivalent to M* / (M* + M_gas) under
mass conservation in the box. Per-sim, t is shifted so that the earliest
snapshot is t=0 and normalized by the cloud freefall time
t_ff = sqrt(3*pi/(32*G*rho)) with rho = 3*M/(4*pi*R^3).

Colors track imf_plots.py: M2e4_R10 → black; M2e4_R1 Z={0.01, 0.1, 1} →
Dark2_3[0..2] in sorted order (same assignment as the IMF / Mmax plots).
"""

import os

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

import numpy as np
from astropy.table import Table
from palettable.colorbrewer.qualitative import Dark2_3


# G in (pc^3 Msun^-1 Myr^-2). Times in global_statistics.fits are in Myr.
G_PC_MSUN_MYR = 4.49849e-3

DARK2 = Dark2_3.mpl_colors

# Same logZ -> color mapping as mclusters_vs_mmax.py / imf_plots.py.
LOGZ_COLORS = {-2: DARK2[0], -1: DARK2[1], 0: DARK2[2]}

BASE = "imf_data/STARFORGE_RT/STARFORGE_v1.2"

# Same simulations as the IMF plot: the three M2e4_R1 metallicity runs
# (driving1, the only one with IMF.dat), and the M2e4_R10 Z=1 ensemble that
# imf_plots.py combines into output_all. The output_all dir has no
# global_statistics.fits, so we plot the eight underlying realizations.
runs = [
    # (path,                                                  M_cloud, R_cloud, logZ, label_or_None)
    (f"{BASE}/M2e4_R1/M2e4_R1_Z0.01_S0_A2_B0.1_I10000_Res271_n2_sol0.5_42/output_turbsphere_driving1",
     2e4, 1.0, -2, r"M2e4_R1, $Z=0.01\,Z_\odot$"),
    (f"{BASE}/M2e4_R1/M2e4_R1_Z0.1_S0_A2_B0.1_I10000_Res271_n2_sol0.5_42/output_turbsphere_driving1",
     2e4, 1.0, -1, r"M2e4_R1, $Z=0.1\,Z_\odot$"),
    (f"{BASE}/M2e4_R1/M2e4_R1_Z1_S0_A2_B0.1_I10000_Res271_n2_sol0.5_42/output_turbsphere_driving1",
     2e4, 1.0,  0, r"M2e4_R1, $Z=Z_\odot$"),
] + [
    (f"{BASE}/M2e4_R10/M2e4_R10_Z1_S0_A2_B0.1_I1_Res271_n2_sol0.5_42/output_turbsphere_driving{i}",
     2e4, 10.0, 0,
     r"M2e4_R10, $Z=Z_\odot$ (8 realizations)" if i == 1 else None)
    for i in range(1, 9)
]


def freefall_time(M_msun, R_pc):
    rho = 3.0 * M_msun / (4.0 * np.pi * R_pc**3)
    return np.sqrt(3.0 * np.pi / (32.0 * G_PC_MSUN_MYR * rho))


fig, ax = plt.subplots(figsize=(4, 4))

for path, M_cloud, R_cloud, logz, label in runs:
    stats_path = f"{path}/global_statistics.fits"
    if not os.path.isfile(stats_path):
        print(f"skip (no global_statistics.fits): {path}")
        continue

    stats = Table.read(stats_path)
    stats.sort("time")
    t = np.asarray(stats["time"])
    mstar = np.maximum.accumulate(np.asarray(stats["stellar_mass_sum"]))

    t_ff = freefall_time(M_cloud, R_cloud)
    x = (t - t[0]) / t_ff
    sfe = mstar / M_cloud

    color = "black" if "M2e4_R10" in path else LOGZ_COLORS[logz]
    ls = "dashdot" if "M2e4_R10" in path else "solid"
    alpha = 0.6 if "M2e4_R10" in path else 1.0

    ax.plot(x, sfe, color=color, ls=ls, lw=1.2, alpha=alpha, label=label)

ax.set(
    xlabel=r"$t / t_{\rm ff}$",
    ylabel=r"$\mathrm{SFE} = M_\star / (M_\star + M_{\rm gas})$",
    xscale="log",
    yscale="log",
    xlim=(1e-1, 5),
    ylim=(1e-4, 1),
)
leg = ax.legend(loc="lower right", fontsize=8, edgecolor="black", borderaxespad=0.4)
leg.get_frame().set_linewidth(0.8)
fig.tight_layout()

out_pdf = "sfe_vs_time.pdf"
fig.savefig(out_pdf, bbox_inches="tight")
plt.close(fig)
print(f"wrote {out_pdf}")
