"""Plots total cluster mass vs. maximum stellar mass, producing an output Mcluster_vs_Mmax_paramspace.pdf and Mcluster_vs_Mmax_physics.pdf"""

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


runs_physics = {
    r"$Z_\odot$": "imf_data/STARFORGE_RT/STARFORGE_v1.2/M2e4_R1/M2e4_R1_Z1_S0_A2_B0.1_I10000_Res271_n2_sol0.5_42/output_turbsphere_driving1",
    r"$Z_\odot$ (No Winds)": "imf_data/STARFORGE_RT/STARFORGE_v1.2/M2e4_R1/M2e4_R1_Z1_S0_A2_B0.1_I10000_Res271_n2_sol0.5_42/output_nowind",
#    r"$Z_\odot$ (No Winds, no IR RP)": "imf_data/STARFORGE_RT/STARFORGE_v1.2/M2e4_R1/M2e4_R1_Z1_S0_A2_B0.1_I10000_Res271_n2_sol0.5_42/output_nowind_noIR",
    r"$Z_\odot$ ($3\times$ Stronger Winds)": "imf_data/STARFORGE_RT/STARFORGE_v1.2/M2e4_R1/M2e4_R1_Z1_S0_A2_B0.1_I10000_Res271_n2_sol0.5_42/output_vink",
    r"$Z_\odot$ (No IR Rad. Pressure)": "imf_data/STARFORGE_RT/STARFORGE_v1.2/M2e4_R1/M2e4_R1_Z1_S0_A2_B0.1_I10000_Res271_n2_sol0.5_42/output_noIR",
    r"$10\% Z_\odot$": "imf_data/STARFORGE_RT/STARFORGE_v1.2/M2e4_R1/M2e4_R1_Z0.1_S0_A2_B0.1_I10000_Res271_n2_sol0.5_42/output_turbsphere_driving1",
    r"$1\% Z_\odot$": "imf_data/STARFORGE_RT/STARFORGE_v1.2/M2e4_R1/M2e4_R1_Z0.01_S0_A2_B0.1_I10000_Res271_n2_sol0.5_42/output_turbsphere_driving1",
}


for maxnum in range(0, 6):
    i = 0
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.set_prop_cycle("color", palettable.colorbrewer.qualitative.Dark2_8.mpl_colors)

    for run, path in runs_physics.items():
        if i > maxnum:
            break

        logz = 0
        if "Z0.1" in path:
            logz = -1
        elif "Z0.01" in path:
            logz = -2

        label = run
        if "M2e4_R10" in run:
            if run == "M2e4_R10_Z1_1":
                label = "M2e4_R10_Z1"
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

        ax.loglog(
            stats["stellar_mass_sum"],
            stats["stellar_mass_max"],
            label=label,
            ls=ls,
            # color=color,
            lw=1,
        )
        i += 1

    ax.legend(labelspacing=0, loc=2, frameon=False, fontsize=10)
    ax.set(
        ylabel=r"$M_{\rm max}$ ($M_\odot$)",  # $M_{\rm \star,max}\,\left(M_\odot\right)$",
        xlabel=r"$M_{\rm cluster}$ ($M_\odot$)",
        xlim=[1e2, 1e4],
        ylim=[1e1, 2000],
        xscale="log",
        yscale="log",
    )

    plt.savefig(f"Mcluster_vs_Mmax_{maxnum}.png", bbox_inches="tight", dpi=400)
    plt.clf()
    plt.close()
