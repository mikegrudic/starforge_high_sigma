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

runs_physics = {
    #   r"$Z_\odot$ (MW GMC)": "imf_data/STARFORGE_RT/STARFORGE_v1.2/M2e5_R3/M2e5_R30_Z1_S0_A2_B0.1_I1_Res271_n2_sol0.5_42/output",
    # r"$Z_\odot$ Fiducial": "M2e4_R1_Z1_S0_A2_B0.1_I10000_Res271_n2_sol0.5_42/output",
    r"$Z_\odot$": "imf_data/STARFORGE_RT/STARFORGE_v1.2/M2e4_R1/M2e4_R1_Z1_S0_A2_B0.1_I10000_Res271_n2_sol0.5_42/output_turbsphere_driving1",
    r"$10\% Z_\odot$": "imf_data/STARFORGE_RT/STARFORGE_v1.2/M2e4_R1/M2e4_R1_Z0.1_S0_A2_B0.1_I10000_Res271_n2_sol0.5_42/output_turbsphere_driving1",
    r"$1\% Z_\odot$": "imf_data/STARFORGE_RT/STARFORGE_v1.2/M2e4_R1/M2e4_R1_Z0.01_S0_A2_B0.1_I10000_Res271_n2_sol0.5_42/output_turbsphere_driving1",
    # r"M2e4_R1_Z0.1": "imf_data/STARFORGE_RT/STARFORGE_v1.2/M2e4_R1/M2e4_R1_Z0.1_S0_A2_B0.1_I10000_Res271_n2_sol0.5_42/output_turbsphere_driving1",
    # r"M2e4_R1_Z0.01": "imf_data/STARFORGE_RT/STARFORGE_v1.2/M2e4_R1/M2e4_R1_Z0.01_S0_A2_B0.1_I10000_Res271_n2_sol0.5_42/output_turbsphere_driving1",
    # r"M2e5_R3_Z1": "imf_data/STARFORGE_RT/STARFORGE_v1.2/M2e5_R3/M2e5_R3_Z1_S0_A2_B0.1_I10000_Res271_n2_sol0.5_42/output_turbsphere_driving1",
    # r"M2e5_R3_Z0.1": "imf_data/STARFORGE_RT/STARFORGE_v1.2/M2e5_R3/M2e5_R3_Z0.1_S0_A2_B0.1_I10000_Res271_n2_sol0.5_42/output_turbsphere_driving1",
    # r"M2e5_R3_Z0.01": "imf_data/STARFORGE_RT/STARFORGE_v1.2/M2e5_R3/M2e5_R3_Z0.01_S0_A2_B0.1_I10000_Res271_n2_sol0.5_42/output_turbsphere_driving1",
    # r"M2e4_R10_Z1_1": "imf_data/STARFORGE_RT/STARFORGE_v1.2/M2e4_R10/M2e4_R10_Z1_S0_A2_B0.1_I1_Res271_n2_sol0.5_42/output_turbsphere_driving1",
    # r"M2e4_R10_Z1_2": "imf_data/STARFORGE_RT/STARFORGE_v1.2/M2e4_R10/M2e4_R10_Z1_S0_A2_B0.1_I1_Res271_n2_sol0.5_42/output_turbsphere_driving2",
    # r"M2e4_R10_Z1_3": "imf_data/STARFORGE_RT/STARFORGE_v1.2/M2e4_R10/M2e4_R10_Z1_S0_A2_B0.1_I1_Res271_n2_sol0.5_42/output_turbsphere_driving3",
    # r"M2e4_R10_Z1_4": "imf_data/STARFORGE_RT/STARFORGE_v1.2/M2e4_R10/M2e4_R10_Z1_S0_A2_B0.1_I1_Res271_n2_sol0.5_42/output_turbsphere_driving4",
    # r"M2e4_R10_Z1_5": "imf_data/STARFORGE_RT/STARFORGE_v1.2/M2e4_R10/M2e4_R10_Z1_S0_A2_B0.1_I1_Res271_n2_sol0.5_42/output_turbsphere_driving5",
    # r"M2e4_R10_Z1_6": "imf_data/STARFORGE_RT/STARFORGE_v1.2/M2e4_R10/M2e4_R10_Z1_S0_A2_B0.1_I1_Res271_n2_sol0.5_42/output_turbsphere_driving6",
    # r"M2e4_R10_Z1_7": "imf_data/STARFORGE_RT/STARFORGE_v1.2/M2e4_R10/M2e4_R10_Z1_S0_A2_B0.1_I1_Res271_n2_sol0.5_42/output_turbsphere_driving7",
    # r"M2e4_R10_Z1_8": "imf_data/STARFORGE_RT/STARFORGE_v1.2/M2e4_R10/M2e4_R10_Z1_S0_A2_B0.1_I1_Res271_n2_sol0.5_42/output_turbsphere_driving8",
    # r"M2e4_R10_Z0.01_1": "imf_data/STARFORGE_RT/STARFORGE_v1.2/M2e4_R10/M2e4_R10_Z0.01_S0_A2_B0.1_I1_Res271_n2_sol0.5_42/output_turbsphere_driving1",
    # r"M2e4_R10_Z0.01_2": "imf_data/STARFORGE_RT/STARFORGE_v1.2/M2e4_R10/M2e4_R10_Z0.01_S0_A2_B0.1_I1_Res271_n2_sol0.5_42/output_turbsphere_driving2",
    # r"M2e4_R10_Z0.01_3": "imf_data/STARFORGE_RT/STARFORGE_v1.2/M2e4_R10/M2e4_R10_Z0.01_S0_A2_B0.1_I1_Res271_n2_sol0.5_42/output_turbsphere_driving3",
    # r"$Z_\odot$ M2e5_R3": "M2e5_R3_Z1_S0_A2_B0.1_I10000_Res271_n2_sol0.5_42/output",
    # r"$1\% Z_\odot$ M2e5_R3": "M2e5_R3_Z0.01_S0_A2_B0.1_I10000_Res271_n2_sol0.5_42/output",
    r"$Z_\odot$ No Winds": "imf_data/STARFORGE_RT/STARFORGE_v1.2/M2e4_R1/M2e4_R1_Z1_S0_A2_B0.1_I10000_Res271_n2_sol0.5_42/output_nowind",
    r"$Z_\odot$ No IR Rad. Pressure": "imf_data/STARFORGE_RT/STARFORGE_v1.2/M2e4_R1/M2e4_R1_Z1_S0_A2_B0.1_I10000_Res271_n2_sol0.5_42/output_noIR",
    r"$Z_\odot$ $3\times$ Stronger Winds": "imf_data/STARFORGE_RT/STARFORGE_v1.2/M2e4_R1/M2e4_R1_Z1_S0_A2_B0.1_I10000_Res271_n2_sol0.5_42/output_vink",
    #    r"$1\% Z_\odot$": "M2e4_R1_Z0.01_S0_A2_B0.1_I10000_Res271_n2_sol0.5_42/output",
    #    r"$Z_\odot$ (Milky Way GMC, $2e5M_\odot$, No Rad. Pressure)": "/home/mgrudic/code/hiz/STARFORGE_v1.1/M2e5_R3/M2e5_R30/M2e5_R3/M2e5_R30_S0_T1_B0.01_Res271_n2_sol0.5_42/output/",
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
    if "turbsphere" in path:
        marker = "D"
    else:
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

ax.scatter([], [], marker="D", color="black", label="TURBSPHERE")
ax.scatter([], [], marker="o", color="black", label="Sphere")

#ax.legend(labelspacing=0, loc=4, frameon=True, fontsize=10, edgecolor="black")
ax.set(
    ylabel=r"$M_{\rm max}\,\left(M_\odot\right)$",
    xlabel=r"$M_{\rm cluster}\,\left(M_\odot\right)$",
    xlim=[100, 1e4],
    ylim=[10, 2400],
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
plt.savefig("Mcluster_vs_Mmax.pdf", bbox_inches="tight")
