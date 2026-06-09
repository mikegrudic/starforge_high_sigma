"""python mdot_vs_m_23.py [N_STARS]

Plot Mdot / M^(2/3) vs M for the N most massive stars in the M2e4_R1_Z0.01 run.
Shares the colormap with accretion_histories.py via vms_colors.
"""

import sys
import numpy as np
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from astropy import units as u

from vms_colors import colors_and_lws

CODE_TO_KYR = float((u.pc / (u.m / u.s)).to(u.kyr))
RUN_PATH = (
    "imf_data/STARFORGE_RT/STARFORGE_v1.2/M2e4_R1/"
    "M2e4_R1_Z0.01_S0_A2_B0.1_I10000_Res271_n2_sol0.5_42/output_turbsphere_driving1"
)
GRADIENT_STRIDE = 1
DEFAULT_N_STARS = 10

n_stars = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_N_STARS

data = np.load(RUN_PATH + "/star_masses.npz")
times_kyr = data["times"] * CODE_TO_KYR
snap_idx = data["snap_idx"]
ids = data["ids"]
masses = data["masses"]

order = np.argsort(ids, kind="stable")
sorted_ids = ids[order]
sorted_snap = snap_idx[order]
sorted_mass = masses[order]
unique_ids, starts = np.unique(sorted_ids, return_index=True)
splits = np.split(np.arange(len(sorted_ids)), starts[1:])
trajectories = {
    int(uid): (sorted_snap[idx], sorted_mass[idx])
    for uid, idx in zip(unique_ids, splits)
}

ranked = sorted(trajectories.items(), key=lambda kv: kv[1][1].max(), reverse=True)
top = ranked[:n_stars]
colors, lws = colors_and_lws(n_stars)

fig, ax = plt.subplots(1, 1, figsize=(4, 4))
for color, lw, (_, (snaps, mass)) in zip(colors, lws, top):
    t_kyr = times_kyr[snaps]
    ts = t_kyr[::GRADIENT_STRIDE]
    ms = mass[::GRADIENT_STRIDE]
    if len(ts) < 2:
        continue
    mdot = np.diff(ms) / np.diff(ts) / 1e3
    m_for_mdot = ms[1:]
    positive = m_for_mdot > 0
    ax.plot(
        m_for_mdot[positive],
        mdot[positive] / m_for_mdot[positive] ** (2.0 / 3.0),
        lw=lw,
        color=color,
    )

ax.set(
    xscale="log",
    yscale="log",
    xlabel=r"Stellar Masses ($M_\odot$)",
    ylabel=r"$\dot{M} /M^{2/3}\,\rm\left(M_\odot^{1/3}\,\rm yr^{-1}\right)$",
    ylim=[1e-5,3e-3]
)
fig.subplots_adjust(hspace=0, wspace=0)
plt.savefig("VMS_M_vs_Mdot_M23.pdf", bbox_inches="tight")
