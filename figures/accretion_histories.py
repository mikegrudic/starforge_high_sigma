"""python accretion_histories.py [N_STARS]

Plots the accretion history of the N most massive stars (default 4) in the
M2e4_R1_Z0.01 run, using the per-snapshot per-star (ID, BH_Mass) records in
imf_data/.../star_masses.npz produced by extract_star_masses.py.
"""

import sys
import numpy as np
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from astropy import units as u

from vms_colors import N_COM_STARS, colors_and_lws

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
coords = data["coords"]

# Group records by particle ID. Stable sort preserves the time order (snap_idx
# is monotone in the unsorted records, since extraction wrote snapshot-by-snapshot).
order = np.argsort(ids, kind="stable")
sorted_ids = ids[order]
sorted_snap = snap_idx[order]
sorted_mass = masses[order]
sorted_coords = coords[order]
unique_ids, starts = np.unique(sorted_ids, return_index=True)
splits = np.split(np.arange(len(sorted_ids)), starts[1:])
trajectories = {
    int(uid): (sorted_snap[idx], sorted_mass[idx], sorted_coords[idx])
    for uid, idx in zip(unique_ids, splits)
}

ranked = sorted(trajectories.items(), key=lambda kv: kv[1][1].max(), reverse=True)
top = ranked[:n_stars]
com_set = ranked[:N_COM_STARS]
tmin = times_kyr[0]
colors, lws = colors_and_lws(n_stars)

# Mass-weighted COM of the N_COM_STARS most massive, per snapshot. NaN at
# snapshots where none of those stars exist yet.
n_snaps = len(times_kyr)
com_m = np.full((n_snaps, N_COM_STARS), np.nan, dtype=np.float32)
com_r = np.full((n_snaps, N_COM_STARS, 3), np.nan, dtype=np.float32)
for k, (_, (s_arr, m_arr, c_arr)) in enumerate(com_set):
    com_m[s_arr, k] = m_arr
    com_r[s_arr, k, :] = c_arr
mask = ~np.isnan(com_m)
m_safe = np.where(mask, com_m, 0.0)
r_safe = np.where(mask[..., None], com_r, 0.0)
total_m = m_safe.sum(axis=1)
with np.errstate(invalid="ignore", divide="ignore"):
    com_pos = (r_safe * m_safe[..., None]).sum(axis=1) / total_m[..., None]
com_pos[total_m == 0] = np.nan


def plot_trajectory(ax_mass, ax_mdot, ax_dist, t_kyr, mass, coord, color, lw):
    ax_mass.plot(t_kyr - tmin, mass, lw=lw, color=color)
    ts = t_kyr[::GRADIENT_STRIDE]
    ms = mass[::GRADIENT_STRIDE]
    if len(ts) < 2:
        return
    mdot = np.diff(ms) / np.diff(ts) / 1e3
    ax_mdot.plot(ts[1:] - tmin, mdot, lw=lw, color=color)
    ax_mdot.plot(ts[1:] - tmin, -mdot, lw=lw, color=color, ls="dashed")


fig, ax = plt.subplots(3, 1, figsize=(4, 6))
for color, lw, (_, (snaps, mass, coord)) in zip(colors, lws, top):
    plot_trajectory(ax[0], ax[1], ax[2], times_kyr[snaps], mass, coord, color, lw)
    dist = np.linalg.norm(coord - com_pos[snaps], axis=1)
    finite = np.isfinite(dist)
    ax[2].plot((times_kyr[snaps] - tmin)[finite], dist[finite], lw=lw, color=color)

ax[0].xaxis.set_ticklabels([])
ax[1].xaxis.set_ticklabels([])
ax[0].set(yscale="log", ylim=[0.1, 3e3], ylabel=r"Stellar Mass ($M_\odot$)", xlim=[0, 175])
ax[1].set(
    yscale="log",
    ylim=[3e-6, 0.1],
    ylabel=r"$\dot{M}\,\rm\left(M_\odot\,\rm yr^{-1}\right)$",
    xlim=[0, 175],
)
ax[2].set(
    yscale="log",
    xlabel="Time (kyr)",
    ylabel=r"Distance from COM (pc)",
    xlim=[0, 175],
    ylim=[1e-4,1]
)
fig.subplots_adjust(hspace=0, wspace=0)
plt.savefig("VMS_Accretion_History.pdf", bbox_inches="tight")
plt.close()


fig, ax = plt.subplots(1, 1, figsize=(4, 4))
for color, lw, (_, (snaps, mass, _coord)) in zip(colors, lws, top):
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
        (mdot[positive] / m_for_mdot[positive] ** 2.0),
        lw=lw,
        color=color,
    )

ax.set(
    xscale="log",
    yscale="log",
    xlabel=r"Stellar Mass ($M_\odot$)",
    ylabel=r"$\dot{M} /M^{2}\,\rm\left(M_\odot^{-1}\,\rm yr^{-1}\right)$",
)
fig.subplots_adjust(hspace=0, wspace=0)
plt.savefig("VMS_M_vs_Mdot.pdf", bbox_inches="tight")
