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
from astropy.constants import G as G_const

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


def plot_trajectory(ax_mass, ax_mdot, t_kyr, mass, color, lw, label=None):
    ax_mass.plot(t_kyr - tmin, mass, lw=lw, color=color)
    ts = t_kyr[::GRADIENT_STRIDE]
    ms = mass[::GRADIENT_STRIDE]
    if len(ts) < 2:
        return
    mdot = np.diff(ms) / np.diff(ts) / 1e3
    ax_mdot.plot(ts[1:] - tmin, mdot, lw=lw, color=color, label=label)
    ax_mdot.plot(ts[1:] - tmin, -mdot, lw=lw, color=color, ls="dashed")


fig, ax = plt.subplots(2, 1, figsize=(4, 4))
for i, (color, lw, (_, (snaps, mass, _coord))) in enumerate(zip(colors, lws, top)):
    if i < 4:
        label = rf"Star {i+1} (${mass[-1]:.0f}\,M_\odot$)"
    elif i == 4:
        label = "Stars 5-10"
    else:
        label = None
    plot_trajectory(ax[0], ax[1], times_kyr[snaps], mass, color, lw, label=label)

# Total stellar accretion rate: dM_tot/dt where M_tot is the per-snapshot sum
# of every star's mass. Matches the per-star convention (solid positive,
# dashed negative).
mass_total = np.bincount(snap_idx, weights=masses, minlength=n_snaps)
mdot_total = np.diff(mass_total) / np.diff(times_kyr) / 1e3
ax[1].plot(times_kyr[1:] - tmin, mdot_total, color="black", lw=0.8, label="Total")
ax[1].plot(times_kyr[1:] - tmin, -mdot_total, color="black", lw=1.2, ls="dashed")

# Same sum, restricted to the four most-massive stars (ranked by max mass).
mass_top4 = np.zeros(n_snaps, dtype=np.float64)
for _, (snaps, mass, _coord) in ranked[:4]:
    mass_top4[snaps] += mass
mdot_top4 = np.diff(mass_top4) / np.diff(times_kyr) / 1e3
ax[1].plot(times_kyr[1:] - tmin, mdot_top4, color="orangered", lw=0.8, label="Top 4 Sum")

# Reference rate M_cloud / t_ff for the M=2e4 Msun, R=1 pc cloud.
_M_CLOUD = 2e4 * u.Msun
_R_CLOUD = 1.0 * u.pc
_rho_cloud = _M_CLOUD / ((4.0 / 3.0) * np.pi * _R_CLOUD**3)
_t_ff_cloud = np.sqrt(3.0 * np.pi / (32.0 * G_const * _rho_cloud)).to(u.yr)
_mdot_ff = float((_M_CLOUD / _t_ff_cloud).to(u.Msun / u.yr).value)
ax[1].axhline(_mdot_ff, color="gray", lw=0.8, ls="dashed",
              label=r"$M_{\rm cloud}/t_{\rm ff}$")

# Legend lists Total + Top 4 Sum first, then the four labeled per-star lines
# (added inside the loop via plot_trajectory's label kwarg).
handles, labels = ax[1].get_legend_handles_labels()
# Reorder so Total and Top 4 Sum come first.
priority = ["Total", "Top 4 Sum", r"$M_{\rm cloud}/t_{\rm ff}$"]
ordered = [(h, lab) for p in priority for h, lab in zip(handles, labels) if lab == p]
ordered += [(h, lab) for h, lab in zip(handles, labels) if lab not in priority]
ax[1].legend(
    [h for h, _ in ordered], [lab for _, lab in ordered],
    loc="lower right", fontsize=6, ncol=2,
    edgecolor="black", borderaxespad=0.4, frameon=True, labelspacing=0
).get_frame().set_linewidth(0.6)

ax[0].xaxis.set_ticklabels([])
ax[0].set(yscale="log", ylim=[0.1, 3e3], ylabel=r"Stellar Mass ($M_\odot$)", xlim=[0, 175])
ax[1].set(
    yscale="log",
    ylim=[3e-6, 0.3],
    ylabel=r"$\dot{M}\,\rm\left(M_\odot\,\rm yr^{-1}\right)$",
    xlim=[0, 175],
    xlabel="Time (kyr)",
)
ax[1].ticklabel_format(axis="x", style="plain", useOffset=False)
fig.subplots_adjust(hspace=0, wspace=0)
plt.savefig("VMS_Accretion_History.pdf", bbox_inches="tight")
plt.close()


# Distance-from-COM trajectories of the top-N stars, on their own figure.
fig, ax = plt.subplots(1, 1, figsize=(4, 4))
for i, (color, lw, (_, (snaps, mass, coord))) in enumerate(zip(colors, lws, top)):
    dist = np.linalg.norm(coord - com_pos[snaps], axis=1)
    finite = np.isfinite(dist)
    if i < 4:
        label = rf"Star {i+1} (${mass[-1]:.0f}\,M_\odot$)"
    elif i == 4:
        label = "Stars 5-10"
    else:
        label = None
    ax.plot((times_kyr[snaps] - tmin)[finite], dist[finite], lw=lw, color=color, label=label)
ax.set(
    yscale="log",
    xlabel="Time (kyr)",
    ylabel=r"Distance from COM (pc)",
    xlim=[0, 175],
    ylim=[1e-4, 1],
)
ax.ticklabel_format(axis="x", style="plain", useOffset=False)
ax.legend(
    loc="lower left", fontsize=6, ncol=2,
    edgecolor="black", borderaxespad=0.4, frameon=True, labelspacing=0,
).get_frame().set_linewidth(0.6)
plt.savefig("VMS_Distance_from_COM.pdf", bbox_inches="tight")
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
