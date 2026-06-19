"""Final SFE vs initial surface density Sigma = M / (pi R^2).

Same simulations as the Mcluster-vs-Mmax plot. Color coding matches
mclusters_vs_mmax.py (logZ -> Dark2_3 via LOGZ_COLORMAP), and the physics
experiments share that color but use distinct markers. Overlays the
Grudic+2021 single-cluster SFE fit:

    eps_int(Sigma) = eps_max / (1 + Sigma_crit / Sigma)

The two constants are isolated at the top so the exact paper values can be
swapped in without editing the rest of the script.
"""

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

import numpy as np
from astropy.table import Table

from sim_paths import (
    FIDUCIAL_RUNS, PHYSICS_EXPERIMENTS, LOGZ_COLORMAP, logz_to_color,
)


# Grudic+2021 fit. Confirm against the paper values.
EPS_MAX = 0.7
SIGMA_CRIT = 3200  # Msun / pc^2


def grudic21_sfe(sigma):
    return EPS_MAX / (1.0 + SIGMA_CRIT / sigma)


def final_sfe(run):
    """Final SFE = max(stellar_mass_sum) / M_cloud_initial."""
    stats = Table.read(run.path + "/global_statistics.fits")
    mstar_final = float(np.max(stats["stellar_mass_sum"]))
    return mstar_final / run.M_cloud


fig, ax = plt.subplots(figsize=(4, 4))

# Horizontal jitter in log-space so points sharing the same nominal Sigma
# (e.g. the eight M2e4_R10 driving realizations) don't fully overlap.
# Deterministic via fixed seed so the figure is reproducible.
_jitter_rng = np.random.default_rng(0)
JITTER_DEX = 0.05


def jitter(sigma):
    return sigma * 10.0 ** _jitter_rng.uniform(-JITTER_DEX, JITTER_DEX)


# Fiducial paramspace runs: circle markers, color by logZ.
for run in FIDUCIAL_RUNS:
    ax.scatter(jitter(run.sigma), final_sfe(run),
               color=logz_to_color(run.logZ), s=30, marker=run.marker,
               edgecolor="black", lw=0.3, zorder=10)

# Physics experiments: same color (Z=1), distinct markers (matches Mmax plot).
for run in PHYSICS_EXPERIMENTS:
    ax.scatter(jitter(run.sigma), final_sfe(run),
               color=logz_to_color(run.logZ), s=30, marker=run.marker,
               edgecolor="black", lw=0.3, zorder=10)

# Unweighted log-space power-law fit to Z=Zsun fiducial (standard physics) points.
_z1_runs = [r for r in FIDUCIAL_RUNS if r.logZ == 0]
_z1_sigma = np.array([r.sigma for r in _z1_runs])
_z1_sfe = np.array([final_sfe(r) for r in _z1_runs])
_alpha, _logb = np.polyfit(np.log10(_z1_sigma), np.log10(_z1_sfe), 1)
_fit_sigma = np.array([_z1_sigma.min(), _z1_sigma.max()])
ax.plot(_fit_sigma, 10.0 ** (_alpha * np.log10(_fit_sigma) + _logb),
        color="black", lw=1.2, ls="dotted", zorder=5)
ax.annotate(
    rf"$\propto \Sigma^{{{_alpha:.2f}}}$",
    xy=(1.6e3, 0.1), fontsize=8,
)

# Grudic+2021 fit curve + 1-sigma scatter band (0.13 dex about the median).
sigma_grid = np.logspace(0, 5, 200)
fit_median = grudic21_sfe(sigma_grid)
FIT_SCATTER_DEX = 0.13
ax.fill_between(
    sigma_grid,
    fit_median * 10.0 ** -FIT_SCATTER_DEX,
    fit_median * 10.0 ** +FIT_SCATTER_DEX,
    color="black", alpha=0.15, lw=0,
)
ax.plot(sigma_grid, fit_median, color="black", lw=1.0,
        label=r"Grudi\'{c}+2021 simulations fit", ls="dashed")

# Marker legend proxies, matching the Mmax plot.
ax.scatter([], [], marker="o", color="black", s=30, edgecolor="black",
           lw=0.3, label="Fiducial Model")
for run in PHYSICS_EXPERIMENTS:
    ax.scatter([], [], marker=run.marker, color="black", s=30,
               edgecolor="black", lw=0.3, label=run.label)

ax.set(
    xscale="log",
    yscale="log",
    xlabel=r"$\Sigma = M/(\pi R^2)\,(M_\odot\,\mathrm{pc}^{-2})$",
    ylabel=r"Final SFE",
    xlim=[10, 3e4],
    ylim=[3e-3, 1],
)
ax.legend(loc="lower right", fontsize=7, edgecolor="black",
          labelspacing=0.2).get_frame().set_linewidth(0.6)

# logZ colorbar in the upper-left corner, same idiom as mclusters_vs_mmax.py.
a = np.array([[-2.5, 0.5], [-2, 0.5]])
img = ax.pcolormesh(a, cmap=LOGZ_COLORMAP)
img.set_visible(False)
c = plt.colorbar(
    img,
    cax=fig.add_axes([0.20, 0.83, 0.3, 0.04]),
    label=r"$\log Z/Z_\odot$",
    orientation="horizontal",
    ticks=[-2, -1, 0],
)
c.minorticks_off()

plt.savefig("sfe_vs_sigma.pdf", bbox_inches="tight")
print("wrote sfe_vs_sigma.pdf")
