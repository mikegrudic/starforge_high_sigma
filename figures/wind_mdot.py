"""Wind mass-loss rate vs. stellar mass for three metallicities.

Implements the SINGLE_STAR_FB_WINDS=2 prescription from
galaxy_sf/stellar_evolution.c :: single_star_wind_mdot():

  dot_M_MS   -- de Jager/3 (Smith 2014): base OB-wind rate
  dot_M_weak -- weak-wind limiter (suppresses at low L)
  dot_M_S22  -- near-Eddington floor (Sahahit 2022 Eq. 13)
  dot_M_wind -- max(min(dot_M_MS, dot_M_weak), dot_M_S22)

L_MS and T_eff from Tout 1996 via starforge_tools.
Color-coding from logz_to_color() in sim_paths (single source of truth).
"""

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np

from sim_paths import logz_to_color
from starforge_tools.star_properties import luminosity_MS, radius_MS, effective_temperature

MASS = np.logspace(1, np.log10(2e3), 500)  # 10–2000 Msun


def _vinf_over_vesc(T_eff):
    """Terminal-to-escape velocity ratio from Lamers 1995."""
    return np.where(T_eff < 1.25e4, 0.7, np.where(T_eff < 2.1e4, 1.3, 2.6))


def wind_mdot(mass, Z_solar):
    """Wind mass-loss rate in Msun/yr for SINGLE_STAR_FB_WINDS=2.

    Parameters
    ----------
    mass : array_like, Msun
    Z_solar : float, metallicity relative to solar
    """
    L = luminosity_MS(mass)
    R = radius_MS(mass)
    T_eff = effective_temperature(lum=L, radius=R)
    vrat = _vinf_over_vesc(T_eff)

    log_mdot_MS   = -6.000 + 1.500 * np.log10(L / 1e6) + 0.690 * np.log10(Z_solar)
    log_mdot_weak = -7.650 + 2.900 * np.log10(L / 1e5)
    log_mdot_S22  = (-8.445
                     + 4.770 * np.log10(L / 1e5)
                     - 3.990 * np.log10(mass / 30)
                     - 1.226 * np.log10(vrat / 2)
                     + 0.761 * np.log10(Z_solar))

    return 10 ** np.maximum(np.minimum(log_mdot_MS, log_mdot_weak), log_mdot_S22)


fig, ax = plt.subplots(figsize=(3.5, 3.0))

for logZ in [-2, -1, 0]:
    Z = 10**logZ
    ax.loglog(
        MASS,
        wind_mdot(MASS, Z),
        color=logz_to_color(logZ),
        lw=1.5,
        label=rf"${Z:g}\,Z_\odot$",
    )

ax.set_xlabel(r"$M\;(M_\odot)$")
ax.set_ylabel(r"$\dot{M}_{\rm wind}\;(M_\odot\,\mathrm{yr}^{-1})$")
ax.set_xlim(10, 2e3)
ax.set_ylim(1e-10, 1e-2)
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig("wind_mdot.pdf", bbox_inches="tight")
