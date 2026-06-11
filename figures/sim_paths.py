"""Single source of truth for the STARFORGE runs plotted in the figures.

Each script in this directory that picks a set of simulations to plot (IMF
comparison, SFE-vs-time, SFE-vs-Sigma, Mcluster-vs-Mmax) imports the list it
needs from here rather than maintaining its own dict.

If you want to add or remove a sim, edit it here and only here.
"""

import math
import os
from dataclasses import dataclass

import numpy as np
from matplotlib.colors import ListedColormap
from palettable.colorbrewer.qualitative import Dark2_3


BASE = "imf_data/STARFORGE_RT/STARFORGE_v1.2"

# Shared logZ -> color map (3 quantized Dark2 colors for logZ in {-2, -1, 0}).
LOGZ_COLORMAP = ListedColormap(Dark2_3.mpl_colors)


def logz_to_color(logz):
    return LOGZ_COLORMAP((logz - (-2)) / (np.log10(1) - (-2)))


@dataclass(frozen=True)
class SimRun:
    label: str
    path: str
    M_cloud: float       # initial cloud mass, Msun
    R_cloud: float       # initial cloud radius, pc
    logZ: float          # log10(Z / Z_sun)
    marker: str = "o"
    is_physics: bool = False
    # For combined output_all entries: how many driving realizations were
    # pooled. 0 means "not a combined run". Used by imf_plots._short_label.
    n_realizations: int = 0

    @property
    def sigma(self):
        return self.M_cloud / (np.pi * self.R_cloud**2)

    @property
    def short_label(self):
        """Legend label: ``Sigma X Msun/pc^2, Y Z_sun`` (+ realization suffix).

        Used by the IMF comparison + alpha plots so labels stay in sync across
        them. Edit here, not in the consumer scripts.
        """
        sigma = self.sigma
        exp = math.floor(math.log10(abs(sigma)))
        sigma_2sf = round(sigma, -int(exp) + 1)
        sigma_str = f"{sigma_2sf:.0f}"
        z_val = f"{10.0 ** self.logZ:g}"

        out_dir = os.path.basename(self.path)
        if self.n_realizations > 0 or out_dir == "output_turbsphere_driving1":
            suffix = ""
        else:
            suffix = f"  ({out_dir.replace('output_', '')})"

        return (
            rf"${sigma_str}\,M_\odot\,\mathrm{{pc}}^{{-2}}$, "
            rf"${z_val}\,Z_\odot${suffix}"
        )


_M2E4_R1 = "M2e4_R1/M2e4_R1_{Z}_S0_A2_B0.1_I10000_Res271_n2_sol0.5_42"
_M2E4_R10 = "M2e4_R10/M2e4_R10_{Z}_S0_A2_B0.1_I1_Res271_n2_sol0.5_42"
_DRIVING1 = "output_turbsphere_driving1"


# M2e4_R1 cloud at three metallicities — the standard driving1 realization.
M2E4_R1_FIDUCIAL = [
    SimRun(r"M2e4_R1, $Z=0.01\,Z_\odot$",
           f"{BASE}/{_M2E4_R1.format(Z='Z0.01')}/{_DRIVING1}",
           2e4, 1.0, -2),
    SimRun(r"M2e4_R1, $Z=0.1\,Z_\odot$",
           f"{BASE}/{_M2E4_R1.format(Z='Z0.1')}/{_DRIVING1}",
           2e4, 1.0, -1),
    SimRun(r"M2e4_R1, $Z=Z_\odot$",
           f"{BASE}/{_M2E4_R1.format(Z='Z1')}/{_DRIVING1}",
           2e4, 1.0, 0),
]

# M2e4_R10 cloud at solar Z (8 driving realizations) and Z=0.01 Z_sun
# (3 driving realizations).
M2E4_R10_FIDUCIAL = [
    SimRun(rf"M2e4_R10, $Z=Z_\odot$ (driving {i})",
           f"{BASE}/{_M2E4_R10.format(Z='Z1')}/output_turbsphere_driving{i}",
           2e4, 10.0, 0)
    for i in range(1, 9)
] + [
    SimRun(rf"M2e4_R10, $Z=0.01\,Z_\odot$ (driving {i})",
           f"{BASE}/{_M2E4_R10.format(Z='Z0.01')}/output_turbsphere_driving{i}",
           2e4, 10.0, -2)
    for i in range(1, 4)
]

# Physics experiments: M2e4_R1 Z=1 cloud rerun with feedback channels toggled.
PHYSICS_EXPERIMENTS = [
    SimRun(r"No Winds",
           f"{BASE}/{_M2E4_R1.format(Z='Z1')}/output_nowind",
           2e4, 1.0, 0, marker="s", is_physics=True),
    SimRun(r"No IR Rad. Pressure",
           f"{BASE}/{_M2E4_R1.format(Z='Z1')}/output_noIR",
           2e4, 1.0, 0, marker="D", is_physics=True),
    SimRun(r"$3\times$ Stronger Winds",
           f"{BASE}/{_M2E4_R1.format(Z='Z1')}/output_vink",
           2e4, 1.0, 0, marker="^", is_physics=True),
]

# IMF plots show each M2e4_R10 metallicity as a single output combining the
# driving realizations. Those directories hold IMF.dat (and posterior samples)
# but no global_statistics.fits — so they're only useful for IMF figures.
M2E4_R10_IMF_COMBINED = [
    SimRun(r"M2e4_R10, $Z=Z_\odot$",
           f"{BASE}/{_M2E4_R10.format(Z='Z1')}/output_all",
           2e4, 10.0, 0, n_realizations=8),
    SimRun(r"M2e4_R10, $Z=0.01\,Z_\odot$",
           f"{BASE}/{_M2E4_R10.format(Z='Z0.01')}/output_all",
           2e4, 10.0, -2, n_realizations=3),
]


# Bundles for consumers. Add to FIDUCIAL_RUNS to propagate a new sim to the
# Mmax / SFE-vs-time / SFE-vs-Sigma plots simultaneously.
FIDUCIAL_RUNS = M2E4_R1_FIDUCIAL + M2E4_R10_FIDUCIAL
FIDUCIAL_AND_PHYSICS = FIDUCIAL_RUNS + PHYSICS_EXPERIMENTS
IMF_RUNS = M2E4_R10_IMF_COMBINED + M2E4_R1_FIDUCIAL

# IMF analysis bundle: every run we want posterior samples for. This is a
# superset of IMF_RUNS (which is what gets *plotted*) because we also want
# samples for the individual M2e4_R10 driving realizations, the physics
# experiments, etc. — even if they're not currently in any IMF figure.
ANALYSIS_RUNS = FIDUCIAL_RUNS + PHYSICS_EXPERIMENTS + M2E4_R10_IMF_COMBINED
