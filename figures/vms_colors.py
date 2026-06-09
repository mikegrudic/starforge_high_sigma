"""Shared color scheme for VMS accretion-history plots.

The top N_COM_STARS get viridis colors spanning [VIRIDIS_LO, VIRIDIS_HI]; the
rest are black at a narrower line width. Used by accretion_histories.py and
mdot_vs_m_23.py so the two plots stay visually consistent.
"""

import numpy as np
from matplotlib import pyplot as plt

N_COM_STARS = 4
VIRIDIS_LO = 0.15
VIRIDIS_HI = 0.85
LW_COLOR = 1.0
LW_BLACK = 0.4


def colors_and_lws(n_stars):
    n_color = min(N_COM_STARS, n_stars)
    colors = 0.5*np.ones((n_stars, 4))
    colors[:, 3] = 1.0
    colors[:n_color] = plt.cm.viridis(np.linspace(VIRIDIS_LO, VIRIDIS_HI, n_color))
    lws = np.where(np.arange(n_stars) < n_color, LW_COLOR, LW_BLACK)
    return colors, lws
