"""n_H vs T phase diagram for the first available snapshot of each fiducial run.

For each entry in sim_paths.FIDUCIAL_RUNS, find the lowest-numbered
``snapshot_*.hdf5`` in its output dir, compute n_H and T particle-wise (same
recipe as gizmo/test/gmc_cooling/test_gmc_cooling.py), and accumulate into a
shared (log n_H, log T) 2D histogram per (cloud, logZ) group. Pooling
realizations by summing histograms keeps memory bounded across the 8 M2e4_R10
Z=Z_sun and 3 M2e4_R10 Z=0.01 realizations.

Curves are the per-n_H-column 16/50/84 percentiles recovered from the CDF.
Color by logZ (sim_paths.logz_to_color), linestyle by cloud size (M2e4_R1
solid, M2e4_R10 dotted), matching the convention in sfe_vs_time.py.
"""

import glob
import hashlib
import os

import h5py
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
from astropy import constants as c, units as u

from sim_paths import FIDUCIAL_RUNS, logz_to_color


RHO_TO_NH_CODE = (u.Msun / u.pc**3).to(c.m_p / u.cm**3)

# sim_paths.SimRun.path points into imf_data/, which only mirrors IMF.dat etc.
# The actual snapshot files live on ceph.
CEPH_ROOT = "/mnt/ceph/users/starforge"
IMF_ROOT = "imf_data"


def ceph_path(p):
    rel = os.path.relpath(p, IMF_ROOT)
    return os.path.join(CEPH_ROOT, rel)

LOG_NH_BINS = np.linspace(-1, 10, 56)
LOG_T_BINS = np.linspace(-1, 5, 251)
NH_CENTERS = 10.0 ** (0.5 * (LOG_NH_BINS[1:] + LOG_NH_BINS[:-1]))
T_CENTERS = 10.0 ** (0.5 * (LOG_T_BINS[1:] + LOG_T_BINS[:-1]))


def _snap_num(path):
    return int(os.path.basename(path).removeprefix("snapshot_").split(".")[0])


def first_snapshot(output_dir):
    files = glob.glob(os.path.join(output_dir, "snapshot_*.hdf5"))
    if not files:
        return None
    return min(files, key=_snap_num)


CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         ".nH_vs_T_cache")


def _cache_path(snap_path):
    h = hashlib.md5(os.path.abspath(snap_path).encode()).hexdigest()[:16]
    return os.path.join(CACHE_DIR, f"{h}.npz")


def _build_snap_hist(snap_path):
    with h5py.File(snap_path, "r") as F:
        Z = F["PartType0/Metallicity"][:, :2]
        XH = 1.0 - Z[:, 0] - Z[:, 1]
        nH = F["PartType0/Density"][:] * XH * RHO_TO_NH_CODE
        T = F["PartType0/Temperature"][:]
    mask = (nH > 0) & (T > 0)
    log_nH = np.log10(nH[mask])
    log_T = np.log10(T[mask])
    h, _, _ = np.histogram2d(log_nH, log_T, bins=(LOG_NH_BINS, LOG_T_BINS))
    return h


def snap_hist(snap_path):
    """Per-snapshot (log n_H, log T) histogram, cached to disk.

    Cache is invalidated if the snapshot's mtime advances or if the bin edges
    differ from what's stored in the cache file.
    """
    cpath = _cache_path(snap_path)
    snap_mtime = os.path.getmtime(snap_path)
    if os.path.exists(cpath):
        try:
            data = np.load(cpath)
            if (float(data["snap_mtime"]) >= snap_mtime
                    and np.array_equal(data["log_nh_bins"], LOG_NH_BINS)
                    and np.array_equal(data["log_t_bins"], LOG_T_BINS)):
                print(f"  cache hit  {snap_path}")
                return data["hist"]
        except Exception:
            pass  # fall through to rebuild
    print(f"  reading    {snap_path}")
    h = _build_snap_hist(snap_path)
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez(cpath,
             snap_mtime=np.array(snap_mtime),
             log_nh_bins=LOG_NH_BINS,
             log_t_bins=LOG_T_BINS,
             hist=h)
    return h


def percentiles_from_hist(hist, qs=(0.16, 0.5, 0.84)):
    """For each n_H column, return T at the requested CDF quantiles.

    NaN where the column is empty.
    """
    out = np.full((len(qs), hist.shape[0]), np.nan)
    col_totals = hist.sum(axis=1)
    for i in range(hist.shape[0]):
        n = col_totals[i]
        if n < 10:  # need at least a few particles for percentiles to be meaningful
            continue
        cdf = np.cumsum(hist[i]) / n
        for j, q in enumerate(qs):
            k = np.searchsorted(cdf, q)
            k = min(k, len(T_CENTERS) - 1)
            out[j, i] = T_CENTERS[k]
    return out


# Group fiducial runs by (cloud_R, logZ). Pooling realizations across drivings
# is done implicitly via the shared histogram for each group.
groups = {}
for run in FIDUCIAL_RUNS:
    key = (run.R_cloud, run.logZ)
    groups.setdefault(key, []).append(run)


fig, ax = plt.subplots(figsize=(4, 4))

# Track each cloud's 16/84 envelope in log space so annotations can dodge it.
data_envelopes = []  # list of (log_nH_centers, log_q16, log_q84) — NaNs already in q

for (R_cloud, logZ) in sorted(groups, key=lambda k: (k[0], k[1])):
    runs = groups[(R_cloud, logZ)]
    hist = np.zeros((len(NH_CENTERS), len(T_CENTERS)))
    n_real = 0
    for run in runs:
        snap = first_snapshot(ceph_path(run.path))
        if snap is None:
            print(f"skip (no snapshots): {run.path}")
            continue
        hist += snap_hist(snap)
        n_real += 1
    if n_real == 0:
        continue

    q = percentiles_from_hist(hist)
    color = logz_to_color(logZ)
    is_r10 = R_cloud == 10.0
    lw, ls = (1.2, "dotted") if is_r10 else (1.8, "solid")

    r_str = "R10" if is_r10 else "R1"
    z_str = f"{10.0 ** logZ:g}"
    base = f"M2e4_{r_str}_Z{z_str}"
    label = f"{base} ({n_real} realizations)" if n_real > 1 else base

    ax.plot(NH_CENTERS, q[1], color=color, lw=lw, ls=ls, label=label)
    ax.fill_between(NH_CENTERS, q[0], q[2], color=color, alpha=0.15, lw=0)

    data_envelopes.append((np.log10(NH_CENTERS), np.log10(q[0]), np.log10(q[2])))

xlim = (0.1, 1e10)
ylim = (1, 1e5)
ax.set(
    xscale="log", yscale="log",
    xlabel=r"$n_{\rm H}\ (\mathrm{cm}^{-3})$",
    ylabel=r"$T\ (\mathrm{K})$",
    xlim=xlim,
    ylim=ylim
)

# Build the legend before drawing guide-line annotations so we know its bbox
# and can keep annotations clear of it. Order M2e4_R10 first (low Σ),
# then M2e4_R1, matching sfe_vs_time.py.
_handles, _labels = ax.get_legend_handles_labels()
_order = sorted(range(len(_labels)),
                key=lambda i: (0 if "M2e4_R10" in _labels[i] else 1, _labels[i]))
leg = ax.legend(
    [_handles[i] for i in _order], [_labels[i] for i in _order],
    loc="lower left", fontsize=8, edgecolor="black",
    borderaxespad=0.4, labelspacing=0.0,
)
leg.get_frame().set_linewidth(0.8)

# Legend bbox in (log n_H, log T) for annotation dodging. Force a layout pass
# so the legend's window_extent reflects its final position.
fig.canvas.draw()
_leg_bbox_disp = leg.get_window_extent()
_inv = ax.transData.inverted()
(_lx0, _ly0) = _inv.transform((_leg_bbox_disp.x0, _leg_bbox_disp.y0))
(_lx1, _ly1) = _inv.transform((_leg_bbox_disp.x1, _leg_bbox_disp.y1))
LEGEND_LOG_BOX = (np.log10(_lx0), np.log10(_lx1), np.log10(_ly0), np.log10(_ly1))

# Lines of constant Jeans mass using the Hennebelle & Grudic 2024 ARAA Eq. 12
# convention: M_J = (pi^{5/2}/6) c_s^3 / sqrt(G^3 rho), with sound speed
# c_s^2 = k T / (mu m_H) for mu = 2.3 (fully molecular gas) and mass density
# rho = mu_H m_H n_H with mu_H = 1.4 (mean mass per H nucleus, He included).
# In M_sun: M_J = MJ_PREFACTOR * T^{3/2} / n_H^{1/2}.
MU = 2.3
MU_H = 1.4
_kB = c.k_B.cgs.value
_mH = c.m_p.cgs.value
_G = c.G.cgs.value
_Msun = c.M_sun.cgs.value
MJ_PREFACTOR = (
    (np.pi ** 2.5 / 6.0) * _kB**1.5
    / ((MU * _mH) ** 1.5 * _G**1.5 * np.sqrt(MU_H * _mH))
    / _Msun
)


def _display_angle_deg(x0, y0, x1, y1):
    """Visual angle (deg) of the segment in axes display pixels."""
    (px0, py0), (px1, py1) = ax.transData.transform([(x0, y0), (x1, y1)])
    return float(np.degrees(np.arctan2(py1 - py0, px1 - px0)))


def _clip_to_box(T_of_nH, nH_at_T):
    """Return (nH_min, nH_max) where the monotonic curve T_of_nH stays in the
    plot box. nH_at_T is the inverse. Works for both positive and negative
    slopes (Jeans-mass lines and isobars).
    """
    n_y_min, n_y_max = sorted((nH_at_T(ylim[0]), nH_at_T(ylim[1])))
    n_min = max(xlim[0], n_y_min)
    n_max = min(xlim[1], n_y_max)
    if n_min >= n_max:
        return None
    return n_min, n_max


def _envelope_clearance(log_nH_c, log_T_c):
    """Signed log clearance of (log_nH_c, log_T_c) from data envelopes and legend.

    Positive = outside everything (good); negative = inside one obstacle.
    """
    out = np.inf
    for d_logx, d_q16, d_q84 in data_envelopes:
        valid = np.isfinite(d_q16) & np.isfinite(d_q84)
        if valid.sum() < 2:
            continue
        if log_nH_c < d_logx[valid][0] or log_nH_c > d_logx[valid][-1]:
            continue
        q16 = float(np.interp(log_nH_c, d_logx[valid], d_q16[valid]))
        q84 = float(np.interp(log_nH_c, d_logx[valid], d_q84[valid]))
        if q16 <= log_T_c <= q84:
            depth = min(log_T_c - q16, q84 - log_T_c)
            out = min(out, -depth)
        else:
            d = min(abs(log_T_c - q16), abs(log_T_c - q84))
            out = min(out, d)
    # Legend bbox: forbidden interior, distance to the rectangle if outside.
    lx0, lx1, ly0, ly1 = LEGEND_LOG_BOX
    inside_x = lx0 <= log_nH_c <= lx1
    inside_y = ly0 <= log_T_c <= ly1
    if inside_x and inside_y:
        depth = min(log_nH_c - lx0, lx1 - log_nH_c,
                    log_T_c - ly0, ly1 - log_T_c)
        out = min(out, -depth)
    else:
        dx = 0 if inside_x else min(abs(log_nH_c - lx0), abs(log_nH_c - lx1))
        dy = 0 if inside_y else min(abs(log_T_c - ly0), abs(log_T_c - ly1))
        out = min(out, float(np.hypot(dx, dy)))
    return out


# Perpendicular text offset (in log units) used when scoring va="bottom"/"top".
# Approximates where the rendered text actually sits relative to its anchor.
_TEXT_OFFSET_LOG = 0.08


def _annotate_diagonal(x_arr, y_arr, label, color):
    """Place label along the curve, picking the spot with most clearance."""
    log_x = np.log10(x_arr)
    log_y = np.log10(y_arr)
    n = len(x_arr)
    # Stay 15% in from each end so labels don't extend past the plot edges.
    lo, hi = max(1, int(0.15 * n)), min(n - 2, int(0.85 * n))
    candidates = np.unique(np.linspace(lo, hi, 30).astype(int))

    best = (lo, "bottom", -np.inf)
    for i in candidates:
        for sign, va in ((+1, "bottom"), (-1, "top")):
            log_T_eff = log_y[i] + sign * _TEXT_OFFSET_LOG
            # Also stay inside the plot box vertically.
            if not (np.log10(ylim[0]) < log_T_eff < np.log10(ylim[1])):
                continue
            score = _envelope_clearance(log_x[i], log_T_eff)
            if score > best[2]:
                best = (int(i), va, score)

    i_best, va_best, _ = best
    nxt = i_best + 1 if i_best + 1 < n else i_best - 1
    angle = _display_angle_deg(x_arr[i_best], y_arr[i_best],
                               x_arr[nxt], y_arr[nxt])
    ax.text(x_arr[i_best], y_arr[i_best], label, fontsize=6, color=color,
            ha="center", va=va_best, rotation=angle, rotation_mode="anchor",
            zorder=0)


def _draw_curve(nH_min, nH_max, T_func, color, ls):
    nH = np.logspace(np.log10(nH_min), np.log10(nH_max), 100)
    T = T_func(nH)
    ax.plot(nH, T, color=color, lw=0.6, ls=ls, zorder=0)
    return nH, T


# Jeans-mass lines (slope +1/3 in log-log).
for log_mj in (-2, 0, 2, 4, 6):
    M_J = 10.0**log_mj
    A = (M_J / MJ_PREFACTOR) ** (2.0 / 3.0)  # T = A * n^(1/3)
    T_of_n = lambda n, A=A: A * n ** (1.0 / 3.0)
    n_of_T = lambda T, A=A: (T / A) ** 3
    clip = _clip_to_box(T_of_n, n_of_T)
    if clip is None:
        continue
    nH_arr, T_arr = _draw_curve(*clip, T_of_n, "0.6", "--")
    label = rf"$M_{{\rm J}}=10^{{{log_mj}}}\,M_\odot$" if log_mj != 0 else r"$M_{\rm J}=1\,M_\odot$"
    _annotate_diagonal(nH_arr, T_arr, label, "0.4")

# Isobars of constant P/k_B = n_H * T (units of K cm^-3). Slope -1 in log-log.
ISOBAR_COLOR = "steelblue"
for log_p in (3, 5, 7, 9, 11):
    P_over_kB = 10.0**log_p
    T_of_n = lambda n, P=P_over_kB: P / n
    n_of_T = lambda T, P=P_over_kB: P / T
    clip = _clip_to_box(T_of_n, n_of_T)
    if clip is None:
        continue
    nH_arr, T_arr = _draw_curve(*clip, T_of_n, ISOBAR_COLOR, ":")
    label = rf"$10^{{{log_p}}}\,\mathrm{{K\,cm^{{-3}}}}$"
    _annotate_diagonal(nH_arr, T_arr, label, ISOBAR_COLOR)

fig.tight_layout()

out_pdf = "nH_vs_T.pdf"
fig.savefig(out_pdf, bbox_inches="tight")
plt.close(fig)
print(f"wrote {out_pdf}")
