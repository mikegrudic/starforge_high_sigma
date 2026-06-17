"""python sigma_movie_vms_com.py [SIM_OUTPUT_DIR] [N_PROC] [--perspective]

Render a CrunchSnaps SinkVis surface-density timelapse of the M2e4_R10_Z0.01
simulation, centered per-frame on the COM of the four most-massive stars, with

    rmax = max(0.01 pc, RMAX_MARGIN * max_i |r_i - r_COM|)

where i runs over the four stars (or those that have formed so far). Defaults to
the Res271 turbsphere_driving1 variant; pass any other output dir to override.
Renders into ./sigma_movie_VMS_COM/ then encodes an mp4 with ffmpeg.

With --perspective, switches to 3D perspective rendering using the PIL backend.
camera_distance is set proportional to the (smoothed) max-from-COM separation
(camera_distance = PERSPECTIVE_CAMERA_FACTOR * separation), and the angular FOV
(rmax = RMAX_MARGIN / PERSPECTIVE_CAMERA_FACTOR, in tan(half-angle) units) is
held constant so the stars subtend the same image fraction as in orthographic
mode. Output goes to ./sigma_movie_VMS_COM_perspective/.
"""

import os
import subprocess
import sys
from glob import glob

import h5py
import numpy as np
from astropy import units as u
from natsort import natsorted

import CrunchSnaps

N_COM_STARS = 4
RMAX_FLOOR_PC = 0.01
RMAX_MARGIN = 1.0
RES = 1024
FPS = 24
RMAX_SMOOTH_KYR = 30.0      # Gaussian sigma for FOV smoothing
SIGMA_LIMITS = (3e3, 3e6)   # surface density colormap range (Msun/pc^2)
PERSPECTIVE_CAMERA_FACTOR = 5.0  # camera_distance = factor * separation (--perspective)

CODE_TO_KYR = float((u.pc / (u.m / u.s)).to(u.kyr))

DEFAULT_SIM = (
    "/mnt/ceph/users/starforge/STARFORGE_RT/STARFORGE_v1.2/M2e4_R1/"
    "M2e4_R1_Z0.01_S0_A2_B0.1_I10000_Res271_n2_sol0.5_42/output_turbsphere_driving1"
)
OUTPUT = "sigma_movie_VMS_COM"


def read_star_records(snaps):
    times = np.empty(len(snaps), dtype=np.float64)
    records = []
    for i, s in enumerate(snaps):
        with h5py.File(s, "r") as F:
            times[i] = F["Header"].attrs["Time"]
            if "PartType5" in F:
                ids = F["PartType5/ParticleIDs"][:].astype(np.uint64)
                masses = F["PartType5/BH_Mass"][:].astype(np.float32)
                coords = F["PartType5/Coordinates"][:].astype(np.float32)
            else:
                ids = np.empty(0, dtype=np.uint64)
                masses = np.empty(0, dtype=np.float32)
                coords = np.empty((0, 3), dtype=np.float32)
        records.append((ids, masses, coords))
    return times, records


def find_top_ids(records, n):
    max_mass_by_id = {}
    for ids, masses, _ in records:
        for uid, m in zip(ids.tolist(), masses.tolist()):
            if m > max_mass_by_id.get(uid, 0.0):
                max_mass_by_id[uid] = m
    ranked = sorted(max_mass_by_id, key=max_mass_by_id.get, reverse=True)
    return set(ranked[:n])


def encode_movie(folder, prefix, fps):
    frames = natsorted(
        f for f in glob(os.path.join(folder, prefix + "_*.png")) if "_incomplete" not in f
    )
    if len(frames) < 2:
        print(f"Skipping movie for {prefix}: only {len(frames)} frame(s)")
        return
    listfile = os.path.join(folder, f".{prefix}_framelist.txt")
    with open(listfile, "w") as f:
        for fr in frames:
            f.write(f"file {os.path.abspath(fr)}\nduration {1/fps}\n")
    out = os.path.join(folder, f"{prefix}.mp4")
    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", listfile,
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(fps), out,
    ]
    print(f"Encoding {out} from {len(frames)} frames...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    os.remove(listfile)
    if result.returncode != 0:
        print("ffmpeg error:\n" + result.stderr)
    else:
        print("Wrote", out)


def main():
    perspective = "--perspective" in sys.argv
    argv = [a for a in sys.argv if a != "--perspective"]
    sim = argv[1] if len(argv) > 1 else DEFAULT_SIM
   # nproc default is conservative: each worker buffers ≥2 snapshots for
    nproc = int(argv[2]) if len(argv) > 2 else 8

    snaps = natsorted(glob(os.path.join(sim, "snapshot_*.hdf5")))
    if not snaps:
        raise SystemExit(f"No snapshots found under {sim}")
    print(f"Found {len(snaps)} snapshots in {sim}")

    times, records = read_star_records(snaps)
    top_ids = find_top_ids(records, N_COM_STARS)
    print(f"Top-{N_COM_STARS} IDs (by max mass): {sorted(top_ids)}")

    out_dir = OUTPUT + ("_perspective" if perspective else "")
    os.makedirs(out_dir, exist_ok=True)

    # Per-snapshot COM and rmax for snapshots where ≥1 of the top-4 has formed.
    com_per_snap = np.full((len(records), 3), np.nan, dtype=np.float64)
    rmax_per_snap = np.full(len(records), np.nan, dtype=np.float64)
    for i, (ids, masses, coords) in enumerate(records):
        in_set = np.fromiter((int(j) in top_ids for j in ids), dtype=bool, count=len(ids))
        if not in_set.any():
            continue
        ms = masses[in_set].astype(np.float64)
        cs = coords[in_set].astype(np.float64)
        com = (cs * ms[:, None]).sum(axis=0) / ms.sum()
        dists = np.linalg.norm(cs - com, axis=1)
        com_per_snap[i] = com
        rmax_per_snap[i] = max(RMAX_FLOOR_PC, RMAX_MARGIN * float(dists.max()))

    valid = ~np.isnan(rmax_per_snap)
    t_valid = times[valid]
    com_valid = com_per_snap[valid]
    rmax_valid = rmax_per_snap[valid]

    # Smooth rmax(t) with a Gaussian kernel of width RMAX_SMOOTH_KYR. Sigma is
    # converted to code-time units. The kernel is built from the actual sample
    # times so non-uniform snapshot spacing is handled correctly.
    sigma_code = RMAX_SMOOTH_KYR / CODE_TO_KYR
    dt_matrix = t_valid[:, None] - t_valid[None, :]
    w = np.exp(-0.5 * (dt_matrix / sigma_code) ** 2)
    rmax_smooth = (w * rmax_valid).sum(axis=1) / w.sum(axis=1)

    # Frame cadence is locked to the simulation: inter-frame sim time = median
    # dt of the last 10 snapshots, so playback at FPS advances at the natural
    # snapshot rate. Movie length is derived (n_frames / FPS seconds).
    # Anchor the grid backward from t_valid[-1] so every snapshot in the
    # constant-cadence tail lands exactly on a frame. Anchoring at t_valid[0]
    # is unsafe — the early snapshot cadence may differ from dt_snap, so the
    # forward grid drifts off the snapshot times by the end.
    dt_snap = float(np.median(np.diff(times[-10:])))
    dt_frame = dt_snap/3
    n_half = int(np.floor((t_valid[-1] - t_valid[0]) / dt_frame))
    frame_times = t_valid[-1] - dt_frame * np.arange(n_half, -1, -1)
    #assert np.allclose(frame_times[-9::2], times[-5:])
    n_frames = len(frame_times)
    print(
        f"dt_snap = {dt_snap * CODE_TO_KYR:.3g} kyr -> "
        f"{n_frames} frames, movie length {n_frames / FPS:.2f}s at {FPS} fps"
    )
    frame_rmax = np.interp(frame_times, t_valid, rmax_smooth)
    frame_com = np.column_stack(
        [np.interp(frame_times, t_valid, com_valid[:, k]) for k in range(3)]
    )

    time_offset = float(times[0])
    # In perspective mode rmax is angular (tan of half-FOV); pick it so the
    # stars subtend the same image fraction as orthographic mode at the focal
    # plane, i.e. rmax * camera_distance == RMAX_MARGIN * separation.
    rmax_angular = RMAX_MARGIN / PERSPECTIVE_CAMERA_FACTOR
    params = []
    for j in range(n_frames):
        com = frame_com[j]
        # Pass center as "x,y,z" string. CrunchSnaps' SinkVis.__init__ does a
        # Python `any(...)` over the center params, which raises on an ndarray
        # of length 3 ("truth value ambiguous"). A string takes the same code
        # path through assign_center, which parses the comma form back to an
        # ndarray before rendering.
        frame_params = {
            "Time": float(frame_times[j]),
            "index": j,
            "center": "{:.10g},{:.10g},{:.10g}".format(*com),
            "res": RES,
            "outputfolder": out_dir,
            "threads": -1,
            "overwrite": False,
            "realstars": False,
            "time_offset": time_offset,
            "limits": np.array(SIGMA_LIMITS),
            "star_legend": True,
            "tight_bbox": False,
        }
        if perspective:
            separation = float(frame_rmax[j]) / RMAX_MARGIN
            frame_params.update({
                "camera_distance": PERSPECTIVE_CAMERA_FACTOR * separation,
                "rmax": rmax_angular,
                "backend": "PIL",
            })
        else:
            frame_params.update({
                "rmax": float(frame_rmax[j]),
                "backend": "matplotlib",
            })
        params.append(frame_params)

    print(f"Rendering {len(params)} interpolated frames into {out_dir}/")
    CrunchSnaps.DoTasksForSimulation(
        snaps=snaps,
        task_types=[CrunchSnaps.SinkVisSigmaGas],
        task_params=[params],
        nproc=nproc,
        nthreads=-1,
    )

    encode_movie(out_dir, "SurfaceDensity", FPS)


if __name__ == "__main__":
    main()
