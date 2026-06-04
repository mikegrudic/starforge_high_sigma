"""IMF analysis driver — JAX/NUTS variant.

Same structure as IMF_analysis.py but uses the salpyter ``jax`` branch
(differentiable IMFs + NUTS via blackjax) instead of master's emcee path.
Drop-in replacement at the file level: same outputs (``samples_<model>.npy``
and ``IMF_<model>.pdf`` per directory) in the same layout.

Notes
-----
* The script prepends the ``jax`` branch's ``src/`` to ``sys.path`` so it
  imports the JAX salpyter without touching whatever's pip-installed.
* JAX is pinned to the CPU backend. Each joblib worker is a fresh Python
  process with its own XLA cache, so there's no GPU contention even if
  jax-metal is installed; the per-process JIT compile cost (~2 s) is paid
  once per (model, masses-shape) combination.
* The emcee ``chainlength`` knob is replaced by ``(NUM_WARMUP, NUM_SAMPLES)``.
  NUTS samples are near-independent so 2000 of them carries much more
  information than emcee's chainlength=1e5 thinned by 1000.
"""

import os
import sys

# Use JAX salpyter from the jax-branch worktree without disturbing the
# installed package. Adjust this path if the worktree lives elsewhere.
_JAX_SALPYTER_SRC = "/Users/mgrudic/code/salpyter-jax/src"
if os.path.isdir(_JAX_SALPYTER_SRC):
    sys.path.insert(0, _JAX_SALPYTER_SRC)

# Force CPU before any JAX import. Multiple joblib workers each grabbing the
# Metal/GPU device would oversubscribe; CPU is plenty for these 3-param fits.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from glob import glob
from os import mkdir
from os.path import isdir, isfile

import matplotlib

matplotlib.use("Agg")

import numpy as np
import salpyter
from joblib import Parallel, delayed
from matplotlib import pyplot as plt


NUM_WARMUP = 500
NUM_SAMPLES = 2000
OVERWRITE = False
PLOT_SAMPLES = True


def run_imf_analysis(run):
    if not isfile(run + "/IMF.dat"):
        return
    masses = np.loadtxt(run + "/IMF.dat")
    if masses.ndim < 2:
        return
    masses = masses[:, 1]
    if not isdir(run + "/imf_samples_jax"):
        mkdir(run + "/imf_samples_jax")

    MMIN, MMAX = 0.1, 2000
    fit_masses = masses[(masses > MMIN) & (masses < MMAX)]
    MMIN, MMAX = fit_masses.min(), fit_masses.max()
    mgrid = np.logspace(-3, 4, 100001)
    NUM_HIST_BINS = 31
    mbins = np.logspace(-3, 4, 1 + NUM_HIST_BINS)
    logm = np.log10(mgrid)

    for imf_model in "chabrier", "chabrier_smooth", "chabrier_smooth_bounds", "chabrier_bounds", "chabrier_smooth_bounds_lognormal", "chabrier_smooth_lognormal":
        print(run, imf_model)
        imf_func = salpyter.get_imf_function(imf_model)
        _, ax = plt.subplots(1, 1, figsize=(4, 4))
        _, bins = ax.hist(masses, mbins, histtype="step", color="black")[:2]
        imf_to_bins = np.log10(bins.max() / bins.min()) / (len(bins) - 1) * len(fit_masses)

        sol = salpyter.imf_mostlikely_params(
            fit_masses,
            imf_model,
            logmmin=np.log10(MMIN),
            logmmax=np.log10(MMAX),
        )
        pmax = sol.x

        samplepath = run + f"/imf_samples_jax/samples_{imf_model}.npy"
        if (not isfile(samplepath)) or OVERWRITE:
            try:
                samples = salpyter.imf_lnprob_samples(
                    fit_masses,
                    model=imf_model,
                    p0=pmax,
                    num_warmup=NUM_WARMUP,
                    num_samples=NUM_SAMPLES,
                    logmmin=np.log10(MMIN),
                    logmmax=np.log10(MMAX),
                    seed=0,
                )
                print(np.median(samples, axis=0))
                np.save(samplepath, samples)
            except Exception as e:
                raise Warning(
                    f"Could not generate samples for {run}+{imf_model}. " f"Initial guess: {pmax}. Error: {e!r}"
                )
        else:
            samples = np.load(samplepath)

        if PLOT_SAMPLES:
            num_lines_to_plot = 100
            stride = max(1, len(samples) // num_lines_to_plot)
            for s in samples[::stride]:
                imf = np.asarray(imf_func(logm, s, logmmin=np.log10(MMIN), logmmax=np.log10(MMAX)))
                ax.plot(mgrid, imf * imf_to_bins, color="steelblue", lw=0.2, alpha=1)

        ref_imf = np.asarray(
            salpyter.get_imf_function("chabrier_smooth")(
                logm,
                np.asarray(salpyter.CHABRIER_SMOOTH_DEFAULT_PARAMS),
                logmmin=np.log10(MMIN),
                logmmax=np.log10(MMAX),
            )
        )
        ax.plot(mgrid, ref_imf * imf_to_bins, color="black", ls="dotted", label="Chabrier 2005")

        ax.set(
            yscale="log",
            xscale="log",
            xlim=[0.01, 5000],
            ylim=[0.1, 1000],
            ylabel="N",
            xlabel=r"$M\,\left(M_\odot\right)$",
        )
        ax.legend()
        plt.savefig(run + f"/imf_samples_jax/IMF_{imf_model}.pdf", bbox_inches="tight")
        plt.close()


dirs = [l.split("/IMF.dat")[0] for l in glob("imf_data/STARFORGE_RT/STARFORGE_v1.2/*/*/*/IMF.dat")]
##for run in dirs:
#    run_imf_analysis(run)
Parallel(n_jobs=-1)(delayed(run_imf_analysis)(run) for run in dirs)
