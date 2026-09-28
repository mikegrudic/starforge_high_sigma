"""IMF analysis driver: posterior samples of the chabrier_smooth_bounds parameters by nested sampling.

For every run in ``sim_paths.ANALYSIS_RUNS`` with an ``IMF.dat``, fits the stars in
``MMIN < m < MMAX`` with salpyter's ``chabrier_smooth_bounds`` model under its default uniform priors
(the ones stated in the appendix) and writes, per run directory,

  imf_samples_nested/samples_<model>.npy   equal-weight posterior samples (both runs merged)
  imf_samples_nested/summary_<model>.json  log-evidence and convergence check
  imf_samples_nested/IMF_<model>.pdf       histogram + posterior IMF curves

Why nested sampling (dynesty) rather than NUTS: the posterior has hard walls (the likelihood
vanishes wherever m_min or m_max crosses a star) and can be multimodal (a Chabrier-like mode and a
pure-lognormal mode in which Gamma is irrelevant). NUTS adapts its step size down at the walls and
does not cross between modes. Nested sampling needs no gradients and weighs modes by their
posterior mass.

Convergence check: each run is fitted twice with independent seeds, and the summary records both
log-evidences and the posterior medians of each run, which should agree within their errors.

The m_min and m_max priors are truncated to the range the data allow (m_min < min(m),
m_max > max(m)). The likelihood is zero outside it, so the posterior is unchanged; the evidence
is corrected back to the full prior box.

Requires a salpyter whose Chabrier IMFs are evaluated in log space (branch logspace-chabrier). The
linear-space form returns inf in part of the prior box (Gamma > 0 with a wide lognormal) that
nested sampling explores and NUTS never did. Point ``SALPYTER_SRC`` at its ``src/`` if the installed
salpyter predates the fix; the script refuses to run without it.
"""

import json
import os
import sys

if os.environ.get("SALPYTER_SRC"):
    sys.path.insert(0, os.environ["SALPYTER_SRC"])
os.environ.setdefault("JAX_PLATFORMS", "cpu")
# one XLA thread per worker: the parallelism is across runs
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["text.usetex"] = False

import numpy as np
from joblib import Parallel, delayed
from matplotlib import pyplot as plt

from sim_paths import ANALYSIS_RUNS

MODEL = "chabrier_smooth_bounds"
MMIN, MMAX = 0.1, 2000
NLIVE = 2000
SEEDS = (0, 1)
OUTDIR = "imf_samples_nested"
OVERWRITE = False
N_JOBS = 16


def _check_salpyter():
    import jax.numpy as jnp
    import salpyter
    f = salpyter.get_imf_function(MODEL)
    corner = jnp.array([np.log10(0.06), 1.92, 1.69, -3.0, 3.6])  # break mass ~1e-182 Msun
    if not np.all(np.isfinite(np.asarray(f(jnp.linspace(-1.0, 2.0, 50), corner)))):
        raise SystemExit(f"salpyter at {salpyter.__file__} overflows in part of the prior box; "
                         "set SALPYTER_SRC to a checkout of branch logspace-chabrier")


def _nested(fit_masses, seed):
    import dynesty
    import jax
    import jax.numpy as jnp
    import salpyter
    f = salpyter.get_imf_function(MODEL)
    lm = jnp.log10(jnp.asarray(fit_masses))
    a, b = float(lm.min()), float(lm.max())
    # the same per-star floor salpyter's own likelihood uses, for IMFs that underflow at a star
    loglike = jax.jit(lambda p: jnp.sum(jnp.log(jnp.clip(f(lm, p), 1e-300, None))))
    box = np.asarray(salpyter.imf_default_bounds(MODEL), dtype=float)
    lo, hi = box[:, 0].copy(), box[:, 1].copy()
    lo[4], hi[3] = b, a
    sampler = dynesty.NestedSampler(lambda p: float(loglike(jnp.asarray(p))), lambda u: lo + (hi - lo) * u,
                                    5, nlive=NLIVE, bound="multi", sample="rslice",
                                    rstate=np.random.default_rng(seed))
    sampler.run_nested(dlogz=0.01, print_progress=False)
    res = sampler.results
    # evidence under the full prior box: the truncated m_min/m_max ranges hold all the likelihood
    log_trunc = np.log((a - box[3, 0]) / (box[3, 1] - box[3, 0])) + np.log((box[4, 1] - b) / (box[4, 1] - box[4, 0]))
    return res, float(res.logz[-1] + log_trunc), float(res.logzerr[-1])


def run_imf_analysis(run):
    import dynesty.utils
    import salpyter
    if not os.path.isfile(run + "/IMF.dat"):
        return
    masses = np.loadtxt(run + "/IMF.dat")
    if masses.ndim < 2:
        return
    masses = masses[:, 1]
    fit_masses = masses[(masses > MMIN) & (masses < MMAX)]
    os.makedirs(os.path.join(run, OUTDIR), exist_ok=True)
    samplepath = os.path.join(run, OUTDIR, f"samples_{MODEL}.npy")

    if OVERWRITE or not os.path.isfile(samplepath):
        fits = [_nested(fit_masses, s) for s in SEEDS]
        merged = dynesty.utils.merge_runs([r for r, _, _ in fits])
        samples = merged.samples_equal(rstate=np.random.default_rng(0))
        np.save(samplepath, samples)
        per_run = []
        for (r, logz, logzerr), s in zip(fits, SEEDS):
            eq = r.samples_equal(rstate=np.random.default_rng(s))
            per_run.append(dict(seed=s, logz=logz, logzerr=logzerr, ncall=int(np.sum(r.ncall)),
                                median=np.median(eq, axis=0).tolist(),
                                p16=np.percentile(eq, 16, axis=0).tolist(),
                                p84=np.percentile(eq, 84, axis=0).tolist()))
        with open(os.path.join(run, OUTDIR, f"summary_{MODEL}.json"), "w") as fh:
            json.dump(dict(model=MODEL, n_fit=int(fit_masses.size), nlive=NLIVE,
                           param_names=["logm0", "logsigma", "alpha", "logmmin", "logmmax"], runs=per_run), fh, indent=1)
    else:
        samples = np.load(samplepath)

    # diagnostic plot: histogram and 100 posterior IMF curves
    mgrid = np.logspace(-3, 4, 2001)
    mbins = np.logspace(-3, 4, 32)
    lmin, lmax = np.log10(fit_masses.min()), np.log10(fit_masses.max())
    _, ax = plt.subplots(1, 1, figsize=(4, 4))
    _, bins = ax.hist(masses, mbins, histtype="step", color="black")[:2]
    imf_to_bins = np.log10(bins.max() / bins.min()) / (len(bins) - 1) * len(fit_masses)
    imf_func = salpyter.get_imf_function(MODEL)
    for s in samples[:: max(1, len(samples) // 100)]:
        ax.plot(mgrid, np.asarray(imf_func(np.log10(mgrid), s, lmin, lmax)) * imf_to_bins,
                color="steelblue", lw=0.2)
    ref = salpyter.get_imf_function("chabrier_smooth")(
        np.log10(mgrid), np.asarray(salpyter.CHABRIER_SMOOTH_DEFAULT_PARAMS), logmmin=lmin, logmmax=lmax)
    ax.plot(mgrid, np.asarray(ref) * imf_to_bins, color="black", ls="dotted", label="Chabrier 2005")
    ax.set(yscale="log", xscale="log", xlim=[0.01, 5000], ylim=[0.1, 1000], ylabel="N",
           xlabel=r"$M\,\left(M_\odot\right)$")
    ax.legend()
    plt.savefig(os.path.join(run, OUTDIR, f"IMF_{MODEL}.pdf"), bbox_inches="tight")
    plt.close()
    print(run, "done", flush=True)


if __name__ == "__main__":
    _check_salpyter()
    Parallel(n_jobs=N_JOBS)(delayed(run_imf_analysis)(r.path) for r in ANALYSIS_RUNS)
