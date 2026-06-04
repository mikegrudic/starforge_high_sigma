from sys import argv
from os.path import isfile, isdir
from os import mkdir
import matplotlib

matplotlib.use("Agg")
import numpy as np
import salpyter
from matplotlib import pyplot as plt
from joblib import delayed, Parallel

# imfs = {}
# for run in argv[1:]:


SAMPLE_CHAINLENGTH = 10**5
OVERWRITE = True
PLOT_SAMPLES = True


def run_imf_analysis(run):
    if isfile(run + "/IMF.dat"):
        masses = np.loadtxt(run + "/IMF.dat")
        if len(masses.shape) < 2:
            return
        masses = masses[:, 1]
    else:
        return
    if not isdir(run + "/imf_samples"):
        mkdir(run + "/imf_samples")

    MMIN, MMAX = 0.1, 2000
    fit_masses = masses[(masses > MMIN) * (masses < MMAX)]
    # fit_masses = np.array(10 * list(fit_masses))
    MMIN, MMAX = fit_masses.min(), fit_masses.max()
    mgrid = np.logspace(-3, 4, 100001)
    NUM_HIST_BINS = 31
    mbins = np.logspace(-3, 4, 1 + NUM_HIST_BINS)
    logm = np.log10(mgrid)

    for imf_model in "chabrier", "chabrier_smooth", "chabrier_smooth_bounds", "chabrier_smooth_lognormal", "chabrier_smooth_bounds_lognormal", "chabrier_bounds", "chabrier_exp_bounds":  # salpyter.IMF_LIST:
        print(run, imf_model)
        imf_func = salpyter.get_imf_function(imf_model)  # getattr(salpyter, imf_model + "_imf")
        _, ax = plt.subplots(1, 1, figsize=(4, 4))
        _, bins = ax.hist(masses, mbins, histtype="step", color="black")[:2]
        imf_to_bins = np.log10(bins.max() / bins.min()) / (len(bins) - 1) * len(fit_masses)
        pmax = salpyter.imf_default_params(imf_model)

        sol = salpyter.imf_mostlikely_params(fit_masses, imf_model)
        pmax = sol.x
        imf = imf_func(logm, pmax, logmmin=np.log10(MMIN), logmmax=np.log10(MMAX))  # , check_norm=True)
        samplepath = run + f"/imf_samples/samples_{imf_model}.npy"

        if (not isfile(samplepath)) or OVERWRITE:
            try:
                samples = salpyter.imf_lnprob_samples(
                    fit_masses,
                    model=imf_model,
                    p0=pmax,
                    chainlength=SAMPLE_CHAINLENGTH,
                    logmmin=np.log10(MMIN),
                    logmmax=np.log10(MMAX),
                )
                print(np.median(samples))
                np.save(samplepath, samples)
            except:
                raise Warning(f"Could not generate samples for {run}+{imf_model}. Initial guess: {pmax}")
                return
        else:
            samples = np.load(samplepath)

        cut = logm > -np.inf  # (logm > np.log10(MMIN)) * (logm < np.log10(MMAX))

        if PLOT_SAMPLES:
            num_lines_to_plot = 100
            for s in samples[:: len(samples) // num_lines_to_plot]:
                # print(s)
                imf = imf_func(logm, s, logmmin=np.log10(MMIN), logmmax=np.log10(MMAX))  # np.log10(MMAX))
                ax.plot(mgrid[cut], imf[cut] * imf_to_bins, color="steelblue", lw=0.2, alpha=1)

        imf = salpyter.get_imf_function("chabrier_smooth")(
            logm, salpyter.CHABRIER_SMOOTH_DEFAULT_PARAMS, logmmin=np.log10(MMIN), logmmax=np.log10(MMAX)
        )
        ax.plot(mgrid, imf * imf_to_bins, color="black", ls="dotted", label="Chabrier 2005")

        #        cut = (mgrid > MMIN) * (mgrid < MMAX)
        #        imf = imf_func(logm, pmax, logmmin=np.log10(MMIN), logmmax=np.log10(MMAX))
        #        ax.plot(mgrid[cut], imf[cut] * imf_to_bins, color="black", ls="dotted")

        ax.set(
            yscale="log",
            xscale="log",
            xlim=[0.01, 5000],
            ylim=[0.1, 1000],
            ylabel="N",
            xlabel=r"$M\,\left(M_\odot\right)$",
        )
        ax.legend()
        plt.savefig(run + f"/imf_samples/IMF_{imf_model}.pdf", bbox_inches="tight")
        plt.close()


# [run_imf_analysis(run) for run in argv[1:]]
from glob import glob

dirs = [l.split("/IMF.dat")[0] for l in glob("imf_data/STARFORGE_RT/STARFORGE_v1.2/*/*/*/IMF.dat")]
Parallel(n_jobs=min(32, len(dirs)))(delayed(run_imf_analysis)(run) for run in dirs)
