PLOTDIR := imf_plots
MODEL := chabrier_smooth_bounds

# Four PDFs:
#   - count-mode histogram + posterior IMF comparison
#   - normalized (PDF) histogram + posterior IMF comparison
#   - alpha plot of Gamma_IMF vs M
#   - multi-panel timelapse surface-density map
COMPARE_PDF        := $(PLOTDIR)/comparison_$(MODEL).pdf
COMPARE_NORM_PDF   := $(PLOTDIR)/comparison_$(MODEL)_normalized.pdf
ALPHA_PDF          := $(PLOTDIR)/alphaplot_$(MODEL).pdf
ALPHA_OBS_PDF      := $(PLOTDIR)/alphaplot_obs_$(MODEL).pdf
LOGNORMAL_PDF      := $(PLOTDIR)/lognormal.pdf
MULTIPANEL_PDF     := multipanel.pdf
MCLUSTER_PDF       := Mcluster_vs_Mmax.pdf

ALL_PLOTS := $(COMPARE_PDF) $(COMPARE_NORM_PDF) $(ALPHA_PDF) $(ALPHA_OBS_PDF) $(LOGNORMAL_PDF) $(MULTIPANEL_PDF) $(MCLUSTER_PDF)


.PHONY: all clean compare_imfs alpha alpha-obs lognormal multipanel mcluster
all: $(ALL_PLOTS)

compare_imfs: $(COMPARE_PDF) $(COMPARE_NORM_PDF)
alpha: $(ALPHA_PDF)
alpha-obs: $(ALPHA_OBS_PDF)
lognormal: $(LOGNORMAL_PDF)
multipanel: $(MULTIPANEL_PDF)
mcluster: $(MCLUSTER_PDF)

# imf_plots.py writes both the count and normalized comparison PDFs in one run;
# we list both targets to a common recipe so either being missing triggers it.
$(COMPARE_PDF) $(COMPARE_NORM_PDF): imf_plots.py
	python imf_plots.py

$(ALPHA_PDF) $(ALPHA_OBS_PDF): imf_alphaplot.py
	python imf_alphaplot.py

$(LOGNORMAL_PDF): lognormal.py
	python lognormal.py

$(MULTIPANEL_PDF): multipanel_maps.py
	python multipanel_maps.py

$(MCLUSTER_PDF): mclusters_vs_mmax.py
	python mclusters_vs_mmax.py

clean:
	rm -rf $(PLOTDIR) $(MULTIPANEL_PDF) $(MCLUSTER_PDF)
