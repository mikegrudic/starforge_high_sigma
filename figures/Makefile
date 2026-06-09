PLOTDIR := imf_plots
MODEL := chabrier_smooth_bounds

COMPARE_PDF        := $(PLOTDIR)/comparison_$(MODEL).pdf
COMPARE_NORM_PDF   := $(PLOTDIR)/comparison_$(MODEL)_normalized.pdf
ALPHA_PDF          := $(PLOTDIR)/alphaplot_$(MODEL).pdf
ALPHA_OBS_PDF      := $(PLOTDIR)/alphaplot_obs_$(MODEL).pdf
LOGNORMAL_PDF      := $(PLOTDIR)/lognormal.pdf
MCLUSTER_PDF       := Mcluster_vs_Mmax.pdf
SFE_PDF            := sfe_vs_time.pdf

ALL_PLOTS := $(COMPARE_PDF) $(COMPARE_NORM_PDF) $(ALPHA_PDF) $(ALPHA_OBS_PDF) $(LOGNORMAL_PDF) $(MCLUSTER_PDF) $(SFE_PDF)


.PHONY: all clean compare_imfs alpha alpha-obs lognormal mcluster sfe
all: $(ALL_PLOTS)

compare_imfs: $(COMPARE_PDF) $(COMPARE_NORM_PDF)
alpha: $(ALPHA_PDF)
alpha-obs: $(ALPHA_OBS_PDF)
lognormal: $(LOGNORMAL_PDF)
mcluster: $(MCLUSTER_PDF)
sfe: $(SFE_PDF)

# imf_plots.py writes both the count and normalized comparison PDFs in one run;
# we list both targets to a common recipe so either being missing triggers it.
$(COMPARE_PDF) $(COMPARE_NORM_PDF): imf_plots.py
	python imf_plots.py

$(ALPHA_PDF) $(ALPHA_OBS_PDF): imf_alphaplot.py
	python imf_alphaplot.py

$(LOGNORMAL_PDF): lognormal.py
	python lognormal.py

$(MCLUSTER_PDF): mclusters_vs_mmax.py
	python mclusters_vs_mmax.py

$(SFE_PDF): sfe_vs_time.py
	python sfe_vs_time.py

clean:
	rm -rf $(PLOTDIR) $(MCLUSTER_PDF) $(SFE_PDF)
