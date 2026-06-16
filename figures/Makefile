PLOTDIR := imf_plots
MODEL := chabrier_smooth_bounds

COMPARE_PDF        := $(PLOTDIR)/comparison_$(MODEL).pdf
COMPARE_NORM_PDF   := $(PLOTDIR)/comparison_$(MODEL)_normalized.pdf
ALPHA_PDF          := $(PLOTDIR)/alphaplot_$(MODEL).pdf
ALPHA_OBS_PDF      := $(PLOTDIR)/alphaplot_obs_$(MODEL).pdf
ALPHA_OBS_Z_PDF    := $(PLOTDIR)/alphaplot_obs_zcolor_$(MODEL).pdf
LOGNORMAL_PDF      := $(PLOTDIR)/lognormal.pdf
MCLUSTER_PDF       := Mcluster_vs_Mmax.pdf
SFE_PDF            := sfe_vs_time.pdf
SFE_SIGMA_PDF      := sfe_vs_sigma.pdf
VMS_ACCRETION_PDF  := VMS_Accretion_History.pdf
VMS_MDOT_PDF       := VMS_M_vs_Mdot.pdf
VMS_MDOT_M23_PDF   := VMS_M_vs_Mdot_M23.pdf
VMS_DIST_PDF       := VMS_Distance_from_COM.pdf
WIND_MDOT_PDF      := wind_mdot.pdf

# Per-star data extracted by extract_star_masses.py; backs the VMS plots.
VMS_STAR_DATA := imf_data/STARFORGE_RT/STARFORGE_v1.2/M2e4_R1/M2e4_R1_Z0.01_S0_A2_B0.1_I10000_Res271_n2_sol0.5_42/output_turbsphere_driving1/star_masses.npz

ALL_PLOTS := $(COMPARE_PDF) $(COMPARE_NORM_PDF) $(ALPHA_PDF) $(ALPHA_OBS_PDF) $(ALPHA_OBS_Z_PDF) $(LOGNORMAL_PDF) $(MULTIPANEL_PDF) $(MCLUSTER_PDF) $(VMS_ACCRETION_PDF) $(VMS_MDOT_PDF) $(VMS_DIST_PDF) $(VMS_MDOT_M23_PDF) $(SFE_PDF) $(SFE_SIGMA_PDF) $(WIND_MDOT_PDF)


.PHONY: all clean compare_imfs alpha alpha-obs lognormal mcluster vms_accretion vms_mdot_m23 sfe sfe_sigma wind_mdot
all: $(ALL_PLOTS)

compare_imfs: $(COMPARE_PDF) $(COMPARE_NORM_PDF)
alpha: $(ALPHA_PDF)
alpha-obs: $(ALPHA_OBS_PDF)
lognormal: $(LOGNORMAL_PDF)
mcluster: $(MCLUSTER_PDF)
sfe: $(SFE_PDF)
sfe_sigma: $(SFE_SIGMA_PDF)
vms_accretion: $(VMS_ACCRETION_PDF) $(VMS_MDOT_PDF)
vms_mdot_m23: $(VMS_MDOT_M23_PDF)
wind_mdot: $(WIND_MDOT_PDF)

# imf_plots.py writes both the count and normalized comparison PDFs in one run;
# we list both targets to a common recipe so either being missing triggers it.
$(COMPARE_PDF) $(COMPARE_NORM_PDF): imf_plots.py
	python imf_plots.py

$(ALPHA_PDF) $(ALPHA_OBS_PDF) $(ALPHA_OBS_Z_PDF): imf_alphaplot.py
	python imf_alphaplot.py

$(LOGNORMAL_PDF): lognormal.py
	python lognormal.py

$(MCLUSTER_PDF): mclusters_vs_mmax.py
	python mclusters_vs_mmax.py

$(SFE_PDF): sfe_vs_time.py
	python sfe_vs_time.py

$(SFE_SIGMA_PDF): sfe_vs_sigma.py
	python sfe_vs_sigma.py

clean:
	rm -rf $(PLOTDIR) $(MCLUSTER_PDF) $(SFE_PDF) $(SFE_SIGMA_PDF)

# accretion_histories.py writes VMS_Accretion_History.pdf,
# VMS_Distance_from_COM.pdf, and VMS_M_vs_Mdot.pdf in one run; list all three
# targets so the recipe re-runs if any is missing.
$(VMS_ACCRETION_PDF) $(VMS_MDOT_PDF) $(VMS_DIST_PDF): accretion_histories.py vms_colors.py $(VMS_STAR_DATA)
	python accretion_histories.py

$(VMS_MDOT_M23_PDF): mdot_vs_m_23.py vms_colors.py $(VMS_STAR_DATA)
	python mdot_vs_m_23.py

$(WIND_MDOT_PDF): wind_mdot.py sim_paths.py
	python wind_mdot.py

clean:
	rm -rf $(PLOTDIR) $(MULTIPANEL_PDF) $(MCLUSTER_PDF) $(VMS_ACCRETION_PDF) $(VMS_MDOT_PDF) $(VMS_DIST_PDF) $(VMS_MDOT_M23_PDF)
