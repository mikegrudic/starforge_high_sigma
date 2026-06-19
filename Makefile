# Manuscript build. Uses pdflatex+bibtex directly (latexmk 4.83 crashes on
# null bytes it generates from mkii/ paths in the pdflatex log).
TEX := manuscript

BIBS := master.bib manuscriptNotes.bib

FIGURE_PDFS := $(wildcard figures/imf_plots/*.pdf) figures/multipanel.pdf figures/Mcluster_vs_Mmax.pdf figures/wind_mdot.pdf

RUN_TABLE := tables/run_table.tex

.PHONY: all bib clean rebuild view figures

all: $(TEX).pdf

$(TEX).pdf: $(TEX).tex $(BIBS) $(FIGURE_PDFS) $(RUN_TABLE) | figures
	pdflatex -interaction=nonstopmode $(TEX)
	bibtex $(TEX)
	pdflatex -interaction=nonstopmode $(TEX)
	pdflatex -interaction=nonstopmode $(TEX)

bib:
	pdflatex -interaction=nonstopmode $(TEX) > /dev/null
	bibtex $(TEX)
	pdflatex -interaction=nonstopmode $(TEX) > /dev/null
	pdflatex -interaction=nonstopmode $(TEX) > /dev/null

$(RUN_TABLE): tables/run_table.py
	python tables/run_table.py

figures:
	$(MAKE) -C figures all

figures/%.pdf:
	$(MAKE) -C figures $(notdir $@)

figures/imf_plots/%.pdf:
	$(MAKE) -C figures imf_plots/$(notdir $@)

clean:
	rm -f $(TEX).aux $(TEX).log $(TEX).pdf $(TEX).bbl $(TEX).blg \
	      $(TEX).out $(TEX).toc $(TEX).fls $(TEX).fdb_latexmk \
	      $(TEX).bcf $(TEX).run.xml equations/*.aux tables/*.aux

rebuild: clean all

view: $(TEX).pdf
	open $(TEX).pdf
