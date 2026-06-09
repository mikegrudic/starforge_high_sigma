# Manuscript build. latexmk drives pdflatex+bibtex enough passes to converge.
TEX := manuscript

# The two bib files are listed explicitly so editing either triggers a rebuild
# of the PDF (latexmk also tracks them, but make's dep-graph view is the one
# that gates this rule).
BIBS := master.bib manuscriptNotes.bib

# Tracked figure PDFs — editing a figure script regenerates the PDF, which
# bumps its mtime and triggers a manuscript rebuild. Wildcard captures what
# exists now; if you add a new figure PDF, re-run make once to pick it up.
FIGURE_PDFS := $(wildcard figures/imf_plots/*.pdf) figures/multipanel.pdf figures/Mcluster_vs_Mmax.pdf

.PHONY: all bib clean view figures

all: $(TEX).pdf

# --pdf builds via pdflatex (not lualatex/xelatex); --halt-on-error stops
# at the first compile error rather than dumping pages of garbage.
$(TEX).pdf: $(TEX).tex $(BIBS) $(FIGURE_PDFS) | figures
	latexmk -pdf -halt-on-error -interaction=nonstopmode $(TEX)

# Force a bibtex pass — useful when only bibliography entries changed and
# pdflatex doesn't notice (it normally sees the .bbl and uses the cached one).
bib:
	pdflatex -interaction=nonstopmode $(TEX) > /dev/null
	bibtex $(TEX)
	pdflatex -interaction=nonstopmode $(TEX) > /dev/null
	pdflatex -interaction=nonstopmode $(TEX) > /dev/null

# Build all figure-side PDFs (delegated to figures/Makefile). Declared as an
# order-only dep of $(TEX).pdf above, so the manuscript build does not
# re-run on figure mtimes — but if you've never built the figures, this
# generates them first so \includegraphics doesn't fail.
figures:
	$(MAKE) -C figures all

# Remove everything latexmk knows about, including manuscript.pdf.
clean:
	latexmk -C $(TEX)

# macOS convenience: open the built PDF in Preview.
view: $(TEX).pdf
	open $(TEX).pdf
