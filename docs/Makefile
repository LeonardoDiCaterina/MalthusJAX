# docs/Makefile — Sphinx documentation build rules
# Usage: cd docs && make <target>
#   or from the repo root: make -C docs <target>

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SPHINXAPIDOC  ?= sphinx-apidoc
SOURCEDIR     = source
BUILDDIR      = build

# Paths relative to the repo root (used by apidoc)
SRCPKG        = ../src/malthusjax
APIDOC_OUT    = $(SOURCEDIR)/api

.PHONY: help clean html dirhtml singlehtml latex latexpdf epub linkcheck \
        doctest apidoc apidoc-force livehtml

# --------------------------------------------------------------------------- #
# Default target
# --------------------------------------------------------------------------- #
help:
	@echo ""
	@echo "MalthusJAX — Sphinx documentation targets"
	@echo "==========================================="
	@echo "  html          Build HTML docs (default output format)"
	@echo "  dirhtml       Build HTML docs in directories (one dir per page)"
	@echo "  singlehtml    Build HTML as a single large page"
	@echo "  latex         Build LaTeX sources"
	@echo "  latexpdf      Build LaTeX sources then compile to PDF"
	@echo "  epub          Build EPUB"
	@echo "  linkcheck     Check all external hyperlinks for integrity"
	@echo "  doctest       Run doctests embedded in the documentation"
	@echo "  apidoc        Re-generate API stubs from source (non-destructive)"
	@echo "  apidoc-force  Re-generate API stubs, overwriting existing files"
	@echo "  livehtml      Start autobuild server with live reload (requires sphinx-autobuild)"
	@echo "  clean         Remove the entire build/ directory"
	@echo ""
	@echo "Pass extra Sphinx options via SPHINXOPTS, e.g.:"
	@echo "  make html SPHINXOPTS=\"-W --keep-going\""
	@echo ""

# --------------------------------------------------------------------------- #
# Clean
# --------------------------------------------------------------------------- #
clean:
	@echo "--- Removing $(BUILDDIR)/ ---"
	rm -rf $(BUILDDIR)

# --------------------------------------------------------------------------- #
# API stubs (sphinx-apidoc)
# --------------------------------------------------------------------------- #
apidoc:
	@echo "--- Generating API stubs (skip existing) ---"
	$(SPHINXAPIDOC) \
		--module-first \
		--separate \
		--no-toc \
		-o $(APIDOC_OUT) \
		$(SRCPKG)

apidoc-force:
	@echo "--- Generating API stubs (overwrite) ---"
	$(SPHINXAPIDOC) \
		--module-first \
		--separate \
		--no-toc \
		--force \
		-o $(APIDOC_OUT) \
		$(SRCPKG)

# --------------------------------------------------------------------------- #
# HTML
# --------------------------------------------------------------------------- #
html:
	@echo "--- Building HTML documentation ---"
	$(SPHINXBUILD) -b html $(SPHINXOPTS) $(SOURCEDIR) $(BUILDDIR)/html
	@echo ""
	@echo "Build finished. Open $(BUILDDIR)/html/index.html to view it."

dirhtml:
	@echo "--- Building dirhtml documentation ---"
	$(SPHINXBUILD) -b dirhtml $(SPHINXOPTS) $(SOURCEDIR) $(BUILDDIR)/dirhtml
	@echo ""
	@echo "Build finished. Open $(BUILDDIR)/dirhtml/index.html to view it."

singlehtml:
	@echo "--- Building single-page HTML documentation ---"
	$(SPHINXBUILD) -b singlehtml $(SPHINXOPTS) $(SOURCEDIR) $(BUILDDIR)/singlehtml
	@echo ""
	@echo "Build finished. Open $(BUILDDIR)/singlehtml/index.html to view it."

# --------------------------------------------------------------------------- #
# LaTeX / PDF
# --------------------------------------------------------------------------- #
latex:
	@echo "--- Building LaTeX sources ---"
	$(SPHINXBUILD) -b latex $(SPHINXOPTS) $(SOURCEDIR) $(BUILDDIR)/latex
	@echo ""
	@echo "Build finished. Run \`make\` in $(BUILDDIR)/latex to compile the PDF."

latexpdf: latex
	@echo "--- Compiling PDF from LaTeX ---"
	$(MAKE) -C $(BUILDDIR)/latex all-pdf
	@echo ""
	@echo "Build finished. PDF is in $(BUILDDIR)/latex/."

# --------------------------------------------------------------------------- #
# EPUB
# --------------------------------------------------------------------------- #
epub:
	@echo "--- Building EPUB ---"
	$(SPHINXBUILD) -b epub $(SPHINXOPTS) $(SOURCEDIR) $(BUILDDIR)/epub
	@echo ""
	@echo "Build finished. EPUB is in $(BUILDDIR)/epub/."

# --------------------------------------------------------------------------- #
# Quality checks
# --------------------------------------------------------------------------- #
linkcheck:
	@echo "--- Checking external links ---"
	$(SPHINXBUILD) -b linkcheck $(SPHINXOPTS) $(SOURCEDIR) $(BUILDDIR)/linkcheck
	@echo ""
	@echo "Link check complete. Results in $(BUILDDIR)/linkcheck/."

doctest:
	@echo "--- Running embedded doctests ---"
	$(SPHINXBUILD) -b doctest $(SPHINXOPTS) $(SOURCEDIR) $(BUILDDIR)/doctest
	@echo ""
	@echo "Doctest results in $(BUILDDIR)/doctest/output.txt"

# --------------------------------------------------------------------------- #
# Live reload (requires: pip install sphinx-autobuild)
# --------------------------------------------------------------------------- #
livehtml:
	@echo "--- Starting live-reload build server ---"
	sphinx-autobuild \
		--open-browser \
		--port 8080 \
		$(SPHINXOPTS) \
		$(SOURCEDIR) \
		$(BUILDDIR)/html
