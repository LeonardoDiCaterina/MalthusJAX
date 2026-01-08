# Makefile for MalthusJAX
#
# This file provides convenient aliases for common development tasks.
# All configurations are stored in `pyproject.toml`.

# Use .PHONY to ensure these commands run even if files with these names exist
.PHONY: help install-dev test lint format type-check check-all docs

# Default target: show help
help:
	@echo "--- MalthusJAX Development ---"
	@echo "  make install-dev    Install for development"
	@echo "  make test           Run pytest with coverage"
	@echo "  make lint           Check for linting errors with Ruff"
	@echo "  make format         Format code with Ruff"
	@echo "  make type-check     Run mypy type checker"
	@echo "  make check-all      Run all checks (lint, format, types, test)"
	@echo "  make docs           Build the Sphinx documentation"

install-dev:
	@echo "--- Installing development dependencies ---"
	pip install -e ".[dev,docs,examples]"

test:
    @echo "--- Running tests with coverage ---"
    python -m pytest

lint:
	@echo "--- Checking code quality with Ruff ---"
	ruff check .

format:
	@echo "--- Formatting code with Ruff ---"
	ruff format .

type-check:
	@echo "--- Running mypy type checker ---"
	mypy src

check-all: lint format type-check test
	@echo "--- All checks passed! ---"

docs:
	@echo "--- Building documentation ---"
	sphinx-build -b html docs/source docs/build/html
	# sphinx-build: This is the main Sphinx command
	# -b html: This says "use the HTML builder"
	# docs/source: This is the source directory for the docs
	# docs/build/html: This is where the built HTML files will go