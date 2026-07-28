# docs/source/conf.py

import os
import sys

sys.path.insert(0, os.path.abspath("../../src"))

# -- Project information -----------------------------------------------------
project = "MalthusJAX"
copyright = "2026, Leonardo Di Caterina"
author = "Leonardo Di Caterina"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "numpydoc",
    "sphinx_copybutton",
    "myst_parser",
]

# Mock optional / heavy C-extension dependencies to avoid slow imports
# and allow building docs without them installed.
autodoc_mock_imports = [
    "jax",
    "jaxlib",
    "flax",
    "chex",
    "optax",
    "numpy",
    "sklearn",
    "scipy",
    "pandas",
    "evosax",
    "qdax",
    "tensorneat",
    "gymnax",
    "brax",
    "jumanji",
    "bbobax",
    "kozax",
    "pytest",
]

# -- Napoleon settings -------------------------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True

# -- NumPy docstring settings ------------------------------------------------
numpydoc_show_class_members = False

# -- Autodoc settings --------------------------------------------------------
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
}

# -- sphinx-autodoc-typehints settings (renders mypy type annotations) -------
typehints_fully_qualified = False
always_document_param_types = True
typehints_document_rtype = True
always_use_bars_union = True

# Suppress duplicate object description warnings from re-exported symbols
suppress_warnings = ["autodoc.import_object", "ref.duplicate"]

# -- MyST settings -----------------------------------------------------------
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# -- HTML output -------------------------------------------------------------
html_theme = "furo"
html_static_path = []

# -- Intersphinx mappings ----------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "flax": ("https://flax.readthedocs.io/en/latest/", None),
}
