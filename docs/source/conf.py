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
    "sphinx_copybutton",
    "myst_parser",
]

# -- Napoleon settings -------------------------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_use_ivar = True

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


def skip_member(app, what, name, obj, skip, options):
    if skip:
        return True

    # Check if the object is imported from another module
    obj_module = getattr(obj, "__module__", None)
    if obj_module:
        current_module = app.env.ref_context.get("py:module")
        if current_module and obj_module != current_module:
            return True

    return False


def setup(app):
    app.connect("autodoc-skip-member", skip_member)
