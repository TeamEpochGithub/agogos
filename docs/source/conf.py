# ruff: noqa: INP001

"""Sphinx configuration file."""

import sys
from pathlib import Path

project = "Agogos"
copyright = "2024, Jasper van Selm"
author = "Jasper van Selm"

# Add root path of repository
sys.path.insert(0, Path("../..").resolve().as_posix())

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.autosummary"]
autosummary_generate = True

templates_path = ["_templates"]
# exclude_patterns = []

# Source suffix
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinxawesome_theme"
html_static_path = ["_static"]
