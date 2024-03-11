# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

from pathlib import Path
import sys

project = "Agogos"
copyright = "2024, Jasper van Selm"
author = "Jasper van Selm"

# Add root path of repository
sys.path.insert(0, Path("..").resolve().as_posix())
sys.path.insert(0, Path("../..").resolve().as_posix())
print(sys.path)

# try:
#     # import agogos

# except Exception as e:
#     print("FAUl")

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

html_theme = "alabaster"
html_static_path = ["_static"]
