# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys

print(os.path.abspath("../suppy"))
sys.path.insert(0, os.path.abspath("../suppy"))  # Adjust to your source folder

project = "suppy"
copyright = "2024, Tobias Becher"
author = "Tobias Becher"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "numpydoc",
    "sphinx_autodoc_typehints",
    "sphinx_design",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

add_module_names = False

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]

autodoc_default_options = {
    "no-module": True,  # Suppress the module/package labels
    "members": True,  # Include members
    "inherited-members": True,  # Include inherited members
}
autodoc_type_aliases = {
    "npt.ArrayLike": "npt.ArrayLike",
}

autodoc_inherit_docstrings = True

numpydoc_class_members_toctree = False
# autosummary_generate = False

coverage_modules = [
    "suppy.feasibility",
    "suppy.feasibility._bands",
    "suppy.feasibility._halfspaces",
    "suppy.feasibility._hyperplanes",
    "suppy.projections",
    "suppy.superiorization",
    "suppy.perturbations",
    "suppy.algorithms",
    "suppy.utils",
]
