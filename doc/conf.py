# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
from importlib import metadata


# -- Project information -----------------------------------------------------

project = "vamb"
copyright = "2024, Jakob Nybo Nissen, Simon Rasmussen"  # ! please update
author = "Jakob Nybo Nissen, Simon Rasmussen"
PACKAGE_VERSION = metadata.version("vamb")
version = PACKAGE_VERSION
release = PACKAGE_VERSION


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autodoc.typehints",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx_new_tab_link",
    "myst_nb",
]

#  https://myst-nb.readthedocs.io/en/latest/computation/execute.html
nb_execution_mode = "auto"

myst_enable_extensions = ["dollarmath", "amsmath"]

# Plolty support through require javascript library
# https://myst-nb.readthedocs.io/en/latest/render/interactive.html#plotly
html_js_files = [
    "https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.4/require.min.js"
]

# https://myst-nb.readthedocs.io/en/latest/configuration.html
# Execution
nb_execution_raise_on_error = True
# Rendering
nb_merge_streams = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    ".npz",
]


# Intersphinx options
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable/index.html", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    # "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    # "scikit-learn": ("https://scikit-learn.org/stable/", None),
    # "matplotlib": ("https://matplotlib.org/stable/", None),
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
# See:
# https://github.com/executablebooks/MyST-NB/blob/master/docs/conf.py
# html_title = ""
html_theme = "sphinx_book_theme"
# html_logo = "_static/logo-wide.svg"
# html_favicon = "_static/logo-square.svg"
html_theme_options = {
    "github_url": "https://github.com/RasmussenLab/vamb",
    "repository_url": "https://github.com/RasmussenLab/vamb",
    "repository_branch": "main",
    "home_page_in_toc": True,
    "path_to_docs": "docs",
    "show_navbar_depth": 2,
    "use_edit_page_button": True,
    "use_repository_button": True,
    "use_download_button": True,
    "launch_buttons": {
        "colab_url": "https://colab.research.google.com"
        #     "binderhub_url": "https://mybinder.org",
        #     "notebook_interface": "jupyterlab",
    },
    "navigation_with_keys": False,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ["_static"]


# -- Setup for sphinx-apidoc -------------------------------------------------

# Read the Docs doesn't support running arbitrary commands like tox.
# sphinx-apidoc needs to be called manually if Sphinx is running there.
# https://github.com/readthedocs/readthedocs.org/issues/1139

if os.environ.get("READTHEDOCS") == "True":
    from pathlib import Path

    PROJECT_ROOT = Path(__file__).parent.parent
    PACKAGE_ROOT = PROJECT_ROOT / "vamb"

#     def run_apidoc(_):
#         from sphinx.ext import apidoc
#
#         apidoc.main(
#             [
#                 "--force",
#                 "--implicit-namespaces",
#                 "--module-first",
#                 "--separate",
#                 "-o",
#                 str(PROJECT_ROOT / "doc" / "reference"),
#                 str(PACKAGE_ROOT),
#                 str(PACKAGE_ROOT / "*.c"),
#                 str(PACKAGE_ROOT / "*.so"),
#             ]
#         )
#
#     def setup(app):
#         app.connect("builder-inited", run_apidoc)
