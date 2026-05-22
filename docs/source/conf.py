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
import sys

sys.path.insert(0, os.path.abspath("../../"))

# -- Project information -----------------------------------------------------

project = "hawk-eye"
copyright = "2021, UAV Austin"
author = "UAV Austin"

docs_repo = "hawk-eye"
docs_user = "uavaustin"

releases_github_path = f"{docs_user}/{docs_repo}"


# The full version, including alpha/beta/rc tags
release = "0.0.2b0"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosectionlabel",
]
autoclass_content = "both"
autodoc_mock_imports = [
    "albumentations",
    "cv2",
    "google",
    "numpy",
    "PIL",
    "pycocotools",
    "requests",
    "torch",
    "torchvision",
    "tqdm",
]

# build the templated autosummary files
autosummary_generate = True
numpydoc_show_class_members = False
todo_include_todos = True

# autosectionlabel throws warnings if section names are duplicated.
# The following tells autosectionlabel to not throw a warning for
# duplicated section names that are in different documents.
autosectionlabel_prefix_document = True

napoleon_use_ivar = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

source_suffix = {
    ".md": "markdown",
    ".rst": "restructuredtext",
}
# The master toctree document.
master_doc = "index"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"
html_theme_options = {}


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ["_static"]

html_context = {
    "display_github": True,
    "source_url_prefix": (
        f"https://github.com/{docs_user}/{docs_repo}/tree/master/docs"
    ),
    "github_host": "github.com",
    "github_user": docs_user,
    "github_repo": docs_repo,
    "github_version": "master",
    "conf_py_path": "/docs/",
    "source_suffix": ".md",
}
