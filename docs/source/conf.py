import datetime
import re
from os import environ

from pystencils import __version__ as pystencils_version

from sphinx.util import logging
logger = logging.getLogger(__name__)

project = "pystencils"
html_title = "pystencils Documentation"

copyright = (
    f"{datetime.datetime.now().year}, Martin Bauer, Markus Holzer, Frederik Hennig"
)
author = "Martin Bauer, Markus Holzer, Frederik Hennig"

release = re.search(r"(([ab]?[0-9]+\.?)+)(\+)?", pystencils_version).groups()[0]

announcement: str | None
if pystencils_version != release:
    announcement = (
        f"This is the documentation for the development revision {pystencils_version} of pystencils. "
        f"<a style='color: inherit' href='https://pycodegen.pages.i10git.cs.fau.de/docs/pystencils/{release}/'>"
        "View the latest release instead</a>"
    )
else:
    announcement = None

#   Populate version switcher with versions for which documentation is published
#   In the CI, `PYSTENCILS_DOC_VERSIONS` is defined as a group CI variable in the `pycodegen` group
#   See README of `pycodegen/pycodegen.pages.i10git.cs.fau.de` repo

doc_current_version = environ.get("PYSTENCILS_DOC_CURRENT_VERSION", release)
doc_versions = ["master"] + environ.get("PYSTENCILS_DOC_VERSIONS", "").split()

logger.info(f"Versions in version switcher: {doc_versions}")
logger.info(f"Current version: {doc_current_version}")

language = "en"
default_role = "any"
pygments_style = "sphinx"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx_design",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.inheritance_diagram",
    "sphinxcontrib.bibtex",
    "sphinx_autodoc_typehints",
    "myst_nb",
]

templates_path = ["_templates"]
exclude_patterns = []

autodoc_member_order = "bysource"
autodoc_typehints = "description"

numfig = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3.8", None),
    "numpy": ("https://docs.scipy.org/doc/numpy/", None),
    "matplotlib": ("https://matplotlib.org/", None),
    "sympy": ("https://docs.sympy.org/latest/", None),
    "cupy": ("https://docs.cupy.dev/en/stable/", None),
    "dpctl": ("https://intelpython.github.io/dpctl/0.21.1/", None),
    "sycl": ("https://github.khronos.org/SYCL_Reference/", None),
    "graphviz": ("https://graphviz.readthedocs.io/en/stable/", None),
}

# -- Options for inheritance diagrams-----------------------------------------

inheritance_graph_attrs = {
    "bgcolor": "white",
}

# -- Options for MyST / MyST-NB ----------------------------------------------

nb_execution_mode = "off"  # do not execute notebooks by default

myst_enable_extensions = [
    "dollarmath",
    "colon_fence",
]

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_css_files = [
    "css/fixtables.css",
]
html_theme_options = {
    "logo": {
        "image_light": "_static/img/pystencils-logo-light.svg",
        "image_dark": "_static/img/pystencils-logo-dark.svg",
    }
}

if announcement:
    html_theme_options["announcement"] = announcement

html_sidebars = {
    "**": [
        "navbar-logo.html",
        "icon-links.html",
        "version_switcher.html",
        "search-button-field.html",
        "sbt-sidebar-nav.html",
    ]
}

html_context = {
    "doc_current_version": doc_current_version,
    "doc_versions": doc_versions
}

# NbSphinx configuration

nbsphinx_execute = "never"
nbsphinx_codecell_lexer = "python3"

#   BibTex
bibtex_bibfiles = ["pystencils.bib"]
