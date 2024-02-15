# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
sys.path.insert(0, os.path.abspath("../../src"))

project = 'madina'
copyright = '2024, City Form Lab'
author = 'City Form Lab'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'nbsphinx'
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']



nbsphinx_allow_errors = True

## COnstants used inside the documentation:
#rst_epilog = """
#.. |notebook_folder_download|: `Download example notebooks <https://www.dropbox.com/scl/fo/vvhukdl6vc2wcprzp9kwc/h?rlkey=3zteo0dj08d5mhbeyo95v8qd2&dl=1>`_
#"""



