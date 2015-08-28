# -*- coding: utf-8 -*-

import sys
import os

sys.path.insert(0, os.path.abspath('..'))

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'alabaster',
    'sphinxcontrib.bibtex',
]
intersphinx_mapping = {
    'python': ('http://docs.python.org/3.4', None),
}

templates_path = ['_templates']

source_suffix = '.rst'

source_encoding = 'utf-8-sig'

master_doc = 'index'

project = 'lpdec'
copyright = '2015, Michael Helmling'
version = '2015.1'
release = '2015.1'
exclude_patterns = ['_build']
pygments_style = 'sphinx'

html_theme = 'alabaster'
html_theme_options = {
    'description': "LP decoding package",
    'github_user': 'supermihi',
    'github_repo': 'lpdec',
}
html_sidebars = {
    '**': [
        'about.html',
        'navigation.html',
        'searchbox.html',
        'donate.html',
    ]
}

htmlhelp_basename = 'lpdecdoc'
autodoc_default_flags = ['members', 'show-inheritance']
autodoc_member_order = 'bysource'
numpydoc_show_class_members = False
add_module_names = False
napoleon_numpy_docstring = True
napoleon_use_ivar = True