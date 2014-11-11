#!/usr/bin/python2
# -*- coding: utf-8 -*-
# Copyright 2014 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation
from __future__ import unicode_literals
import os
import fnmatch
import io
import re
import sys
from os.path import join, dirname, abspath

from setuptools import setup, find_packages
from Cython.Build import cythonize
import numpy as np


here = abspath(dirname(__file__))


def makeExtensions():
    """Returns an Extension object for the given submodule of lpdecoding."""

    sources = []
    for root, dirnames, filenames in os.walk(join(here, 'lpdec')):
        for filename in fnmatch.filter(filenames, '*.pyx'):
            sources.append(str(join(root, filename)))
    extensions = cythonize(sources, include_path=[np.get_include()])
    if '--no-glpk' in sys.argv:
        extensions = [e for e in extensions if 'glpk' not in e.libraries]
        sys.argv.remove('--no-glpk')
    return extensions


with io.open(os.path.join(here, 'lpdec', '__init__.py'), 'r', encoding='UTF-8') as f:
    version_file = f.read()
    version_match = re.search(r'^__version__ = [\'"]([^\'"]*)[\'"]', version_file, re.M)
    version = version_match.group(1)


scriptName = 'lpdec3' if sys.version_info.major == 3 else 'lpdec'

setup(
    name='lpdec',
    version=version,
    author='Michael Helmling',
    author_email='helmling@uni-koblenz.de',
    url='https://github.com/supermihi/lpdec',
    classifiers=[
      'Development Status :: 3 - Alpha',
      'Intended Audience :: Science/Research',
      'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
      'Operating System :: OS Independent',
      'Programming Language :: Python :: 2.7',
      'Programming Language :: Python :: 3',
      'Topic :: Scientific/Engineering :: Mathematics',
    ],
    license='GPL3',
    install_requires=['numpy', 'sqlalchemy', 'cython', 'python-dateutil', 'jinja2'],
    include_package_data=True,
    ext_modules=makeExtensions(),
    packages=find_packages(exclude=['test']),
    entry_points={'console_scripts': ['{} = lpdec.cli:script'.format(scriptName),]},
    test_suite='test',
)
