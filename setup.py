#!/usr/bin/python2
# -*- coding: utf-8 -*-
# Copyright 2014-2015 Michael Helmling
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
from os.path import join

from setuptools import setup, find_packages
from Cython.Build import cythonize
import numpy as np


requirements = ['numpy', 'sqlalchemy', 'cython', 'python-dateutil', 'jinja2', 'sympy', 'scipy']


def makeExtensions():
    """Returns an Extension object for the given submodule of lpdecoding."""

    sources = []
    for root, dirnames, filenames in os.walk('lpdec'):
        for filename in fnmatch.filter(filenames, '*.pyx'):
            sources.append(str(join(root, filename)))
    directives = dict(embedsignature=True)
    if '--profile' in sys.argv:
        directives['profile'] = True
        sys.argv.remove('--profile')
    if '--debug' in sys.argv:
        sys.argv.remove('--debug')
    else:
        directives['boundscheck'] = False
        directives['nonecheck'] = False
        directives['initializedcheck'] = False
    extensions = cythonize(sources, include_path=[np.get_include()],
                           compiler_directives=directives)
    for e in extensions:
        e.include_dirs += [np.get_include()] # the above does not work on windows
    if '--no-glpk' in sys.argv:
        extensions = [e for e in extensions if 'glpk' not in e.libraries]
        sys.argv.remove('--no-glpk')
    if '--no-gurobi' in sys.argv:
        extensions = [e for e in extensions if 'gurobi65' not in e.libraries]
        sys.argv.remove('--no-gurobi')
    else:
        requirements.append('gurobimh')
        # find library version: library name includes major/minor version information (e.g.
        # libgurobi65.so vs libgurobi60.so). This hack-ish solution parses version information from
        # the C header file.
        try:
            gurobihome = os.environ['GUROBI_HOME']
        except KeyError:
            raise RuntimeError('GUROBI_HOME not set')
        with open(join(gurobihome, 'include', 'gurobi_c.h'), 'rt') as f:
            gurobi_c_h = f.read()
        major = re.findall('define GRB_VERSION_MAJOR\s+([0-9]+)', gurobi_c_h)[0]
        minor = re.findall('define GRB_VERSION_MINOR\s+([0-9]+)', gurobi_c_h)[0]
        libraryName = 'gurobi' + major + minor
        for e in extensions:
            if 'gurobi65' in e.libraries:
                e.libraries[e.libraries.index('gurobi65')] = libraryName
                e.library_dirs = [join(gurobihome, 'lib')]
                e.include_dirs = [join(gurobihome, 'include')]
    return extensions


with io.open(join('lpdec', '__init__.py'), 'r', encoding='UTF-8') as f:
    version_file = f.read()
    version_match = re.search(r'^__version__ = [\'"]([^\'"]*)[\'"]', version_file, re.M)
    version = version_match.group(1)


scriptName = 'lpdec'

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
    install_requires=requirements,
    include_package_data=True,
    ext_modules=makeExtensions(),
    packages=find_packages(exclude=['test']),
    entry_points={'console_scripts': ['{} = lpdec.cli:script'.format(scriptName),]},
    test_suite='test',
)
