# -*- coding: utf-8 -*-
# Copyright 2014 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation

"""This module contains various functions related to the polyhedral structure of a code."""

import os
import tempfile
import shutil
import subprocess
import numpy as np


class Polytope:
    def __init__(self, vertices):
        self.vertices = vertices
        self._facets = None

    @property
    def facets(self):
        if not self._facets:
            A, b = convexHull(self.vertices)
            self._facets = list(zip(A, b))
        return self._facets

    def adjacentVertices(self, a, b):
        for v in self.vertices:
            if np.allclose(np.dot(a, v), b):
                yield v

    def adjacentFacets(self, v):
        for a, b in self.facets:
            if np.allclose(np.dot(a, v), b):
                yield (a, b)

    def __contains__(self, item):
        for a, b in self.facets:
            if np.dot(a, item) > b:
                return False
        return True


def convexHull(points):
    """Use lrs to compute the convex hull (by means of facet-defining inequalities) of the given
    set of points.

    :returns: Tuple (A,b) such that :math:`Ax \leq b` defines the convex hull.
    """
    if not hasattr(points, '__len__'):
        points = list(points)
    dim = len(points[0])
    tmpdir = tempfile.mkdtemp()
    try:
        tmpExt = os.path.join(tmpdir, 'poly.ext')
        with open(tmpExt, 'wt') as tmp:
            tmp.write('points\nV-representation\nbegin\n')
            tmp.write('{} {} rational\n'.format(len(points), dim + 1))
            for point in points:
                tmp.write('1 ' + ' '.join(str(v) for v in point) + '\n')
            tmp.write('end\n')

        with open(os.devnull, 'w') as devnull:
            subprocess.check_call(['cddf+', tmpExt], stdout=devnull, stderr=subprocess.STDOUT)
        tmpIne = tmpExt[:-3] + 'ine'
        with open(tmpIne, 'rt') as ine:
            hrep = ine.read().splitlines()
        if hrep[-1].startswith('*Input Error'):
            raise RuntimeError(hrep[-1])
        first = hrep.index('H-representation')
        hrep = hrep[first + 2:-1]
        rows, cols = hrep[0].split()[:2]
        rows = int(rows)
        cols = int(cols) - 1
        A = np.zeros((rows, cols), dtype=np.int)
        b = np.zeros((rows, 1), dtype=np.int)
        for i, row in enumerate(hrep[1:]):
            brow, Arow = row.strip().split(None, 1)
            b[i] = int(brow)
            for j, Aentry in enumerate(Arow.split()):
                A[i, j] = -int(Aentry)
        return A, b
    finally:
        shutil.rmtree(tmpdir)