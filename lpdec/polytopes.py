# -*- coding: utf-8 -*-
# Copyright 2014-2015 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation

"""This module contains various functions related to the polyhedral structure of a code."""

import os
import tempfile
import shutil
import subprocess
import itertools
from fractions import Fraction
import numpy as np


def feldmanInequalities(hmat, fundamentalCone=False):
    """Compute the forbidden set inequalities for the binary control matrix *matrix*. They are
    returned by means of two *A* (dimension 2) and *b* (dimension 1) such
    that Ax <= b describes the constraints.

    If *fundamentalCone* is True, only the inequalities adjacent to the origin are generated.
    """
    if fundamentalCone:
        numConstraints = np.sum(hmat)
    else:
        numConstraints = 0
        for s in hmat.sum(1):
            numConstraints += 2 ** (s - 1)
    A = np.zeros((numConstraints, hmat.shape[1]), dtype=np.int)
    b = np.empty((numConstraints,), dtype=np.int)
    k = 0
    for row in hmat:
        N_j = np.flatnonzero(row)
        if fundamentalCone:
            maxS = 2
        else:
            maxS = len(N_j) + 1
        for s in range(1, maxS, 2):
            for subset in itertools.combinations(N_j, s):
                newRowA = np.zeros(hmat.shape[1])
                for i in N_j:
                    A[k, i] = -1
                for i in subset:
                    A[k, i] = 1
                b[k] = len(subset) - 1
                k += 1
    return A, b


def boxInequalities(mat):
    A = np.zeros((2*mat.shape[1], mat.shape[1]), dtype = np.int)
    b = np.zeros(2*mat.shape[1], dtype = np.int)
    for i in range(mat.shape[1]):
        A[2*i][i] = -1
        A[2*i+1][i] = 1
        b[2*i+1] = 1
    return A,b


class Polytope:
    def __init__(self, vertices):
        self.vertices = vertices
        self._facets = None


    def computeFacets(self, program='cddf+'):
        A, b = convexHull(self.vertices, program)
        self._facets = list(zip(A, b))
        return self._facets

    @property
    def facets(self):
        if not self._facets:
            self.computeFacets()
        return self._facets

    def adjacentVertices(self, a, b):
        for v in self.vertices:
            if np.allclose(np.dot(a, v), b):
                yield v

    def adjacentFacets(self, v):
        for a, b in self.facets:
            if np.allclose(np.dot(a, v), b):
                yield (a, b)

    def violatedFacets(self, v, tolerance=0):
        for a, b in self.facets:
            if np.dot(a, v) > b + tolerance:
                yield a, b

    def __contains__(self, item):
        for a, b in self.facets:
            if np.dot(a, item) > b:
                return False
        return True

    @staticmethod
    def isFeldmanType(a, b):
        """Return True iff a^Tx <= b represents a Feldman-type or box constraint inequality."""
        nz = np.flatnonzero(a)
        # test box
        if b == 1 and len(nz) == 1 and np.sum(a) == 1:
            # x_i <= 1
            return True
        if b == 0 and len(nz) == 1 and np.sum(a) == -1:
            # x_i >= 0
            return True
        if np.sum(a == 1) + np.sum(a == -1) != len(nz):
            return False
        return b == np.sum(a == 1) - 1


def convexHull(points, program='cddf+'):
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
            subprocess.check_call([program, tmpExt], stdout=devnull, stderr=subprocess.STDOUT)
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
        b = np.zeros((rows), dtype=np.int)

        for i, row in enumerate(hrep[1:]):
            aTmp = []
            brow, Arow = row.strip().split(None, 1)
            bTmp = Fraction(brow)
            #b[i] = int(brow)
            for j, Aentry in enumerate(Arow.split()):
                #A[i, j] = -int(Aentry)
                aTmp.append(-Fraction(Aentry))
            kgv = (bTmp + sum(aTmp)).denominator
            b[i] = int(kgv*bTmp)
            for j, a in enumerate(aTmp):
                A[i, j] = int(kgv*a)

        return A, b
    finally:
        shutil.rmtree(tmpdir)