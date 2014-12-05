# -*- coding: utf-8 -*-
# Copyright 2014 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation

import itertools
import numpy as np

def feldmanInequalities(mat, fundamentalCone=False):
    """Compute the forbidden set inequalities for the binary control matrix *mat*. They are
    returned by means of two numpy.arrays *A* (dimension 2) and *b* (dimension 1) such
    that Ax <= b describes the constraints.

    If *fundamentalCone* is True, only the inequalities adjacent to the origin are generated.
    """
    A = np.empty((0,mat.shape[1]), dtype=np.int)
    b = np.empty(0, dtype = np.int)
    for row in mat:
        N_i = np.nonzero(row)[0]
        if fundamentalCone:
            maxS = 2
        else:
            maxS = len(N_i) + 1
        for s in range(1, maxS, 2):
            for subset in itertools.combinations(N_i, s):
                newRowA = np.zeros(mat.shape[1])
                for i in N_i:
                    newRowA[i] = -1
                for i in subset:
                    newRowA[i] = 1
                A = np.vstack((A, newRowA))
                b = np.hstack((b, [len(subset)-1]))

    return A,b


def boxInequalities(mat):
    A = np.zeros((2*mat.shape[1], mat.shape[1]), dtype = np.int)
    b = np.zeros(2*mat.shape[1], dtype = np.int)
    for i in range(mat.shape[1]):
        A[2*i][i] = -1
        A[2*i+1][i] = 1
        b[2*i+1] = 1
    return A,b