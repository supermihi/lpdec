# -*- coding: utf-8 -*-
# Copyright 2014-2015 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation

"""This module contains classes and helper functions for non-binary codes. Currently,
the contents are not well integrated with the rest of the lpdec package.
"""

from __future__ import division, unicode_literals, print_function
import os.path
from collections import OrderedDict
import numpy as np
from lpdec.codes import LinearBlockCode
from lpdec import matrices, utils, gfqla


class NonbinaryLinearBlockCode(LinearBlockCode):
    """Base class for non-binary linear block codes over :math:`\mathbb F_q`.

    `name` is a string uniquely describing the code. When storing results into a database, there
    must not be two different codes with the same name. If the parity-check matrix is given by a
    file, the code's name defaults to the name of that file.

    The class can be instanciated directly by providing a parity check matrix; the argument
    `parityCheckMatrix` must be either the path of a file containig the matrix, or
    a two-dimensional list or a :class:`np.ndarray` representation of the matrix.
    Subclasses using a different code representation should leave the default value of ``None``.
    """

    def __init__(self, name=None, parityCheckMatrix=None, q=None):
        if parityCheckMatrix is not None:
            if utils.isStr(parityCheckMatrix):
                self.filename = os.path.expanduser(parityCheckMatrix)
                self._parityCheckMatrix = matrices.getNonbinaryMatrix(self.filename)
                if name is None:
                    name = os.path.basename(self.filename)
            elif not isinstance(parityCheckMatrix, np.ndarray):
                    self._parityCheckMatrix = matrices.getNonbinaryMatrix(parityCheckMatrix)
            else:
                self._parityCheckMatrix = parityCheckMatrix
            if q is None:
                q = int(np.max(self.parityCheckMatrix) + 1)  # avoid having q stored as numpy int
            self.blocklength = self.parityCheckMatrix.shape[1]
            rank = gfqla.rank(self.parityCheckMatrix, q)
            self.infolength = self.blocklength - rank
            cols = np.hstack((np.arange(self.infolength, self.blocklength),
                                  np.arange(self.infolength)))
            self._generatorMatrix = gfqla.orthogonalComplement(self.parityCheckMatrix,
                                                               columns=cols, q=q)
        LinearBlockCode.__init__(self, q, name)


def flanaganEmbedding(vector, q):
    """Return the binary "Flanagan" embedding :cite:`Flanagan+09NonBinary` of a q-ary vector
    :math:`\in \mathbb F_q^n` into :math:`\mathbb F_2^{(q-1)\\times n}` using the map::

        0   -> 0 ... 0
        1   -> 1 ... 0
        2   -> 0 1...0
        ..
        q-1 -> 0 0.. 1
    """
    vector = np.asarray(vector)
    out = np.zeros((q-1) * vector.size, dtype=np.int)
    for i, val in enumerate(vector):
        val = val % q
        if val != 0:
            out[i*(q-1)+val-1] = 1
    return out


def reverseEmbedding(vector, q):
    ret = np.zeros(vector.size//(q-1), np.int)
    for i in np.flatnonzero(vector):
        ret[i//q] = 1 + i % q
    return ret


if __name__ == '__main__':
    H = [[1,1,1, 1,1]]
    code = NonbinaryLinearBlockCode(parityCheckMatrix=H, q=2, name='NBtestCode')
    codewords = [flanaganEmbedding(cw, code.q) for cw in code.allCodewords()]
    from lpdec.polytopes import convexHull
    print(codewords)
    print(len(codewords))
    print(code.generatorMatrix)
    A,b  = convexHull(codewords)
    print(matrices.formatMatrix(A))
    print(matrices.formatMatrix(b))
    print(A.shape, b.reshape((b.size, 1)).shape)
    print(matrices.formatMatrix(np.hstack((A,b.reshape((b.size, 1)))), width=3))