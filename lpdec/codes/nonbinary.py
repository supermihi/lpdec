# -*- coding: utf-8 -*-
# Copyright 2014 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation

"""This module contains classes and helper functions for non-binary codes. Currently,
the contents are not well integrated with the rest of the lpdec package.
"""

from __future__ import division, unicode_literals
import os.path
from collections import OrderedDict
import itertools
import numpy as np
from lpdec.persistence import JSONDecodable
from lpdec import matrices


def getNonbinaryMatrix(source):
    """Creates a non-binary matrix of type :class:`np.ndarray` from either a file or a
    two-dimensional python list.

    If `source` is a file path, the file must contain an explicit representation of the
    matrix (by means of whitespace-separated numbers).

    :rtype: :class:`np.ndarray` with dtype `np.int`.
    :returns: A numpy ndarray representation of the matrix.
    """
    if isinstance(source, basestring):
        with open(source, 'rt') as f:
            return [[int(x) for x in line.strip().split()]
                    for line in f.readlines()
                    if len(line.strip()) > 0]
    else:
        assert hasattr(source, '__iter__') and hasattr(source[0], '__iter__')
        import copy
        lines = copy.copy(source)
    return np.array(lines, dtype=np.int)


class NonbinaryLinearBlockCode(JSONDecodable):
    """Base class for non-binary linear block codes over GF(q).

    `name` is a string uniquely describing the code. When storing results into a database, there
    must not be two different codes with the same name. If the parity-check matrix is given by a
    file, the code's name defaults to the name of that file.

    The class can be instanciated directly by providing a parity check matrix; the argument
    `parityCheckMatrix` must be either the path of a file containig the matrix, or
    a two-dimensional list or a :class:`np.ndarray` representation of the matrix.
    Subclasses using a different code representation should leave the default value of ``None``.
    """

    def __init__(self, name=None, parityCheckMatrix=None, q=None):
        JSONDecodable.__init__(self)
        if parityCheckMatrix is not None:
            if isinstance(parityCheckMatrix, basestring):
                self.filename = os.path.expanduser(parityCheckMatrix)
                self.parityCheckMatrix = getNonbinaryMatrix(self.filename)
                if name is None:
                    name = os.path.basename(self.filename)
            elif not isinstance(parityCheckMatrix, np.ndarray):
                    self.parityCheckMatrix = getNonbinaryMatrix(parityCheckMatrix)
            else:
                self.parityCheckMatrix = parityCheckMatrix
            if q is None:
                q = np.max(self.parityCheckMatrix) + 1
            self.blocklength = self.parityCheckMatrix.shape[1]
            rank = gaussianElimination(q, self.parityCheckMatrix.copy(), diagonalize=False).size
            self.infolength = self.blocklength - rank
            self.rate = self.infolength / self.blocklength
            cols = np.hstack((np.arange(self.infolength, self.blocklength),
                                  np.arange(self.infolength)))
            self.generatorMatrix = orthogonalComplement(q, self.parityCheckMatrix, cols)
        self.q = q
        if name is None:
            raise ValueError("A code must have a name.")
        self.name = name

    def __contains__(self, item):
        """Check if the given word is a codeword of this code.
        """
        return np.all(self.parityCheckMatrix.dot(item) % self.q == 0)

    def encode(self, infoword):
        return np.dot(infoword, self.generatorMatrix) % self.q

    def allCodewords(self):
        for infoword in itertools.product(list(range(self.q)), repeat=self.infolength):
            yield self.encode(infoword)

    def params(self):
        matrix = self.parityCheckMatrix
        pcm = matrix.tolist()
        return OrderedDict([('parityCheckMatrix', pcm), ('name', self.name), ('q', self.q)])


def inv(a, p):
    """Computes the multiplicative inverse of a mod p."""
    t = 0
    newt = 1
    r = p
    newr = a
    while newr != 0:
        quotient = r // newr
        t, newt = newt, t - quotient * newt
        r, newr = newr, r - quotient * newr
    if r > 1:
        raise ValueError('{} not invertible'.format(a))
    return t if t >= 0 else t + p


def gaussianElimination(q, matrix, columns=None, diagonalize=True):
        """The Gaussian elimination algorithm in GF(2) arithmetics.

        When called on a `(k × n)` matrix, the algorithm performs Gaussian elimination,
        bringin the matrix to reduced row echelon form.

        `columns`, if given, is a sequence of column indices, giving the the order in which columns
        of the matrix are visited, and defaults to the canonical ordering.

        `diagonalize` specifies whether the tridiagonalized columns should also be made unit
        vectors, yielding a diagonal structure among that columns.

        Warning: this is an in-place method, modifying the original matrix!
        """
        nrows = matrix.shape[0]
        ncols = matrix.shape[1]
        curRow = 0
        colIndex = 0
        successfulCols = np.zeros(nrows, dtype=np.int)
        numSuccessfulCols = 0
        invs = [-1] + [inv(a, q) for a in range(1, q)]
        if columns is None:
            columns = np.arange(ncols)
        while True:
            if colIndex >= columns.shape[0]:
                break
            curCol = columns[colIndex]
            # search for a pivot row
            pivotRow = -1
            for row in range(curRow, nrows):
                val = matrix[row, curCol]
                if val != 0:
                    pivotRow = row
                    break
            if pivotRow == -1:
                # did not find a pivot row -> this column is linearly dependent of the previously
                # visited; continue with next column
                colIndex += 1
                continue
            if pivotRow > curRow:
                # need to swap rows
                matrix[[curRow, pivotRow], :] = matrix[[pivotRow, curRow], :]
            # do the actual pivoting
            matrix[curRow] = (matrix[curRow] * invs[matrix[curRow, curCol]]) % q
            for row in range(curRow + 1, nrows):
                val = matrix[row, curCol]
                if val != 0:
                    matrix[row] = (matrix[row] - val*matrix[curRow]) % q
            successfulCols[numSuccessfulCols] = curCol
            numSuccessfulCols += 1
            if numSuccessfulCols == nrows:
                break
            curRow += 1
            colIndex += 1
        if diagonalize:
            for colIndex in range(numSuccessfulCols):
                curCol = successfulCols[colIndex]
                for row in range(colIndex):
                    val = matrix[row, curCol]
                    if val != 0:
                        matrix[row] = (matrix[row] - val*matrix[colIndex]) % q
        return successfulCols[:numSuccessfulCols]


def orthogonalComplement(q, matrix, columns=None):
    """Computes an orthogonal complement (in GF(q)) to the given matrix."""
    matrix = np.asarray(matrix.copy())
    m, n = matrix.shape
    unitCols = gaussianElimination(q, matrix, columns)
    nonunitCols = np.array([x for x in xrange(n) if x not in unitCols])
    rank = unitCols.size
    nonunitPart = matrix[:rank, nonunitCols].transpose()
    k = n - rank
    result = np.zeros((k, n), dtype=np.int)
    for i, c in enumerate(unitCols):
        result[:, c] = (-nonunitPart[:, i]) % q
    for i, c in enumerate(nonunitCols):
        result[i, c] = 1
    return result


def binaryEmbedding(vector, q):
    """Return the binary embedding of a q-ary vector :math:`\in GF(q)^n` into :math:`GF(2)^{(
    q-1)\times n}` using the map
      0   -> 0 ... 0
      1   -> 0 ... 1
      2   -> 0 ..1 0
      ..
      q-1 -> 1 0.. 0
    """
    out = np.zeros((q-1) * vector.size, dtype=np.int)
    for i, val in enumerate(vector):
        if val != 0:
            out[i*(q-1)+(q-val-1)] = 1
    return out

if __name__ == '__main__':
    H = [[1,1,1,1,1]]
    code = NonbinaryLinearBlockCode(parityCheckMatrix=H, q=3, name='NBtestCode')
    codewords = [binaryEmbedding(cw, code.q) for cw in code.allCodewords()]
    from lpdec.polytopes import convexHull
    print(codewords)
    print(len(codewords))
    print(code.generatorMatrix)
    A,b  = convexHull(codewords)
    print(matrices.formatMatrix(A))
    print(matrices.formatMatrix(b))
    print(matrices.formatMatrix(np.hstack((A,b)), width=3))