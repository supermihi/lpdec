# -*- coding: utf-8 -*-
# cython: embedsignature=True
# cython: boundscheck=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: wraparound=False
# Copyright 2014 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation

"""This module contains linear algebra functions in mod2 arithmetics."""

cimport numpy as np
import numpy as np


cpdef gaussianElimination(np.int_t[:,:] matrix, np.int_t[:] columns=None, bint diagonalize=True):
        """The Gaussian elimination algorithm in GF(2) arithmetics.

        When called on a `(k × n)` matrix, the algorithm performs Gaussian elimination,
        bringin the matrix to reduced row echelon form.

        `columns`, if given, is a sequence of column indices, giving the the order in which columns
        of the matrix are visited, and defaults to the canonical ordering.

        `diagonalize` specifies whether the tridiagonalized columns should also be made unit
        vectors, yielding a diagonal structure among that columns.

        Warning: this is an in-place method, modifying the original matrix!
        """
        cdef:
            int nrows = matrix.shape[0]
            int ncols = matrix.shape[1]
            int curRow = 0, row, curCol, colIndex = 0
            int pivotRow, val, i
            np.int_t[:] successfulCols = np.empty(nrows, dtype=np.int)
            int numSuccessfulCols = 0

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
                for i in range(ncols):
                    val = matrix[curRow, i]
                    matrix[curRow, i] = matrix[pivotRow, i]
                    matrix[pivotRow, i] = val
            # do the actual pivoting
            for row in range(curRow + 1, nrows):
                val = matrix[row, curCol]
                if val != 0:
                    # modulo-2 add curRow to row
                    for i in range(ncols):
                        matrix[row, i] ^=  matrix[curRow, i]
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
                        for i in range(ncols):
                            matrix[row, i] ^= matrix[colIndex, i]
        return successfulCols[:numSuccessfulCols]


def rank(matrix):
    """Return the rank (in GF(2)) of a matrix."""
    ans = gaussianElimination(matrix.copy(), diagonalize=False)
    return ans.size


def orthogonalComplement(matrix, columns=None):
    """Computes an orthogonal complement (in GF(2)) to the given matrix."""
    matrix = np.asarray(matrix.copy())
    m, n = matrix.shape
    unitCols = gaussianElimination(matrix, columns, diagonalize=True)
    nonunitCols = np.array([x for x in xrange(n) if x not in unitCols])
    rank = unitCols.size
    nonunitPart = matrix[:rank, nonunitCols].transpose()
    k = n - rank
    result = np.zeros((k, n), dtype=np.int)
    for i, c in enumerate(unitCols):
        result[:, c] = nonunitPart[:, i]
    for i, c in enumerate(nonunitCols):
        result[i, c] = 1
    return result