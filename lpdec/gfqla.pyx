# -*- coding: utf-8 -*-
# cython: cdivision=False, wraparound=False, initializedcheck=False, nonecheck=False
# Copyright 2014-2015 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation

"""This module contains linear algebra functions in :math:`GF(q)` arithmetics, i.e. vector spaces
defined over finite fields."""

import numpy as np
cimport numpy as np


cpdef gaussianElimination(np.int_t[:,::1] matrix, np.intp_t[:] columns=None, bint diagonalize=True,
                          np.intp_t[::1] successfulCols=None, int q=2):
        """
        gaussianElimination(matrix, columns=None, diagonalize=True, successfulCols=None, q=2)

        The Gaussian elimination algorithm in :math:`\mathbb F_q` arithmetics, turning a given
        matrix into reduced row echelon form by elementary row operations.

        .. warning:: This algorithm operates **in-place**!

        Parameters
        ----------
        matrix : np.int_t[:,::1]
            The matrix to operate on.
        columns : np.intp_t[:], optional
            A sequence of column indices, giving the the order in which columns of the matrix are
            visited. Defaults to ``range(matrix.shape[1])``.
        diagonalize : bool, True
            If ``True``, matrix elements above the pivots will be turned to zeros, leading to a
            diagonal submatrix. Otherwise, the result contains an upper triangular submatrix.
        successfulCols : np.intp_t[::1], optinonal
            Numpy array in which successfully diagonalized column indices are stored. If supplied,
            this array will be used for the return value. Otherwise, a new array will be created,
            resulting in a slight performance drain.
        q : int, optional
            Field size in which operations should be performed. Defaults to ``2``.

        Returns
        -------
        np.intp_t[::1]
            Indices of successfully diagonalized columns.
        """
        cdef:
            int nrows = matrix.shape[0]
            int ncols = matrix.shape[1]
            int curRow = 0, row, curCol, colIndex = 0
            int pivotRow, val, i, factor
            int numSuccessfulCols = 0
        # assert q < cachedInvs.shape[0]
        if successfulCols is None:
            successfulCols = np.empty(nrows, dtype=np.intp)
        if columns is None:
            columns = np.arange(ncols, dtype=np.intp)
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
                # swap rows
                for i in range(ncols):
                    val = matrix[curRow, i]
                    matrix[curRow, i] = matrix[pivotRow, i]
                    matrix[pivotRow, i] = val
            # do the actual pivoting
            if matrix[curRow, curCol] > 1:
                # "divide" by pivot element to set it to 1
                if q > 2:
                    factor = cachedInvs[q, matrix[curRow, curCol]]
                    for i in range(ncols):
                        matrix[curRow, i] = (matrix[curRow, i] * factor) % q
            for row in range(curRow + 1, nrows):
                val = matrix[row, curCol]
                if val != 0:
                    for i in range(ncols):
                        if q == 2:
                            matrix[row, i] ^= matrix[curRow, i]
                        else:
                            matrix[row, i] =  (matrix[row, i] -val*matrix[curRow, i]) % q
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
                            if q == 2:
                                matrix[row, i] ^= matrix[colIndex, i]
                            else:
                                matrix[row, i] = (matrix[row, i] - val*matrix[colIndex, i]) % q
        return successfulCols[:numSuccessfulCols]


# precompute multiplicative inverses for primes 2, 3, 5, 7, 11, 13
cdef int maxCachedQ = 13;
cdef int[:, ::1] cachedInvs = np.zeros((maxCachedQ + 1, maxCachedQ), dtype=np.intc)
for q in (2, 3, 5, 7, 11, 13):
    for a in range(1, q):
        for b in range(1, q):
            if a*b % q == 1:
                cachedInvs[q, a] = b
                break


def inv(int a, int q):
    """
    Computes the multiplicative inverse of `a` in :math:`\mathbb F_q`."""
    if q < cachedInvs.shape[0]:
        if cachedInvs[q, a] == 0:
            raise ValueError('{} not inverible modulo {}'.format(a, q))
        return cachedInvs[q, a]
    cdef int t, newt, r, newr, quotient
    t = 0
    newt = 1
    r = q
    newr = a
    while newr != 0:
        quotient = r // newr
        t, newt = newt, t - quotient * newt
        r, newr = newr, r - quotient * newr
    if r > 1:
        raise ValueError('{} not invertible modulo {}'.format(a, q))
    return t if t >= 0 else t + q


def rank(matrix, q=2):
    """Return the rank (in GF(q)) of a matrix."""
    diagCols = gaussianElimination(matrix.copy(), diagonalize=False, q=q)
    return diagCols.size


def orthogonalComplement(matrix, columns=None, q=2):
    """Computes an orthogonal complement (in GF(q)) to the given matrix."""
    matrix = np.asarray(matrix.copy())
    m, n = matrix.shape
    unitCols = gaussianElimination(matrix, columns, diagonalize=True, q=q)
    nonunitCols = np.array([x for x in range(n) if x not in unitCols])
    rank = unitCols.size
    nonunitPart = matrix[:rank, nonunitCols].transpose()
    k = n - rank
    result = np.zeros((k, n), dtype=np.int)
    for i, c in enumerate(unitCols):
        result[:, c] = (-nonunitPart[:, i]) % q
    for i, c in enumerate(nonunitCols):
        result[i, c] = 1
    return result


cpdef inKernel(np.int_t[:, :] matrix, np.int_t[:] vector, int q=2):
    """
    inKernel(matrix, vector, q=2)

    Return True iff `matrix` :math:`\\times` `vector` = 0 in :math:`\mathbb F_q`."""
    cdef int row, col, sum
    assert matrix.shape[1] == vector.shape[0]
    for row in range(matrix.shape[0]):
        sum = 0
        for col in range(matrix.shape[1]):
            sum += matrix[row, col] * vector[col]
        if sum % q != 0:
            return False
    return True