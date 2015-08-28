# -*- coding: utf-8 -*-
# Copyright 2014-2015 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation

from __future__ import unicode_literals, print_function
import os.path, sys, bz2
import numpy as np
from lpdec import utils


def alistToNumpy(lines):
    """Converts a parity-check matrix in AList format to a 0/1 numpy array. The argument is a
    list-of-lists corresponding to the lines of the AList format, already parsed to integers
    if read from a text file.

    The AList format is introduced on http://www.inference.phy.cam.ac.uk/mackay/codes/alist.html.

    This method supports a "reduced" AList format where lines 3 and 4 (containing column and row
    weights, respectively) and the row-based information (last part of the Alist file) are omitted.

    Example:
        >>> alistToNumpy([[3,2], [2, 2], [1,1,2], [2,2], [1], [2], [1,2], [1,2,3,4]])
        array([[1, 0, 1],
               [0, 1, 1]])
    """
    nCols, nRows = lines[0]
    if len(lines[2]) == nCols and len(lines[3]) == nRows:
        startIndex = 4
    else:
        startIndex = 2
    matrix = np.zeros((nRows, nCols), dtype=np.int)
    for col, nonzeros in enumerate(lines[startIndex:startIndex + nCols]):
        for rowIndex in nonzeros:
            if rowIndex != 0:
                matrix[rowIndex - 1, col] = 1
    return matrix


def getBinaryMatrix(source):
    """Creates a binary matrix of type :class:`np.ndarray` from either a file or a
    two-dimensional python list.

    If `source` is a file path, the file must either contain an explicit representation of the
    matrix (by means of whitespace-separated '0' and '1' characters) or be in the AList format
    (see alistToNumpy docstring).

    The file may be bzip2'ed, in which case it will be decompressed automatically.

    Returns
    -------
    np.ndarray[dtype=int]
        Numpy ndarray representation of the given matrix.
    """
    if isinstance(source, np.ndarray):
        return source.astype(np.int)
    if utils.isStr(source):
        source = os.path.expanduser(source)
        fileObj = bz2.BZ2File(source, 'r') if source.endswith('bz2') else open(source, 'rt')
        with fileObj as f:
            lines = [[int(x) for x in l.strip().split()]
                     for l in f.readlines()
                     if len(l.strip()) > 0]

    else:
        assert hasattr(source, '__iter__') and hasattr(source[0], '__iter__')
        lines = source
    if lines[0][0] in (0, 1):  # explicit 0/1 representation
        return np.array(lines, dtype=np.int)
    return alistToNumpy(lines)


def numpyToAlist(matrix):
    """Converts a 2-dimensional 0/1 numpy array into MacKay's AList format, in form of a list of
    lists of integers.
    """
    if sys.version_info[0] == 2:
        import cStringIO
        StringIO = cStringIO.StringIO
    else:
        import io
        StringIO = io.StringIO

    with StringIO() as output:
        nRows, nCols = matrix.shape
        # first line: matrix dimensions
        output.write('{} {}\n'.format(nCols, nRows))

        # next three lines: (max) column and row degrees
        colWeights = matrix.sum(axis=0)
        rowWeights = matrix.sum(axis=1)

        maxColWeight = max(colWeights)
        maxRowWeight = max(rowWeights)

        output.write('{} {}\n'.format(maxColWeight, maxRowWeight))
        output.write(' '.join(map(str, colWeights)) + '\n')
        output.write(' '.join(map(str, rowWeights)) + '\n')

        def writeNonzeros(rowOrColumn, maxDegree):
            nonzeroIndices = np.flatnonzero(rowOrColumn) + 1  # AList uses 1-based indexing
            output.write(' '.join(map(str, nonzeroIndices)))
            # fill with zeros so that every line has maxDegree number of entries
            output.write(' 0' * (maxDegree - len(nonzeroIndices)))
            output.write('\n')

        # column-wise nonzeros block
        for column in matrix.T:
            writeNonzeros(column, maxColWeight)
        for row in matrix:
            writeNonzeros(row, maxRowWeight)
        return output.getvalue()


def numpyToString(array, width=2):
    """Formats a numpy matrix as string, using `width` columns per entry."""
    assert array.ndim <= 2
    if array.ndim == 2:
        return '\n'.join(numpyToString(row, width=width) for row in array)
    return ''.join(('{:<' + str(width) + 'd}').format(v) for v in array)


def formatMatrix(matrix, format='plain', width=2, filename=None):
    """Converts a matrix to a string in the requested format.

    Parameters
    ----------
    matrix : np.ndarray[int]
        The matrix to be formatted.
    format : {'plain', 'alist'}, optional
        The output format. ``'plain'`` (the default) means the "canonical" representation (entries
        separated by spaces and newlines), while ``'alist'`` leads to MacKay's Alist format.
    width : int, optional
        In ``plain`` output format, each entry of the matrix will be left-padded with blanks to
        match the given width.
    filename : str, optional
        If provided, write formatted matrix string to the given file.

    Returns
    -------
    str
        String representation of the given matrix.
    """
    if format == 'plain':
        mstring = numpyToString(matrix, width=width)
    else:
        assert format == 'alist'
        mstring = numpyToAlist(matrix)
    if filename:
        filename = os.path.expanduser(filename)
        fileObj = bz2.BZ2File(filename, 'w') if filename.endswith('bz2') else open(filename, 'wt')
        with fileObj as f:
            f.write(mstring.encode('ASCII'))
    return mstring


def numpyToReducedAlist(matrix):
    """Convert the matrix to reduced AList format in form of a list of lists.

    The first entry is a pair containing number of columns and rows, respectively. The second entry
    is an empty list (in MacKay's Alist format, this contains the max number of entries per column
    and row, respectively). Then, for each column, a list of nonzero positions follows.
    """
    out = [[matrix.shape[1], matrix.shape[0]], []]
    for i in range(matrix.shape[1]):
        out.append((matrix[:, i].nonzero()[0] + 1).tolist())
    return out


def getNonbinaryMatrix(source):
    """Reads a nonbinary matrix (no AList support)."""
    if utils.isStr(source):
        with open(source, 'rt') as f:
            lines = [[int(x) for x in l.strip().split()]
                     for l in f.readlines() if len(l.strip()) > 0]
        return np.array(lines, dtype=np.int)
    else:
        return np.array(source, dtype=np.int)