# -*- coding: utf-8 -*-
# Copyright 2014 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation

from __future__ import unicode_literals, print_function
import numpy as np


def getBinaryMatrix(source):
    """Creates a binary matrix of type :class:`np.ndarray` from either a file or a
    two-dimensional python list.

    If `source` is a file path, the file must either contain an explicit representation of the
    matrix (by means of whitespace-separated '0' and '1' characters) or be in the AList format by
    David MacKay (http://www.inference.phy.cam.ac.uk/mackay/codes/alist.html).

    :rtype: :class:`np.ndarray` with dtype `np.int`.
    :returns: A numpy ndarray representation of the matrix.
    """
    try:
        isString = isinstance(source, basestring)
    except NameError:
        isString = isinstance(source, str)
    if isString:
        with open(source, 'rt') as f:
            lines = [[int(x) for x in l.strip().split()]
                     for l in f.readlines()
                     if len(l.strip()) > 0]
    else:
        assert hasattr(source, '__iter__') and hasattr(source[0], '__iter__')
        import copy
        lines = copy.copy(source)
    if lines[0][0] in (0, 1):  # explicit 0/1 representation
        return np.array(lines, dtype=np.int)
    # AList format
    cols, rows = lines[0]
    data = np.zeros((rows, cols), dtype=np.int)
    # lines[1] contains max per row/col information, not needed here
    nextline = lines[2]
    if len(nextline) == cols:
        # skip also the next two line which are not needed (contain exact per row/col information)
        # (omitted in the 'reduced Alist format')
        del lines[:4]
    else:
        del lines[:2]
    for column, line in zip(range(0, cols), lines):
        for index in line:
            if index != 0:
                data[index - 1, column] = 1
    return data


def toListAlist(matrix):
    """Convert the matrix into an "alist-like" representation as list of lists.

    The result is a list of lists. The first entry is a pair containing number of columns and rows,
    respectively. The second entry is an empty list (in MacKay's Alist format, this contains the
    max number of entries per column and row, respectively). Then, for each column, a list of
    nonzero positions follows.
    """
    out = [[matrix.shape[1], matrix.shape[0]], []]
    for i in range(matrix.shape[1]):
        out.append((matrix[:, i].nonzero()[0] + 1).tolist())
    return out


def formatMatrix(matrix, format='plain', width=2, filename=None):
    """Converts a matrix to a string in the requested format.

    :param str format: The output format. It is either ``'plain'`` (the default), which is the
        "canonical" representation (entries separated by spaces and newlines), or 'alist',
        which leads to MacKay's Alist format.
    :param str width: For `plain` output format, this specifies the width in which a matrix entry is
        formatted.
    :param str filename: If given, the string is written to the given file; otherwise it is
        returned.
    """
    if format == 'plain':
        if matrix.ndim == 2:
            mstring = '\n'.join(formatMatrix(row, width=width) for row in matrix)
        else:
            mstring = ''.join(('{:' + str(width) + 'd}').format(k) for k in matrix)
    else:
        assert format == 'alist'
        import cStringIO
        output = cStringIO.StringIO()
        maxColSum = max(col.sum() for col in matrix.T)
        maxRowSum = max(row.sum() for row in matrix)
        output.write('{1} {0}\n'.format(*matrix.shape))
        output.write('{0} {1}\n'.format(maxColSum, maxRowSum))
        output.write(' '.join(str(col.sum()) for col in matrix.transpose()) + '\n')
        output.write(' '.join(str(row.sum()) for row in matrix) + '\n')
        for i in range(matrix.shape[1]):
            nzIndices = matrix[:, i].nonzero()[0] + 1  # indices start with '1' in alist format
            output.write(' '.join(str(v) for v in nzIndices))
            for j in range(nzIndices.size, maxColSum):
                output.write(' 0')
            output.write('\n')
        for row in matrix:
            nzIndices = row.nonzero()[0] + 1
            output.write(' '.join(str(v) for v in nzIndices))
            for j in range(nzIndices.size, maxRowSum):
                output.write(' 0')
            output.write('\n')
        mstring = output.getvalue()
        output.close()
    if filename:
        with open(filename, 'wt') as f:
            f.write(mstring)
    else:
        return mstring