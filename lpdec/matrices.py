# -*- coding: utf-8 -*-
# Copyright 2014 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation

from __future__ import unicode_literals, print_function
import numpy as np


def strBinary(matrix, n=2):
    """Convert a binary matrix to a string using `n` characters for each column."""
    if matrix.ndim == 2:
        return "\n".join(strBinary(row, n) for row in matrix)
    else:
        return "".join(('{0:' + str(n) + 'd}').format(int(k)) for k in matrix)


def getBinaryMatrix(source):
    """Creates a binary matrix of type :class:`np.ndarray` from either a file or a
    two-dimensional python list.

    If `source` is a file path, the file must either contain an explicit representation of the
    matrix (by means of whitespace-separated '0' and '1' characters) or be in the AList format by
    David MacKay (http://www.inference.phy.cam.ac.uk/mackay/codes/alist.html).

    :rtype: :class:`np.ndarray` with dtype `np.int`.
    :returns: A numpy ndarray representation of the matrix.
    """
    if isinstance(source, basestring):
        with open(source, "rt") as f:
            lines = [ [int(x) for x in l.strip().split()]
                      for l in f.readlines()
                      if len(l.strip()) > 0 ]
    else:
        assert hasattr(source, "__iter__") and hasattr(source[0], "__iter__")
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