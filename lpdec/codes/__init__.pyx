# -*- coding: utf-8 -*-
# cython: embedsignature=True
# Copyright 2014 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation
from __future__ import division, unicode_literals
import os.path
from collections import OrderedDict
cimport numpy as np
import numpy as np
from lpdec.persistence cimport JSONDecodable
from lpdec import matrices, mod2la


cdef class BinaryLinearBlockCode(JSONDecodable):
    """Base class for binary linear block codes.

    `name` is a string uniquely describing the code. When storing results into a database, there
    must not be two different codes with the same name. If the parity-check matrix is given by a
    file, the code's name defaults to the name of that file.

    The class can be instanciated directly by providing a parity check matrix; the argument
    `parityCheckMatrix` must be either the path of a file containig the matrix, or
    a two-dimensional list or a :class:`np.ndarray` representation of the matrix.
    Subclasses using a different code representation should leave the default value of ``None``.

    """

    def __init__(self, name=None, parityCheckMatrix=None):
        JSONDecodable.__init__(self)
        if parityCheckMatrix is not None:
            if isinstance(parityCheckMatrix, basestring):
                self.filename = os.path.expanduser(parityCheckMatrix)
                hmatrix = matrices.getBinaryMatrix(self.filename)
                if name is None:
                    name = os.path.basename(self.filename)
            elif not isinstance(parityCheckMatrix, np.ndarray):
                hmatrix = matrices.getBinaryMatrix(parityCheckMatrix)
            else:
                hmatrix = parityCheckMatrix
            self._parityCheckMatrix = hmatrix
            cols = hmatrix.shape[1]
            rank = mod2la.rank(hmatrix)
            self.blocklength = cols
            self.infolength = cols - rank
        if name is None:
            raise ValueError("A code must have a name.")
        self.name = name

    @property
    def rate(self):
        return self.infolength / self.blocklength

    @property
    def generatorMatrix(self):
        """Generator matrix of this code; generated on first access if not available."""
        if self._generatorMatrix is None:
            if self._parityCheckMatrix is None:
                # neither parity-check nor generator matrix exist -> encode all unit vectors
                self._generatorMatrix = np.zeros((self.infolength, self.blocklength), dtype=np.int)
                infoWord = np.zeros(self.infolength, dtype=np.int)
                for i in range(self.infolength):
                    infoWord[i] = 1
                    if i > 0:
                        infoWord[i-1] = 0
                    self._generatorMatrix[i, :] = self.encode(infoWord)
            else:
                cols = np.hstack((np.arange(self.infolength, self.blocklength),
                                  np.arange(self.infolength)))
                self._generatorMatrix = mod2la.orthogonalComplement(self._parityCheckMatrix, cols)
        return self._generatorMatrix

    @property
    def parityCheckMatrix(self):
        """The parity-check matrix, calculated on first access if not given a priori."""
        if self._parityCheckMatrix is None:
            self._parityCheckMatrix = mod2la.orthogonalComplement(self.generatorMatrix)
        return self._parityCheckMatrix

    cpdef np.ndarray[ndim=1, dtype=np.int_t] encode(self, np.ndarray[ndim=1, dtype=np.int_t] infoword):
        """Encode an information word.

        :param infoword: The information word. Must be a numpy array with integer type.
        :returns: The resulting codeword.
        :rtype: numpy.ndarray of dimension one and type numpy.int_t.
        """
        return infoword.dot(self.generatorMatrix) % 2

    def __contains__(self, item):
        """Check if the given word is a codeword of this code.
        """
        return np.all(self.parityCheckMatrix.dot(item) % 2 == 0)

    def __str__(self):
        return self.name

    def params(self):
        matrix = self.parityCheckMatrix
        if np.sum(matrix) / ( matrix.shape[0] * matrix.shape[1]) < .1:
            # sparse matrix
            pcm = matrices.toListAlist(matrix)
        else:
            pcm = matrix.tolist()
        return OrderedDict([('parityCheckMatrix', pcm), ('name', self.name)])
