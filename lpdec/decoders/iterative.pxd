# -*- coding: utf-8 -*-
# Copyright 2014 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation

from lpdec.decoders.base cimport Decoder
cimport numpy as np


cdef class IterativeDecoder(Decoder):

    cdef:
        np.intp_t[:]    checkNodeSatStates
        np.double_t[:] varSoftBits
        np.intp_t[:]    varHardBits
        np.intp_t[:]    varNodeDegree
        np.intp_t[:]    checkNodeDegree
        np.intp_t[:,:]  varNeighbors
        np.intp_t[:,:]  checkNeighbors
        np.double_t[:,:]  varToChecks
        np.double_t[:,:]  checkToVars
        np.double_t[:] fP, bP
        np.double_t[:] fixes
        int            iterations
        int            reencodeOrder
        bint           minSum, reencodeIfCodeword
        public bint    excludeZero
        # helpers for the order-i reprocessing
        np.intp_t[:]    syndrome, candidate, indices, pool, varDeg2, fixSyndrome
        np.intp_t[:,:]  varNeigh2
        int            maxRange
        double         reencodeRange
        np.intp_t[:,:]  matrix

    cpdef solve(self, double lb=?, double ub=?)
    cpdef params(self)

    cpdef fix(self, int index, int val)
    cpdef release(self, int index)

    cdef int reprocess(self) except 1