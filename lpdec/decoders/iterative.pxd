# -*- coding: utf-8 -*-
# Copyright 2014-2015 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation

from lpdec.decoders.base cimport Decoder
cimport numpy as np

cdef class IterativeDecoder(Decoder):

    cdef:
        int[:]    checkNodeSatStates
        double[:] varSoftBits
        int[:]    varHardBits
        int[:]    varNodeDegree
        int[:]    checkNodeDegree
        np.intp_t[:,::1]  varNeighbors
        np.intp_t[:,::1]  checkNeighbors
        double[:,:]  varToChecks
        double[:,:]  checkToVars
        double[:] fP, bP
        double[:] fixes
        int            iterations
        public int     reencodeOrder
        bint           minSum, reencodeIfCodeword
        public bint    excludeZero
        # helpers for the order-i reprocessing
        int[:]    syndrome, candidate, varDeg2, fixSyndrome
        np.intp_t[:]   indices, pool
        np.intp_t[:,:] varNeigh2
        int            maxRange
        double         reencodeRange, sentObjective
        np.int_t  [:,::1] matrix

    cpdef setLLRs(self, double[::1] llrs, np.int_t[::1] sent=?)
    cpdef solve(self, double lb=?, double ub=?)

    cpdef fix(self, int index, int val)
    cpdef release(self, int index)

    cdef int reprocess(self) except 1
