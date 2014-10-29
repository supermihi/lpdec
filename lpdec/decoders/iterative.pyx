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

from __future__ import division, print_function
from collections import OrderedDict
cimport numpy as np
import numpy as np
from libc.math cimport tanh, atanh, fmin, fmax, fabs, isnan
from lpdec.decoders cimport Decoder
from lpdec.mod2la cimport gaussianElimination


cdef double almostInf = 1e5
cdef double inf = np.inf


cdef class IterativeDecoder(Decoder):
    """Class for iterative decoders, i.e., min-sum or sum-product.
    """

    def __init__(self, code, minSum=False, iterations=20,
                 reencodeOrder=-1,
                 reencodeRange=1,
                 reencodeIfCodeword=False,
                 name=None):
        if name is None:
            name = ('MinSum' if minSum else 'SumProduct') + '({})'.format(iterations)
            if reencodeOrder >= 0:
                name += '[order-{}]'.format(reencodeOrder)
        Decoder.__init__(self, code, name)
        self.name = name
        self.minSum = minSum
        self.reencodeRange = reencodeRange
        self.iterations = iterations
        self.reencodeOrder = reencodeOrder
        self.reencodeIfCodeword = reencodeIfCodeword
        mat = self.code.parityCheckMatrix
        k, n = code.parityCheckMatrix.shape
        if reencodeOrder >= 0:
            self.syndrome = np.zeros(code.blocklength, dtype=np.int)
            self.candidate = np.zeros(code.blocklength, dtype=np.int)
            self.indices = np.zeros(reencodeOrder, dtype=np.int)
            self.matrix = mat.copy()
            self.pool = np.zeros(code.blocklength, dtype=np.int)
            self.varNeigh2 = np.zeros((n, k), dtype=np.int)
            self.varDeg2 = np.zeros(code.blocklength, dtype=np.int)
            self.fixSyndrome = np.zeros(code.blocklength, dtype=np.int)
        self.fixes = np.zeros(n, dtype=np.double)
        self.solution = np.empty(n, dtype=np.double)
        self.checkNodeSatStates = np.empty(k, dtype=np.int)
        self.varSoftBits = np.empty(n, dtype=np.double)
        self.varHardBits = np.empty(n, dtype=np.int)
        self.varNodeDegree = np.empty(n, dtype=np.int)
        self.checkNodeDegree = np.empty(k, dtype=np.int)
        self.varToChecks = np.empty( (n, k), dtype=np.double)
        self.checkToVars = np.empty( (k, n), dtype=np.double)
        self.checkNeighbors = np.empty( (k, n), dtype=np.int)
        self.varNeighbors = np.empty( (n, k), dtype=np.int)
        self.fP = np.empty(n+1, dtype=np.double)
        self.bP = np.empty(n+1, dtype=np.double)

        for j in range(n):
            self.varNodeDegree[j] = 0
            for i in range(k):
                if mat[i, j]:
                    self.varNeighbors[j, self.varNodeDegree[j]] = i
                    self.varNodeDegree[j] += 1
        for i in range(k):
            self.checkNodeDegree[i] = 0
            for j in range(n):
                if mat[i, j]:
                    self.checkNeighbors[i, self.checkNodeDegree[i]] = j
                    self.checkNodeDegree[i] += 1

    cpdef setStats(self, object stats):
        for param in 'iterations', 'noncodewords':
            if param not in stats:
                stats[param] = 0
        Decoder.setStats(self, stats)

    cpdef fix(self, int index, int val):
        """Variable fixing is implemented by adding :attr:`fixes` to the LLRs. This vector
        contains :math:`\infty` for a fix to zero and :math:`-\infty` for a fix to one.
        """
        self.fixes[index] = (.5 - val) * inf

    cpdef release(self, int index):
        self.fixes[index] = 0

    cpdef solve(self, double lb=-inf, double ub=inf):
        cdef:
            np.int_t[:]      checkNodeSatStates = self.checkNodeSatStates
            np.int_t[:]      varHardBits = self.varHardBits
            np.int_t[:]      varNodeDegree = self.varNodeDegree
            np.int_t[:]      checkNodeDegree = self.checkNodeDegree
            np.int_t[:,:]    varNeighbors = self.varNeighbors, checkNeighbors = self.checkNeighbors
            np.double_t[:,:] varToChecks = self.varToChecks, checkToVars = self.checkToVars
            np.double_t[:]   varSoftBits = self.varSoftBits, bP = self.bP, fP = self.fP
            np.double_t[:]   llrs = self.llrs, solution = self.solution
            np.double_t[:]   llrFixed = self.llrs + self.fixes
            int i, j, deg, iteration, checkIndex, varIndex
            int numVarNodes = self.code.blocklength
            int numCheckNodes = self.code.parityCheckMatrix.shape[0]
            bint codeword, sign

        self.foundCodeword = False
        # reset messages
        for j in range(numVarNodes):
            for i in range(varNodeDegree[j]):
                varToChecks[j, varNeighbors[j, i]] = 0
        for i in range(numCheckNodes):
            for j in range(checkNodeDegree[i]):
                checkToVars[i, checkNeighbors[i, j]] = 0
        # reset satisfy state of check nodes
        for i in range(numCheckNodes):
            checkNodeSatStates[i] = False

        iteration = 0
        while iteration < self.iterations:
            iteration += 1
            # variable node processing
            for i in range(numVarNodes):
                varSoftBits[i] = llrFixed[i]
                for j in range(varNodeDegree[i]):
                    varSoftBits[i] += checkToVars[varNeighbors[i,j], i]
                    if isnan(varSoftBits[i]):
                        # this might happen if contradicting bits are fixed
                        self.objectiveValue = np.inf
                        return
                varHardBits[i] = ( varSoftBits[i] <= 0 )
                for j in range(varNodeDegree[i]):
                    checkIndex = varNeighbors[i,j]
                    varToChecks[i, checkIndex] = varSoftBits[i] - checkToVars[checkIndex, i]
                    checkNodeSatStates[checkIndex] ^= varHardBits[i]
            # check node processing
            codeword = True
            for i in range(numCheckNodes):
                deg = checkNodeDegree[i]
                if checkNodeSatStates[i]:
                    codeword = checkNodeSatStates[i] = False # reset for next iteration
                if self.minSum:
                    fP[0] = bP[deg] = inf
                    sign = False
                    for j in range(deg):
                        varIndex = checkNeighbors[i,j]
                        if varToChecks[varIndex, i] < 0:
                            fP[j+1] = fmin(fP[j], -varToChecks[varIndex, i])
                            sign = not sign
                        else:
                            fP[j+1] = fmin(fP[j], varToChecks[varIndex, i])
                        varIndex = checkNeighbors[i, deg-j-1]
                        bP[deg-1-j] = fmin(bP[deg-j], fabs(varToChecks[varIndex, i]))
                    for j in range(deg):
                        varIndex = checkNeighbors[i,j]
                        if sign ^ (varToChecks[varIndex,i] < 0):
                            checkToVars[i, varIndex] = -fmin(fP[j], bP[j+1])
                        else:
                            checkToVars[i, varIndex] = fmin(fP[j], bP[j+1])
                else:
                    fP[0] = bP[deg] = 1
                    for j in range(deg):
                        varIndex = checkNeighbors[i,j]
                        fP[j+1] = fP[j]*tanh(varToChecks[varIndex, i]/2.0)
                        varIndex = checkNeighbors[i, deg-j-1]
                        bP[deg-1-j] = bP[deg-j]*tanh(varToChecks[varIndex, i]/2.0)
                    for j in range(deg):
                        checkToVars[i, checkNeighbors[i,j]] = 2*atanh(fP[j]*bP[j+1])
            if codeword:
                self.foundCodeword = True
                break
        self.objectiveValue = 0
        # create solution from varHardBits
        for i in range(numVarNodes):
            solution[i] = varHardBits[i]
            if varHardBits[i]:
                self.objectiveValue += llrs[i]
        if not codeword:
            self._stats['noncodewords'] += 1
        if not codeword or (self.excludeZero and self.objectiveValue == 0) or self.reencodeIfCodeword:
            if not codeword or (self.excludeZero and self.objectiveValue == 0):
                self.objectiveValue = inf
            if self.reencodeOrder >= 0:
                self.reprocess()
        self._stats['iterations'] += iteration

    cdef int reprocess(self) except 1:
        """Perform order-i reprocessing, where i is given by :attr:`self.reencodeOrder`.
        """
        cdef int mod2sum, i, j, index, order, poolSize = 0, poolRange
        cdef double objVal
        cdef np.int_t[:] sorted = np.argsort(np.abs(self.varSoftBits))
        cdef np.int_t[:] indices = self.indices, pool = self.pool
        cdef np.int_t[:] candidate = self.candidate, syndrome = self.syndrome, fixSyndrome = \
            self.fixSyndrome
        cdef np.int_t[:] varHardBits = self.varHardBits, varDeg = self.varDeg2
        cdef np.int_t[:,:] matrix = self.matrix
        cdef np.int_t[:,:] varNeigh = self.varNeigh2
        cdef np.double_t[:] fixes = self.fixes, solution = self.solution, llrs = self.llrs
        cdef np.int_t[:] unit = np.asarray(gaussianElimination(matrix, sorted, True))

        for j in unit:
            if fixes[j] != 0:
                # unit column is fixed -> not enough "free" unit columns -> no reprocessing possible
                return 0
        # first, data structures are built up.
        # pool: contains the variable indices that are neither part of the unit matrix nor fixed,
        #   ie, the pool of indices that can be flipped during order-i reprocessing. poolSize
        #   indicates what part of pool is valid.
        # fixSyndrome: per check, stores the syndrome (mod-2 sum) of the fixed columns of H.
        for j in range(matrix.shape[0]):
            fixSyndrome[j] = 0
        for i in range(self.code.blocklength):
            j = sorted[i]
            if j not in unit:
                if fixes[j] == 0:
                    pool[poolSize] = j
                    poolSize += 1
                elif fixes[j] == -inf:
                    for row in range(matrix.shape[0]):
                        fixSyndrome[row] ^= matrix[row, j]
        # poolRange: one plus maximum pool index for flipping due to reencodeRange limitation.
        poolRange = int(poolSize * self.reencodeRange)
        # varDeg: record variable degrees of indices in pool (wrt Gaussed matrix)
        # varNeigh: record neighborhood of variables in pool (wrt Gaussed matrix)
        for j in range(poolSize):
            varDeg[j] = 0
            for i in range(matrix.shape[0]):
                if matrix[i, pool[j]] == 1:
                    varNeigh[j, varDeg[j]] = i
                    varDeg[j] += 1

        for order in range(0, self.reencodeOrder+1):
            # we need at least `order` flippable positions in the allowed pool range!
            if order > poolRange:
                break
            # reset candidate and syndrome
            for j in range(self.code.blocklength):
                candidate[j] = <int>varHardBits[j]
            for row in range(matrix.shape[0]):
                syndrome[row] = fixSyndrome[row]
            for j in range(poolSize):
                if candidate[pool[j]]:
                    for i in range(varDeg[j]):
                        syndrome[varNeigh[j, i]] ^= 1
            # now, syndrome reflects the syndrome of the matrix except for the unit part

            # this is inspired by the example implementation of itertools.combinations in the
            # python docs
            # First, flip bits 0, ..., order-1, and reencode this configuration.
            for i in range(order):
                indices[i] = i
                candidate[pool[indices[i]]] ^= 1
                for j in range(varDeg[indices[i]]):
                    syndrome[varNeigh[indices[i], j]] ^= 1
            # reencode
            objVal = 0
            for row in range(unit.shape[0]):
                candidate[unit[row]] = syndrome[row]
            for j in range(self.code.blocklength):
                objVal += candidate[j]*llrs[j]
            if objVal < self.objectiveValue and (not self.excludeZero or objVal != 0):
                self.objectiveValue = objVal
                self.foundCodeword = True
                for j in range(self.code.blocklength):
                    solution[j] = candidate[j]
            # now, move the `order` positions through all configurations among the feasible pool.
            while True:
                for i in range(order - 1, -1, -1):
                    if indices[i] != i + poolRange - order:
                        break
                else:
                    break
                index = pool[indices[i]]
                candidate[pool[indices[i]]] ^= 1
                for j in range(varDeg[indices[i]]):
                    syndrome[varNeigh[indices[i], j]] ^= 1
                indices[i] += 1
                index = pool[indices[i]]
                candidate[pool[indices[i]]] ^= 1
                for j in range(varDeg[indices[i]]):
                    syndrome[varNeigh[indices[i], j]] ^= 1
                for j in range(i + 1, order):
                    candidate[pool[indices[j]]] ^= 1
                    for index in range(varDeg[indices[j]]):
                        syndrome[varNeigh[indices[j], index]] ^= 1
                    indices[j] = indices[j-1] + 1
                    candidate[pool[indices[j]]] ^= 1
                    for index in range(varDeg[indices[j]]):
                        syndrome[varNeigh[indices[j], index]] ^= 1
                # reencode
                objVal = 0
                for row in range(unit.shape[0]):
                    candidate[unit[row]] = syndrome[row]
                for j in range(self.code.blocklength):
                    objVal += candidate[j] * llrs[j]
                if objVal < self.objectiveValue and (not self.excludeZero or objVal != 0):
                    self.objectiveValue = objVal
                    self.foundCodeword = True
                    for j in range(self.code.blocklength):
                        solution[j] = candidate[j]

    cpdef params(self):
        parms = OrderedDict()
        if self.minSum:
            parms['minSum'] = True
        parms['iterations'] = self.iterations
        if self.reencodeOrder != -1:
            parms['reencodeOrder'] = self.reencodeOrder
        if self.reencodeRange != 1:
            parms['reencodeRange'] = self.reencodeRange
        if self.reencodeIfCodeword:
            parms['reencodeIfCodeword'] = True
        parms['name'] = self.name
        return parms
