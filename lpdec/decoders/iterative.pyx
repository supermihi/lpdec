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
from libc.math cimport tanh, atanh, fmin, fmax, fabs
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
                 excludeZero=False,
                 name=None):
        if name is None:
            name = ('MinSum' if minSum else 'SumProduct') + '({})'.format(iterations)
            if reencodeOrder >= 0:
                name += '[order-{}]'.format(reencodeOrder)
        Decoder.__init__(self, code, name)
        if reencodeOrder >= 0:
            self.syndrome = np.empty(code.blocklength, dtype=np.int)
            self.candidate = np.empty(code.blocklength, dtype=np.int)
            self.indices = np.empty(reencodeOrder, dtype=np.int)
            self.matrix = code.parityCheckMatrix.copy()
            self.pool = np.empty(code.blocklength, dtype=np.int)
        self.name = name
        self.minSum = minSum
        self.reencodeRange = reencodeRange
        self.maxRange = int(self.reencodeRange * self.code.blocklength)
        self.excludeZero = excludeZero
        self.iterations = iterations
        self.reencodeOrder = reencodeOrder
        self.reencodeIfCodeword = reencodeIfCodeword
        mat = self.code.parityCheckMatrix
        k, n = code.parityCheckMatrix.shape
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


    cpdef solve(self, np.int_t[:] hint=None, double lb=-np.inf, double ub=np.inf):
        cdef:
            np.int_t[:] checkNodeSatStates = self.checkNodeSatStates

            np.double_t[:] varSoftBits = self.varSoftBits
            np.int_t[:]    varHardBits = self.varHardBits
            np.int_t[:]    varNodeDegree = self.varNodeDegree
            np.int_t[:]    checkNodeDegree = self.checkNodeDegree
            np.int_t[:,:]  varNeighbors = self.varNeighbors
            np.int_t[:,:]  checkNeighbors = self.checkNeighbors
            np.double_t[:,:]  varToChecks = self.varToChecks
            np.double_t[:,:]  checkToVars = self.checkToVars
            int i, j, deg, iteration, outerIteration = 0
            int checkIndex, varIndex
            int numVarNodes = self.code.blocklength
            int numCheckNodes = self.code.parityCheckMatrix.shape[0]
            bint codeword, sign
            np.double_t[:] bP = self.bP, fP = self.fP
            np.double_t[:] llrs = self.llrs
            np.double_t[:] solution = self.solution
            np.double_t[:] llrFixed = self.llrs + self.fixes

        self.foundCodeword = False
        for j in range(numVarNodes):
            for i in range(varNodeDegree[j]):
                varToChecks[j, varNeighbors[j, i]] = 0
        for i in range(numCheckNodes):
            for j in range(checkNodeDegree[i]):
                checkToVars[i, checkNeighbors[i, j]] = 0

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
                    codeword = False
                    checkNodeSatStates[i] = False # reset for next iteration
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
        for i in range(numVarNodes):
            solution[i] = varHardBits[i]
            if varHardBits[i]:
                self.objectiveValue += llrs[i]

        if not codeword or (self.excludeZero and self.objectiveValue == 0) or self.reencodeIfCodeword:
            if not codeword:
                self._stats['noncodewords'] += 1
            if self.reencodeOrder >= 0:
                self.reprocess()
        self._stats['iterations'] += iteration


    cdef void _flipBit(self, int index):
        cdef int row
        cdef np.int_t[:] syndrome = self.syndrome, candidate = self.candidate
        cdef np.int_t[:,:] matrix = self.matrix
        candidate[index] = 1 - candidate[index]
        for row in range(matrix.shape[0]):
            if matrix[row, index] == 1:
                syndrome[row] = 1 - syndrome[row]


    cdef void _reencode(self):
        cdef int i
        cdef double objVal = 0
        cdef np.int_t[:] candidate = self.candidate, syndrome = self.syndrome, unit = self.unit
        cdef np.double_t[:] llrs = self.llrs
        for i in range(unit.shape[0]):
            candidate[unit[i]] = syndrome[i]
        for i in range(self.code.blocklength):
            objVal += candidate[i]*llrs[i]
        if objVal < self.objectiveValue and (not self.excludeZero or objVal != 0):
            self.objectiveValue = objVal
            self.foundCodeword = True
            for i in range(self.code.blocklength):
                self.solution[i] = candidate[i]
            assert self.solution in self.code


    cdef int reprocess(self):
        cdef int mod2sum, i, j, index
        cdef np.int_t[:] sorted = np.argsort(np.abs(self.varSoftBits))
        cdef np.int_t[:] indices = self.indices, pool = self.pool
        cdef np.int_t[:] candidate = self.candidate, syndrome = self.syndrome
        cdef np.int_t[:,:] matrix = self.matrix
        self.unit = np.asarray(gaussianElimination(self.matrix, sorted, True))
        self.pool = np.array([i for i in sorted[:self.maxRange]
                              if i not in self.unit and self.fixes[i] == 0])
        self.objectiveValue = np.inf
        pool = self.pool

        for self.order in range(0, self.reencodeOrder+1):
            # need at least ``order`` flippable positions!
            if self.order > pool.shape[0]:
                break
            # reset candidate and syndrome
            for j in range(self.code.blocklength):
                candidate[j] = <int>self.solution[j]
            for i in range(self.matrix.shape[0]):
                mod2sum = 0
                for j in range(pool.size):
                    if self.matrix[i, pool[j]] == 1:
                        mod2sum += candidate[pool[j]]
                self.syndrome[i] = mod2sum % 2
            # this is inspired by the example implementation of itertools.combinations in the
            # python docs
            for i in range(self.order):
                indices[i] = i
                self._flipBit(pool[i])
            self._reencode()
            if self.order == 0:
                continue
            while True:
                for i in range(self.order - 1, -1, -1):
                    if indices[i] != i + pool.shape[0] - self.order:
                        break
                else:
                    break
                #self._flipBit(pool[indices[i]])
                index = pool[indices[i]]
                candidate[index] = 1 - candidate[index]
                for row in range(matrix.shape[0]):
                    if matrix[row, index] == 1:
                        syndrome[row] = 1 - syndrome[row]
                indices[i] += 1
                index = pool[indices[i]]
                candidate[index] = 1 - candidate[index]
                for row in range(matrix.shape[0]):
                    if matrix[row, index] == 1:
                        syndrome[row] = 1 - syndrome[row]
                for j in range(i + 1, self.order):
                    index = pool[indices[j]]
                    candidate[index] = 1 - candidate[index]
                    for row in range(matrix.shape[0]):
                        if matrix[row, index] == 1:
                            syndrome[row] = 1 - syndrome[row]
                    indices[j] = indices[j-1] + 1
                    index = pool[indices[j]]
                    candidate[index] = 1 - candidate[index]
                    for row in range(matrix.shape[0]):
                        if matrix[row, index] == 1:
                            syndrome[row] = 1 - syndrome[row]
                self._reencode()
            # un-flip all bits
            for i in range(self.order):
                index = pool[indices[i]]
                candidate[index] = 1 - candidate[index]
                for row in range(matrix.shape[0]):
                    if matrix[row, index] == 1:
                        syndrome[row] = 1 - syndrome[row]

    cpdef params(self):
        parms = OrderedDict()
        if self.minSum:
            parms['minSum'] = True
        parms['iterations'] = self.iterations
        if self.reencodeOrder != -1:
            parms['reencodeOrder'] = self.reencodeOrder
        if self.reencodeIfCodeword:
            parms['reencodeIfCodeword'] = True
        if self.excludeZero:
            parms['excludeZero'] = True
        parms['name'] = self.name
        return parms