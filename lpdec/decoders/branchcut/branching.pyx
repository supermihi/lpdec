# -*- coding: utf-8 -*-
# Copyright 2015 Michael Helmling
# cython: language_level=3
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation
from __future__ import division

from numpy.math cimport INFINITY
cimport numpy as np
import numpy as np
from libc.math cimport fabs, fmax, fmin
from lpdec.decoders.base cimport Decoder
from lpdec.decoders.branchcut.node cimport Node

cdef class BranchingRule:

    def __init__(self, code, Decoder bcDecoder, lamb=None, mu=1./6):
        self.code = code
        if lamb is None:
            lamb = code.blocklength  # practically infinity
        self.lamb = lamb
        self.mu = mu
        self.bcDecoder = bcDecoder
        self.candInds = np.zeros(code.blocklength, dtype=np.intc)

    cdef int callback(self, Node node) except -1:
        pass

    cdef int reset(self) except -1:
        pass

    cdef int[:] candidates(self):
        cdef int index = 0, i
        for i in range(self.code.blocklength):
            if not self.bcDecoder.fixed(i):
                self.candInds[index] = i
                index += 1
        return self.candInds[:index]

    cdef double calculateScore(self, int index):
        raise NotImplementedError()

    cdef double score(self, double qminus, double qplus):
        """Implements the score function with parameter self.mu"""
        return (1-self.mu)*fmin(qminus, qplus) + self.mu*fmax(qminus, qplus)

    cdef int branchIndex(self, Node node, double ub, double[::1] solution) except -1:
        """Determines the index of the current branching variable"""
        cdef:
            int[:] candidates = self.candidates()
            int i, index, maxIndex = -1, itersSinceBest = 0
            double maxScore = -INFINITY, score
        self.ub = ub
        self.node = node
        self.beginScoreComputation()
        for i in range(candidates.size):
            index = candidates[i]
            score = self.calculateScore(index)
            if score > maxScore:
                maxIndex = i
                maxScore = score
                itersSinceBest = 0
            else:
                itersSinceBest += 1
            if itersSinceBest > self.lamb:
                break
        self.endScoreComputation()
        if maxIndex == -1:
            raise RuntimeError('no branch index found')
        return maxIndex

    cdef int beginScoreComputation(self) except -1:
        """Subclasses can put preparation for score computation in here."""
        pass

    cdef int endScoreComputation(self) except -1:
        """Subclasses can put cleanup operations after score computations in here."""
        pass


cdef class MostFractional(BranchingRule):

    cdef double calculateScore(self, int index):
        return -fabs(self.bcDecoder.lbProvider.solution[index] - .5)


cdef class LeastReliable(BranchingRule):

    cdef double calculateScore(self, int index):
        return -fabs(self.bcDecoder.llrs[index])


cdef class FirstFractional(BranchingRule):

    cdef bint onlyFractional

    def __init__(self, code, Decoder bcDecoder, bint onlyFractional=True):
        BranchingRule.__init__(self, code, bcDecoder)
        self.onlyFractional = onlyFractional

    cdef int branchIndex(self, Node node, double ub, double[::1] solution) except -1:
        cdef int i, index = -1
        for i in range(self.code.blocklength):
            if not self.bcDecoder.fixed(i):
                if index == -1:
                    index = i  # fallback if no entries are fractional
                if not self.onlyFractional or fabs(solution[i] - .5) < .4999:
                    return i
        return index


cdef class ReliabilityBranching(BranchingRule):

    cdef int rpcRounds, etaRel
    cdef double iterLimit, objBufLim
    cdef double[::1] PsiMinus, PsiPlus
    cdef int[::1] etaPlus, etaMinus
    cdef double[::1] scores

    def __init__(self, code, Decoder bcDecoder, lamb=4, mu=1./6, etaRel=4, **kwargs):
        BranchingRule.__init__(self, code, bcDecoder, lamb, mu)
        self.rpcRounds = kwargs.get('rpcRounds', 10)
        self.objBufLim = kwargs.get('objBufLim', .3)
        self.iterLimit = kwargs.get('iterLimit', 25)
        self.PsiMinus = 10*np.ones(code.blocklength, dtype=np.double)
        self.PsiPlus = 10*np.ones(code.blocklength, dtype=np.double)
        self.etaPlus = np.zeros(code.blocklength, dtype=np.intc)
        self.etaMinus = np.zeros(code.blocklength, dtype=np.intc)
        self.scores = np.zeros(code.blocklength, dtype=np.double)
        self.etaRel = etaRel

    def reset(self):
        self.etaPlus[:] = 0
        self.etaMinus[:] = 0
        self.PsiMinus[:] = 10
        self.PsiPlus[:] = 10


    cdef void updatePsi(self, int i, bint branch, double value):
        if branch:
            self.PsiPlus[i] = (self.PsiPlus[i]*self.etaPlus[i] + value) / (self.etaPlus[i] + 1)
            self.etaPlus[i] += 1
        else:
            self.PsiMinus[i] = (self.PsiMinus[i]*self.etaMinus[i] + value) / (self.etaMinus[i] + 1)
            self.etaMinus[i] += 1

    cdef int branchIndex(self, Node node, double ub, double[::1] solution) except -1:
        self.ub = ub
        self.node = node
        origLim = self.bcDecoder.lbProvider.objBufLim
        origRPC = self.bcDecoder.lbProvider.maxRPCrounds
        self.bcDecoder.lbProvider.objBufLim = self.objBufLim
        self.bcDecoder.lbProvider.maxRPCrounds = self.rpcRounds
        self.bcDecoder.lbProvider.model.setParam('IterationLimit', self.iterLimit)
        candidates = np.array([i for i in range(solution.size) if solution[i] > 1e-6 and solution[i] < 1-1e-6])
        for i in range(candidates.size):
            index = candidates[i]
            xi = solution[index]
            self.scores[i] = self.score(xi*self.PsiMinus[index], (1-xi)*self.PsiPlus[index])
        sortedByScore = np.argsort(self.scores[:candidates.size])[::-1]
        maxScore = -INFINITY # self.scores[sortedByScore[0]]
        maxIndex = -1 #candidates[sortedByScore[0]]
        itersSinceChange = 0
        for index in candidates:
            #xi = solution[index]
            #if min(self.etaPlus[index], self.etaMinus[index]) < self.etaRel:
                # unreliable score -> strong branch!
            deltaMinus, deltaPlus = self.strongBranchScore(index)
            # self.scores[i] = self.score(deltaMinus, deltaPlus)
            # self.updatePsi(index, 1, deltaPlus/(1-xi) if xi < .9999 else 0)  #TODO: perhaps don't update eta?
            # self.updatePsi(index, 0, deltaMinus/xi if xi > .0001 else 0)
            score = self.score(deltaMinus, deltaPlus)
            if score > maxScore:
                maxIndex = index
                maxScore = score
                itersSinceChange = 0
            else:
                itersSinceChange += 1
                if itersSinceChange >= self.lamb:
                    break

        self.bcDecoder.lbProvider.objBufLim = origLim
        self.bcDecoder.lbProvider.maxRPCrounds = origRPC
        self.bcDecoder.lbProvider.model.setParam('IterationLimit', INFINITY)
        node.fractionalPart = solution[maxIndex]
        if maxIndex == -1:
            raise ValueError()
        return maxIndex

    cdef (double, double) strongBranchScore(self, int index):
        cdef double deltaMinus, deltaPlus, objMinus, objPlus
        self.bcDecoder.lbProvider.fix(index, 0)
        self.bcDecoder.lbProvider.solve(-INFINITY, self.ub)
        objMinus = self.bcDecoder.lbProvider.objectiveValue
        self.bcDecoder.lbProvider.release(index)
        if self.bcDecoder.lbProvider.status in (Decoder.UPPER_BOUND_HIT, Decoder.INFEASIBLE):
            deltaMinus = self.ub - self.node.lpObj #INFINITY
        else:
            deltaMinus = self.bcDecoder.lbProvider.objectiveValue - self.node.lpObj
        self.bcDecoder.lbProvider.fix(index, 1)
        self.bcDecoder.lbProvider.solve(-INFINITY, self.ub)
        self.bcDecoder.lbProvider.release(index)
        objPlus = self.bcDecoder.lbProvider.objectiveValue
        if self.bcDecoder.lbProvider.status in (Decoder.UPPER_BOUND_HIT, Decoder.INFEASIBLE):
            deltaPlus = self.ub - self.node.lpObj #INFINITY
        else:
            deltaPlus = self.bcDecoder.lbProvider.objectiveValue - self.node.lpObj
        if fmin(objMinus, objPlus) > self.node.lb:
            # we can update the node's lower bound using the branching results!
            self.node.lb = fmin(objMinus, objPlus)
            if self.node.parent is not None:
                self.node.parent.updateBound(self.node.lb, self.node.branchValue)
        return deltaMinus, deltaPlus

    cdef int callback(self, Node node) except -1:
        if node.parent is not None:
            if self.bcDecoder.lbProvider.objectiveValue < node.parent.lpObj:
                return 0
            f = node.parent.fractionalPart
            if node.branchValue:
                f = 1 - f
            if f == 0:
                self.updatePsi(node.branchIndex, node.branchValue, 0)
            else:
                self.updatePsi(node.branchIndex, node.branchValue,
                               (self.bcDecoder.lbProvider.objectiveValue - node.parent.lpObj) / f)
