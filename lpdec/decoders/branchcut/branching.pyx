# -*- coding: utf-8 -*-
# Copyright 2015 Michael Helmling
# cython: language_level=3
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation
from __future__ import division

from collections import OrderedDict
from numpy.math cimport INFINITY
cimport numpy as np
import numpy as np
from libc.math cimport fabs, fmax, fmin
from lpdec.persistence cimport JSONDecodable
from lpdec.decoders.base cimport Decoder
from lpdec.decoders.branchcut.node cimport Node


cdef class BranchingRule(JSONDecodable):

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

    def params(self):
        ret = OrderedDict(lamb=self.lamb)
        ret['mu'] = self.mu
        return ret


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
    cdef double iterLimit, objBufLim, minCutoff
    cdef double[::1] sigmaPlus, sigmaMinus, psiPlus, psiMinus
    cdef int[::1] etaPlus, etaMinus
    cdef bint sort

    def __init__(self, code, Decoder bcDecoder, lamb=4, mu=1./6, etaRel=4, sort=False, **kwargs):
        BranchingRule.__init__(self, code, bcDecoder, lamb, mu)
        self.rpcRounds = kwargs.get('rpcRounds', 10)
        self.objBufLim = kwargs.get('objBufLim', .3)
        self.iterLimit = kwargs.get('iterLimit', 25)
        self.minCutoff = kwargs.get('minCutoff', .2)
        self.sigmaMinus = np.empty(code.blocklength, dtype=np.double)
        self.sigmaPlus = np.empty(code.blocklength, dtype=np.double)
        self.psiMinus = np.empty(code.blocklength, dtype=np.double)
        self.psiPlus = np.empty(code.blocklength, dtype=np.double)
        self.etaPlus = np.zeros(code.blocklength, dtype=np.intc)
        self.etaMinus = np.zeros(code.blocklength, dtype=np.intc)
        self.etaRel = etaRel
        self.sort = sort

    cdef int reset(self) except -1:
        self.etaPlus[:] = 0
        self.etaMinus[:] = 0
        self.psiPlus[:] = 1
        self.psiMinus[:] = 1
        self.sigmaMinus[:] = 0
        self.sigmaPlus[:] = 0
        for stat in 'deltaNeg', 'deltaPos', 'strongCount':
            if stat not in self.bcDecoder._stats:
                self.bcDecoder._stats[stat] = 0

    cdef void updatePsiPlus(self, int index, double sigmaLast):
        cdef int i, num = 0
        cdef double avg = 0
        self.sigmaPlus[index] += sigmaLast
        self.etaPlus[index] += 1
        self.psiPlus[index] = self.sigmaPlus[index] / self.etaPlus[index]
        for i in range(self.sigmaPlus.size):
            if self.etaPlus[i] != 0:
                num += 1
                avg += self.psiPlus[i]
        if num < self.sigmaPlus.size:
            avg /= num
            # there are uninitialized pseudocosts -> set them to average psi plus
            for i in range(self.sigmaPlus.size):
                if self.etaPlus[i] == 0:
                    self.psiPlus[i] = avg
                    
    cdef void updatePsiMinus(self, int index, double sigmaLast):
        cdef int i, num = 0
        cdef double avg = 0
        self.sigmaMinus[index] += sigmaLast
        self.etaMinus[index] += 1
        self.psiMinus[index] = self.sigmaMinus[index] / self.etaMinus[index]
        for i in range(self.sigmaMinus.size):
            if self.etaMinus[i] != 0:
                num += 1
                avg += self.psiMinus[i]
        if num < self.sigmaMinus.size:
            avg /= num
            # there are uninitialized pseudocosts -> set them to average psi plus
            for i in range(self.sigmaMinus.size):
                if self.etaMinus[i] == 0:
                    self.psiMinus[i] = avg

    cdef int branchIndex(self, Node node, double ub, double[::1] solution) except -1:
        self.ub = ub
        self.node = node
        origLim = self.bcDecoder.lbProvider.objBufLim
        origRPC = self.bcDecoder.lbProvider.maxRPCrounds
        origCut = self.bcDecoder.lbProvider.minCutoff
        self.bcDecoder.lbProvider.objBufLim = self.objBufLim
        self.bcDecoder.lbProvider.maxRPCrounds = self.rpcRounds
        self.bcDecoder.lbProvider.model.setParam('IterationLimit', self.iterLimit)
        self.bcDecoder.lbProvider.minCutoff = self.minCutoff
        candidates = np.array([i for i in range(solution.size) if solution[i] > 1e-6 and solution[i] < 1-1e-6])
        scores = np.array([self.score(solution[i]*self.psiMinus[i], (1-solution[i])*self.psiPlus[i])
                          for i in candidates])
        if self.sort:
            sortedByScore = np.argsort(scores)[::-1]
            candidates = candidates[sortedByScore]
            scores = scores[sortedByScore]
        maxIndex = candidates[0]
        maxScore = scores[0]
        itersSinceChange = 0
        for i in range(len(candidates)):
            index = candidates[i]
            if min(self.etaPlus[index], self.etaMinus[index]) < self.etaRel:
                # unreliable score -> strong branch!
                deltaMinus, deltaPlus = self.strongBranchScore(index)
                # self.updatePsiPlus(index, deltaPlus / (1 - solution[index]))
                # self.updatePsiMinus(index, deltaMinus/solution[index])
                # self.sigmaPlus[index] += deltaPlus/(1-xi)
                # self.etaPlus[index] += 1
                # self.sigmaMinus[index] += deltaMinus/xi
                # self.etaMinus[index] += 1
                score = self.score(deltaMinus, deltaPlus)
                scores[i] = score
            if scores[i] > maxScore:
                maxIndex = index
                maxScore = scores[i]
                itersSinceChange = 0
            else:
                itersSinceChange += 1
            if itersSinceChange >= self.lamb:
                break
        self.bcDecoder.lbProvider.objBufLim = origLim
        self.bcDecoder.lbProvider.maxRPCrounds = origRPC
        self.bcDecoder.lbProvider.model.setParam('IterationLimit', INFINITY)
        self.bcDecoder.lbProvider.minCutoff = origCut
        node.fractionalPart = solution[maxIndex]  # record f_i^+
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
        self.bcDecoder._stats['strongCount'] += 1
        return deltaMinus, deltaPlus

    cdef int callback(self, Node node) except -1:
        cdef double Delta
        if node.parent is not None:
            Delta = self.bcDecoder.lbProvider.objectiveValue - node.parent.lpObj
            if Delta < 0:
                self.bcDecoder._stats['deltaNeg'] += 1
            else:
                self.bcDecoder._stats['deltaPos'] += 1
            #     print('omg', Delta)
            #     return 0
            if node.branchValue:
                self.updatePsiPlus(node.branchIndex, Delta / (1 - node.parent.fractionalPart))
            else:
                self.updatePsiMinus(node.branchIndex, Delta / node.parent.fractionalPart)

    cpdef params(self):
        ret = BranchingRule.params(self)
        ret['etaRel'] = self.etaRel
        ret['sort'] = self.sort
        ret['rpcRounds'] = self.rpcRounds
        ret['objBufLim'] = self.objBufLim
        ret['iterLimit'] = self.iterLimit
        ret['minCutoff'] = self.minCutoff