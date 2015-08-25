# -*- coding: utf-8 -*-
# Copyright 2015 Michael Helmling
# cython: language_level=3
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation
from __future__ import division

from collections import OrderedDict
from collections import defaultdict
from numpy.math cimport INFINITY
cimport numpy as np
import numpy as np
from libc.math cimport fabs, fmax, fmin, sqrt, exp
from lpdec.persistence cimport JSONDecodable
from lpdec.decoders.base cimport Decoder
from lpdec.decoders.branchcut.node cimport Node


cdef class BranchingRule(JSONDecodable):
    """Base class for branching rules used in a branch-and-bound decoder.

    Parameters
    ----------
    code : :class:`lpdec.codes.BinaryLinearBlockCode`
        The code under consideration.
    bcDecoder : :class:`.BranchAndCutDecoder`
        The decoder instance.
    lamb : int, optional
    mu : int, optional

    Attributes
    ----------
    ub : float
    canPrune : bool
    index : int
    """
    def __init__(self, code, Decoder bcDecoder, int lamb=-1, mu=1./6):
        self.code = code
        self.lamb = lamb
        self.mu = mu
        self.bcDecoder = bcDecoder
        self.candInds = np.zeros(code.blocklength, dtype=np.intc)
        self.index = -1
        self.canPrune = False
        self.ub = INFINITY

    cdef int callback(self, Node node) except -1:
        pass

    cdef int rootCallback(self, int rounds, int iters) except -1:
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

    cdef int computeBranchIndex(self, Node node, double ub, double[::1] solution) except -1:
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
                maxIndex = index
                maxScore = score
                itersSinceBest = 0
            else:
                itersSinceBest += 1
            if self.lamb != -1 and itersSinceBest > self.lamb:
                break
        self.endScoreComputation()
        self.index = maxIndex

    cdef int beginScoreComputation(self) except -1:
        """Subclasses can put preparation for score computation in here."""
        pass

    cdef int endScoreComputation(self) except -1:
        """Subclasses can put cleanup operations after score computations in here."""
        pass

    def params(self):
        ret = OrderedDict()
        if self.lamb != -1:
            ret['lamb'] = self.lamb
        if self.mu != 1./6:
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

    cdef int computeBranchIndex(self, Node node, double ub, double[::1] solution) except -1:
        cdef int i, index = -1
        for i in range(self.code.blocklength):
            if not self.bcDecoder.fixed(i):
                if index == -1:
                    index = i  # fallback if no entries are fractional
                if not self.onlyFractional or fabs(solution[i] - .5) < .4999:
                    self.index = i
                    return 0
        self.index = index


cdef class ReliabilityBranching(BranchingRule):

    cdef int etaRel
    cdef double deltaMinus, deltaPlus, objMinus, objPlus
    cdef double[::1] sigmaPlus, sigmaMinus, psiPlus, psiMinus, bs0, bs1
    cdef int[::1] etaPlus, etaMinus
    cdef bint sort, updateInStrong, updateInCallback, initStrong
    cdef public double[::1] codeword
    cdef Decoder lbProvider

    def __init__(self, code, Decoder bcDecoder, lamb=4, mu=1./6, etaRel=4, sort=False, **kwargs):
        BranchingRule.__init__(self, code, bcDecoder, lamb, mu)
        self.updateInCallback = kwargs.get('updateInCallback', True)
        self.updateInStrong = kwargs.get('updateInStrong', False)
        self.initStrong = kwargs.get('initStrong', True)
        self.sigmaMinus = np.empty(code.blocklength, dtype=np.double)
        self.sigmaPlus = np.empty(code.blocklength, dtype=np.double)
        self.psiMinus = np.empty(code.blocklength, dtype=np.double)
        self.psiPlus = np.empty(code.blocklength, dtype=np.double)
        self.etaPlus = np.zeros(code.blocklength, dtype=np.intc)
        self.etaMinus = np.zeros(code.blocklength, dtype=np.intc)
        self.etaRel = etaRel
        self.codeword = None
        self.lbProvider = bcDecoder.lbProvider
        self.bs0 = np.empty(code.blocklength, dtype=np.double)
        self.bs1 = np.empty(code.blocklength, dtype=np.double)
        self.sort = sort

    cdef int reset(self) except -1:
        self.etaPlus[:] = 0
        self.etaMinus[:] = 0
        self.psiPlus[:] = 1
        self.psiMinus[:] = 1
        self.sigmaMinus[:] = 0
        self.sigmaPlus[:] = 0
        for stat in 'deltaNeg', 'deltaPos', 'strongCount', 'brStopLim':
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

    cdef int updBranchLb(self, Node node, int index, double lb0, double lb1):
        if node.branchLb is None:
            node.branchLb = defaultdict(lambda: -INFINITY)
        node.branchLb[index, 0] = lb0
        node.branchLb[index, 1] = lb1
    
    cdef int computeBranchIndex(self, Node node, double ub, double[::1] solution) except -1:
        self.ub = ub
        self.node = node
        self.canPrune = False
        self.index = -1
        self.codeword = None
        candidates = np.array([i for i in range(solution.size) if solution[i] > 1e-6 and solution[i] < 1-1e-6])
        if len(candidates) == 0:
            for i in range(solution.size):
                if not self.bcDecoder.fixed(i):
                    self.index = i
                    return 0
            self.canPrune = True
            return 0
        itersSinceChange = 0
        scores = np.array([self.score(solution[i]*self.psiMinus[i], (1-solution[i])*self.psiPlus[i])
                          for i in candidates])

        computedThisRound = np.zeros(len(candidates))
        if self.initStrong:
            initBestScore = -INFINITY
            for i in range(len(candidates)):
                index = candidates[i]
                if self.etaPlus[index] == 0 or self.etaMinus[index] == 0:
                    self.strongBranchScore(index)
                    self.updatePsiPlus(index, self.deltaPlus / (1 - solution[index]))
                    self.updatePsiMinus(index, self.deltaMinus / solution[index])
                    scores[i] = self.score(self.deltaMinus, self.deltaPlus)
                    computedThisRound[i] = 1
                    if self.canPrune:
                        break
        if not self.canPrune:
            if self.sort:
                sortedByScore = np.argsort(scores)[::-1]
                candidates = candidates[sortedByScore]
                scores = scores[sortedByScore]
            self.index = candidates[0]
            maxScore = scores[0]
            for i in range(len(candidates)):
                index = candidates[i]
                if min(self.etaPlus[index], self.etaMinus[index]) < self.etaRel and not computedThisRound[i]:
                    # unreliable score -> strong branch!
                    self.strongBranchScore(index)
                    if self.canPrune:
                        break
                    if self.updateInStrong:
                        self.updatePsiPlus(index, self.deltaPlus / (1 - solution[index]))
                        self.updatePsiMinus(index, self.deltaMinus/solution[index])
                    score = self.score(self.deltaMinus, self.deltaPlus)
                    scores[i] = score
                if scores[i] > maxScore:
                    self.index = index
                    maxScore = scores[i]
                    itersSinceChange = 0
                else:
                    itersSinceChange += 1
                if self.lamb != -1 and itersSinceChange >= self.lamb:# and node.depth > 0:
                    break
        node.fractionalPart = solution[self.index]  # record f_i^+
        if self.lbProvider.fixed(self.index):
            # this happens if the current index is implicitly fixed
            for i in range(solution.size):
                if not self.lbProvider.fixed(i):
                    self.index = i
                    return 0
            self.canPrune = True
            self.index = -1
            return 0

    cdef int strongBranchScore(self, int index) except -1:
        cdef double deltaMinus, deltaPlus, objMinus, objPlus
        self.lbProvider.fix(index, 0)
        self.lbProvider.solve(self.node.lb, self.ub)
        objMinus = self.lbProvider.objectiveValue
        self.lbProvider.release(index)
        if self.lbProvider.status == Decoder.UPPER_BOUND_HIT:
            deltaMinus = objMinus - self.node.lpObj
        elif self.lbProvider.status ==  Decoder.INFEASIBLE:
            deltaMinus = objMinus - self.node.lpObj
        else:
            deltaMinus = objMinus - self.node.lpObj
            if self.lbProvider.foundCodeword and objMinus < self.ub:
                self.codeword = self.lbProvider.solution.copy()
                self.ub = objMinus
        if self.lbProvider.status == Decoder.LIMIT_HIT:
            self.bcDecoder._stats['brStopLim'] += 1
        self.bs0[:] = self.lbProvider.solution[:]
        self.lbProvider.fix(index, 1)
        self.lbProvider.solve(self.node.lb, self.ub)
        self.lbProvider.release(index)
        objPlus = self.lbProvider.objectiveValue
        if self.lbProvider.status == Decoder.UPPER_BOUND_HIT:
            deltaPlus = objPlus - self.node.lpObj
        elif self.lbProvider.status == Decoder.INFEASIBLE:
            deltaPlus = objPlus - self.node.lpObj
        else:
            deltaPlus = objPlus - self.node.lpObj
            if self.lbProvider.foundCodeword and objPlus < self.ub:
                self.codeword = self.lbProvider.solution.copy()
                self.ub = objPlus
        if fmin(objMinus, objPlus) > self.node.lb:
            # we can update the node's lower bound using the branching results!
            self.node.lb = fmin(objMinus, objPlus)
            if self.node.parent is not None:
                self.node.parent.updateBound(self.node.lb, self.node.branchValue)
        if self.lbProvider.status == Decoder.LIMIT_HIT:
            self.bcDecoder._stats['brStopLim'] += 1
        self.bs1[:] = self.lbProvider.solution[:]
        if objMinus > self.ub - 1e-6 and objPlus > self.ub - 1e-6:
            self.canPrune = True
        else:
            if objMinus > self.ub - 1e-6:
                self.node.implicitFixes.append((index, 1))
                self.bcDecoder.fix(index, 1)
            elif objPlus > self.ub - 1e-6:
                self.node.implicitFixes.append((index, 0))
                self.bcDecoder.fix(index, 0)
            self.updBranchLb(self.node, index, objMinus, objPlus)
        self.bcDecoder._stats['strongCount'] += 1

    cdef int callback(self, Node node) except -1:
        cdef double Delta
        if not self.updateInCallback:
            return 0
        if node.depth > 0:
            # if self.lbProvider.objectiveValue > self.ub - 1e-6:
            #     print('ua')
            #     return 0
            if node.parent.fractionalPart == 0 or node.parent.fractionalPart == 1:
                return 0
            Delta = node.lpObj - node.parent.lpObj
            if Delta < 0:
                self.bcDecoder._stats['deltaNeg'] += 1
                #delta = 0
            else:
                self.bcDecoder._stats['deltaPos'] += 1
            if node.branchValue == 1:
                self.updatePsiPlus(node.branchIndex, Delta / (1 - node.parent.fractionalPart))
            else:
                self.updatePsiMinus(node.branchIndex, Delta / node.parent.fractionalPart)

    cpdef params(self):
        ret = BranchingRule.params(self)
        ret['etaRel'] = self.etaRel
        if not self.sort:
            ret['sort'] = self.sort
        #ret['objBufLim'] = self.objBufLim
        #ret['iterLimit'] = self.iterLimit
        #ret['minCutoff'] = self.minCutoff
        if not self.updateInCallback:
            ret['updateInCallback'] = False
        if self.updateInStrong:
            ret['updateInStrong'] = True
        if not self.initStrong:
            ret['initStrong'] = False
        return ret
