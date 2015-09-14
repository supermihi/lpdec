# -*- coding: utf-8 -*-
# Copyright 2014-2015 Michael Helmling
# cython: cdivision=True
# cython: wraparound=True
# cython: language_level=3
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation

from __future__ import division, print_function

import logging
import itertools
from collections import OrderedDict

import numpy as np
cimport numpy as np
from numpy.math cimport INFINITY

from lpdec.decoders.base cimport Decoder
from lpdec.decoders.adaptivelp_glpk import AdaptiveLPDecoder
from lpdec.decoders.iterative import IterativeDecoder
from lpdec.decoders.branchcut.node cimport Node, move
from lpdec.decoders.branchcut.branching cimport BranchingRule
from lpdec.decoders.branchcut.branching import MostFractional
from lpdec import utils, persistence

logger = logging.getLogger(name='b&c')


cdef enum SelectionMethod:
    # Node selection methods supported by this class. Used internally for speed reasons
    mixed, dfs, bbs


cdef class BranchAndCutDecoder(Decoder):
    """
    Maximum-Likelihood decoder using a branch-and-cut approach.

    Parameters
    ----------
    selectionMethod : {'dfs', 'bbs', 'mixed<steps>/<gap>'}
        Method to determine the next node from the set of active nodes. Possible values are:

        - ``'dfs'``: Depth-first search
        - ``'bbs'``: Best-bound search
        - ``'mixed<steps>/<gap>'``: Mixed strategy. Uses depth-first search in general but
          jumps to the node with smallest lower bound every `<steps>` iterations, but only if the
          duality gap at the current node is at least `<gap>`.

    maxDecay : float
    maxDecayDepthFactor : float
    dfsDepthFactor : float

    childOrder : {'01', '10', 'llr', 'random'}
        Determines the order in which newly created child branch-and-bound nodes are appended to the
        list of active nodes. Possible values are:

        - ``'01'``: child with zero-fix is added first
        - ``'10'``: child with one-fix is added first
        - ``'llr'``: add first the node whose fix-value equals the hard-decision value of the fixed bit.
        - ``'random'``: add in random order

        The default value is ``'01'``.

    highSNR : bool
        Use optimizations for high SNR values. Defaults to `False`.

    lpClass : type
        Class of the LP decoder. Default: :class:`.AdaptiveLPDecoder`.
    lpParams : dict
        Parameters for the LP decoder
    iterClass : type
        Class of the upper bound provider. Default: :class:`.IterativeDecoder`.
    iterParams : dict
        Parameters for the iterative decoder.
    initOrder : int, optional
        When using an iterative decoder, this number specifies the re-encoding order used at the
        initial (root) node. Higher value increases computation time but might lead to a good starting
        solution.
    branchClass : type
        Class of branching rule to use; see also :mod:`.branching`.
    branchParams : dict
        Parameters for instantiating the branching rule.
    """
    cdef:
        public Decoder lbProvider, ubProvider
        public list activeNodes

        object timer
        SelectionMethod selectionMethod
        BranchingRule branchRule
        Node root, bestBoundNode

        bint runUbProviderNextIter, highSNRMode, minDistance
        bytes childOrder
        int mixParam, maxDecayDepth, initialReencodeOrder, originalReencodeOrder, dfsStepCounter
        double mixGap, sentObjective, objBufLimOrig, cutoffOrig, cutDecayFactor, bufDecayFactor
        double maxDecay, maxDecayDepthFactor, dfsDepthFactor, ub


    def __init__(self, code, name='BranchAndCutDecoder', **kwargs):
        self.code = code
        self._createProviders(**kwargs)
        self._createBranchRule(**kwargs)
        self._parseChildOrder(**kwargs)
        self._parseSelectionMethod(**kwargs)
        self.highSNRMode = kwargs.get('highSNR', False)
        self.activeNodes = []
        Decoder.__init__(self, code, name=name)

        self.timer = utils.Timer()
        self.bestBoundNode = None
        self.initialReencodeOrder = kwargs.get('initOrder', 0)
        if self.initialReencodeOrder != 0:
            self.originalReencodeOrder = self.ubProvider.reencodeOrder


    def _parseSelectionMethod(self, **kwargs):
        """Configures selection method based on given keyword arguments."""
        selectionMethod = kwargs.get('selectionMethod', 'dfs')
        if selectionMethod.startswith('mixed'):
            self.selectionMethod = mixed
            mixParamStr, mixGapStr = selectionMethod[len('mixed'):].split('/')
            self.mixParam = int(mixParamStr)
            self.mixGap = float(mixGapStr)
            self.cutoffOrig = self.lbProvider.minCutoff
            self.objBufLimOrig = self.lbProvider.objBufLim
            self.maxDecay = kwargs.get('maxDecay', 4.0)
            self.maxDecayDepthFactor = kwargs.get('maxDecayDepthFactor', 2.0)
            maxDecayDepth = int((self.code.blocklength - self.code.infolength) / self.maxDecayDepthFactor)
            self.bufDecayFactor = (self.objBufLimOrig / self.maxDecay - 0.001) / maxDecayDepth
            self.cutDecayFactor = (self.cutoffOrig / self.maxDecay - 1e-5) / maxDecayDepth
            self.maxDecayDepth = maxDecayDepth
            self.dfsDepthFactor = kwargs.get('dfsDepthFactor', 10)
        elif selectionMethod == 'dfs':
            self.selectionMethod = dfs
        else:
            assert selectionMethod == 'bbs'
            self.selectionMethod = bbs

    def _createProviders(self, **kwargs):
        """Initializes the lower and upper bound providers (usually LP and iterative decoder,
        respectively).
        """
        lpParams = kwargs.get('lpParams', dict(keepCuts=True, removeInactive=300, minCutoff=0.15,
                                               objBufLim=0.2, objBufSize=6))
        iterParams = kwargs.get('iterParams', {})
        lpClass = kwargs.get('lpClass', AdaptiveLPDecoder)
        if isinstance(lpClass, basestring):
            lpClass = persistence.classByName(lpClass)
        self.lbProvider = lpClass(self.code, **lpParams)
        iterClass = kwargs.get('iterClass', IterativeDecoder)
        if isinstance(iterClass, basestring):
            iterClass = persistence.classByName(iterClass)
        self.ubProvider = iterClass(self.code, **iterParams)

    def _createBranchRule(self, **kwargs):
        """Initializes the branch method."""
        branchParams = kwargs.get('branchParams', {})
        branchClass = kwargs.get('branchClass', MostFractional)
        if isinstance(branchClass, basestring):
            branchClass = persistence.classByName(branchClass)
        self.branchRule = branchClass(self.code, self, **branchParams)

    def _parseChildOrder(self, **kwargs):
        """Configures child order processing based on keyword args."""
        childOrder = kwargs.get('childOrder', b'01')
        if isinstance(childOrder, unicode):
            self.childOrder = childOrder.encode('utf8')
        else:
            self.childOrder = childOrder
        assert self.childOrder in (b'01', b'10', b'llr', b'random')

    def setStats(self, stats):
        for item in 'nodes', 'prBd1', 'prBd2', 'prInf', 'prBranch', 'prOpt', 'termEx', 'termGap',\
                    'termSent', 'lpTime', 'iterTime', 'maxDepth', 'branchTime', 'initUbOpt':
            if item not in stats:
                stats[item] = 0
        if 'lpStats' in stats:
            self.lbProvider.setStats(stats['lpStats'])
            del stats['lpStats']
        else:
            self.lbProvider.setStats(dict())
        if 'iterStats' in stats:
            self.ubProvider.setStats(stats['iterStats'])
            del stats['iterStats']
        else:
            self.ubProvider.setStats(dict())
        Decoder.setStats(self, stats)

    def stats(self):
        stats = self._stats.copy()
        # insert stats from lb/ub providers
        stats['lpStats'] = self.lbProvider.stats().copy()
        stats['iterStats'] = self.ubProvider.stats().copy()
        return stats

    cpdef setLLRs(self, double[::1] llrs, np.int_t[::1] sent=None):
        """Set LLRs for decoding; implements abstract method of :class:`.Decoder`.

        If the `sent` codeword is given, it is used as an initial candidate solution and further to
        shortcut the decoding whenever a better (i.e. smaller-objective) candidate than the sent one
        was found.

        Except when ``highSNR`` was set in the constructor, this will also run an initial ubProvider
        decoding.
        """
        cdef int i
        self.ubProvider.setLLRs(llrs, sent)
        if sent is not None:
            self.ub = self.sentObjective = np.dot(sent, llrs)
            for i in range(sent.size):
                self.solution[i] = sent[i]
        else:
            self.sentObjective = -INFINITY
            self.ub = 1
        if self.highSNRMode:
            self.ubProvider.foundCodeword = self.ubProvider.mlCertificate = False
        else:
            self.timer.start()
            if self.initialReencodeOrder != 0:
                self.ubProvider.reencodeOrder = self.initialReencodeOrder
            self.ubProvider.solve()
            if self.initialReencodeOrder != 0:
                self.ubProvider.reencodeOrder = self.originalReencodeOrder
            self._stats['iterTime'] += self.timer.stop()
            if self.ubProvider.foundCodeword:
                if self.ubProvider.objectiveValue < self.sentObjective:
                    self.objectiveValue = self.ubProvider.objectiveValue
                else:
                    self.solution[:] = self.ubProvider.solution[:]
                    self.objectiveValue = self.ub = self.ubProvider.objectiveValue
        self.lbProvider.setLLRs(llrs, sent)
        Decoder.setLLRs(self, llrs)

    cpdef fix(self, int index, int value):
        self.lbProvider.fix(index, value)
        self.ubProvider.fix(index, value)

    cpdef release(self, int index):
        self.lbProvider.release(index)
        self.ubProvider.release(index)

    cpdef fixed(self, int index):
        return self.lbProvider.fixed(index)

    cdef Node selectBestBoundNode(self):
        """Return the node in :attr:`activeNodes` with minimum lower bound."""
        cdef int i, minIndex = -1
        cdef double minValue = INFINITY
        for i in range(len(self.activeNodes)):
            if self.activeNodes[i].lb < minValue:
                minIndex = i
                minValue = self.activeNodes[i].lb
        if minValue > self.root.lb:
            raise StopIteration()
        return self.activeNodes.pop(minIndex)

    cdef Node selectNode(self, Node currentNode, double ub):
        """Select the next branch-and-bound node according to the configured selection method.

        This method checks that the node's lower bound is below the given upper bound `ub`.

        Raises
        ------
        StopIteration
            When active node list is empty.
        """
        cdef Node node
        while True:
            if len(self.activeNodes) == 0:
                raise StopIteration
            if self.selectionMethod == mixed:
                node = self.selectNodeMixed(currentNode, ub)
            elif self.selectionMethod == dfs:
                node = self.activeNodes.pop()
            elif self.selectionMethod == bbs:
                node = self.selectBestBoundNode()
            if node.lb < ub - 1e-6:
                return node

    cdef Node selectNodeMixed(self, Node currentNode, double ub):
        """Select a current node using "mixed" strategy.

        This strategy performs either a best-bound or a depth-first selection, based on the
        following set of rules:

        - If `ub` - ``currentNode.lb`` <= :attr:`mixGap`, always do depth-first
        - Also, if the potential depth-first node is not a descendant of the *last* best-bound node,
          do best-bound.
        - If more than ``n`` depth-first steps have been performed, do best-bound, where ``n`` is
          determined by :attr:`mixedParam`, multiplied by a factor depending on the last best-bound
          node's depth and :attr:`dfsDepthFactor`.
        - In min-distance computation, we do best-bound as long as the root node's lower bound is 1

        Besides that, other behavior of the decoder depends on which mode was chosen:

        - ubProvider is run only after best-bound steps
        - in depth-first steps, objBufLim and minCutoff of :attr:`lbProvider` are increased, based
          on the node's depth, :attr:`maxDecayDepth` and :attr:`bufDecayFactor` / :attr:`cutDecayFactor`
        """
        cdef bint bestBoundStep = False
        if ub - currentNode.lb > self.mixGap:
            if self.dfsStepCounter >= self.mixParam*(1+self.bestBoundNode.depth/self.dfsDepthFactor):
                bestBoundStep = True
            elif self.minDistance and self.root.lb == 1:
                bestBoundStep = True
        if not (self.bestBoundNode is None or self.activeNodes[-1].isDescendantOf(self.bestBoundNode)):
            bestBoundStep = True
        if bestBoundStep:
            self.bestBoundNode = self.selectBestBoundNode()
            self.dfsStepCounter = 1
            # set special DFS rules to lbProvider
            self.lbProvider.objBufLim = min(self.maxDecayDepth, self.bestBoundNode.depth)*self.bufDecayFactor + 0.001
            self.lbProvider.minCutoff = min(self.maxDecayDepth, self.bestBoundNode.depth)*self.cutDecayFactor + 1e-5
            self.runUbProviderNextIter = True
            return self.bestBoundNode
        else:
            # reset normal lbProvider behavior
            self.lbProvider.objBufLim = self.objBufLimOrig
            self.lbProvider.minCutoff = self.cutoffOrig
            self.dfsStepCounter += 1
            self.runUbProviderNextIter = False
            return self.activeNodes.pop()

    cpdef solve(self, double lb=-INFINITY, double ub=INFINITY):
        cdef:
            Node currentNode, newNode
            int iteration = 0
            bint initialCandidateIsOptimal = True  # statistical use only
        self.activeNodes = []
        ub = self.ub
        if ub < self.sentObjective:
            return
        self.branchRule.reset()
        self.foundCodeword = self.mlCertificate = True
        self.root = currentNode = Node()
        self.runUbProviderNextIter = True
        self.dfsStepCounter = 0
        self._stats['nodes'] += 1
        if self.selectionMethod == mixed:
            self.bestBoundNode = self.root
            # set strong values for root node processing for root node
            self.lbProvider.objBufLim = 0.001
            self.lbProvider.minCutoff = 1e-5

        while True:
            iteration += 1
            if currentNode.depth > self._stats['maxDepth']:
                self._stats['maxDepth'] = currentNode.depth
            # print('{}/{}, d {}, it {}, n {}, lp {:6f}, heu {:6f} bra {:6f}'.format(
            #     self.root.lb, ub,
            #     node.depth, iteration, len(activeNodes), self._stats["lpTime"], self._stats['iterTime'], self._stats['branchTime']))

            # ========== upper bound calculation ==================================
            if iteration > 1 and self.runUbProviderNextIter:
                self.timer.start()
                self.ubProvider.solve()
                self._stats['iterTime'] += self.timer.stop()
            if self.ubProvider.foundCodeword and self.ubProvider.objectiveValue < ub:
                self.solution[:] = self.ubProvider.solution[:]
                ub = self.ubProvider.objectiveValue
                if iteration > 1:
                    initialCandidateIsOptimal = False
                if ub < self.sentObjective  - 1e-5:
                    self.mlCertificate = False
                    self._stats['termSent'] += 1
                    break

            # ========== lower bound calculation ==================================
            self.timer.start()
            self.lbProvider.solve(currentNode.lb, ub)
            currentNode.lpObj = self.lbProvider.objectiveValue
            self._stats['lpTime'] += self.timer.stop()
            if self.lbProvider.status == Decoder.UPPER_BOUND_HIT:
                currentNode.lb = INFINITY
                self._stats['prBd2'] += 1
            elif self.lbProvider.status == Decoder.INFEASIBLE:
                currentNode.lb = INFINITY
                self._stats['prInf'] += 1
            elif self.lbProvider.objectiveValue > currentNode.lb:
                currentNode.lb = self.lbProvider.objectiveValue
            self.branchRule.callback(currentNode)

            # =========== pruning or branching =====================================
            if self.lbProvider.foundCodeword:
                # prune by optimality
                self._stats['prOpt'] += 1
                if self.lbProvider.objectiveValue < ub:
                    self.solution[:] = self.lbProvider.solution[:]
                    ub = self.lbProvider.objectiveValue
                    initialCandidateIsOptimal = False
                    if ub < self.sentObjective - 1e-5:
                        self.mlCertificate = False
                        break
            elif currentNode.lb < ub-1e-6:
                # branch
                self.timer.start()
                self.branchRule.computeBranchIndex(currentNode, ub, self.lbProvider.solution.copy())
                self._stats['branchTime'] += self.timer.stop()
                if self.branchRule.ub < ub:
                    self.solution = self.branchRule.codeword.copy()
                    ub = self.branchRule.ub
                    print('new codeword from branching LP')
                if self.branchRule.canPrune or currentNode.lb >= ub - 1e-6:
                    self._stats['prBranch'] += 1
                elif self.branchRule.index >= 0:
                    self.activeNodes.extend(currentNode.branch(self.branchRule.index, self.childOrder, self, ub))
                # no branch index found -> all indices fixed, skip node! (implies infeasibiliy
                # that wasn't detected by the LP decoder due to cutoff bounds)

            # =========== bound update ===================================================
            if currentNode.parent is not None:
                currentNode.parent.updateBound(currentNode.lb, currentNode.branchValue)
                if self.root.lb >= ub - 1e-6:
                    self._stats['termGap'] += 1
                    break
            if len(self.activeNodes) == 0:
                self._stats['termEx'] += 1
                break

            # =========== new node selection ==============================================
            try:
                newNode = self.selectNode(currentNode, ub)
            except StopIteration:
                break
            move(self.lbProvider, self.ubProvider, currentNode, newNode)
            currentNode = newNode
        self.objectiveValue = ub
        if initialCandidateIsOptimal:
            self._stats['initUbOpt'] += 1
        if self.selectionMethod == mixed:
            self.lbProvider.objBufLim = self.objBufLimOrig
            self.lbProvider.minCutoff = self.cutoffOrig
        for i in range(self.code.blocklength):
            self.release(i)
            self.release(i)


    def minimumDistance(self, randomized=True, cyclic=False):
        """Compute the minimum distance of the code."""
        # switch to min distance computation mode
        self.ubProvider.excludeZero = self.minDistance = True
        llrs = np.ones(self.code.blocklength, dtype=np.double)

        if randomized:
            # add small random values to the llrs, to enforce a unique solution
            delta = 0.001
            epsilon = delta/self.code.blocklength
            np.random.seed(239847)
            llrs += epsilon*np.random.random_sample(self.code.blocklength)
        else:
            delta = 1e-5
        for i in range(cyclic): # fix bits to one for cyclic codes (or other symmetries)
            self.fix(i, 1)
        self.setLLRs(llrs)
        self.dfsStepCounter = 1
        self.root = node = Node()
        self.root.lb = 1
        if self.selectionMethod == mixed:
            self.bestBoundNode = self.root
            # set strong values for root node processing for root node
            self.lbProvider.objBufLim = 0.001
            self.lbProvider.minCutoff = 1e-5
        self.activeNodes = []
        self.branchRule.reset()
        ub = INFINITY
        self._stats['nodes'] += 1
        for iteration in itertools.count(start=1):
            # statistic collection and debug output
            print('MD {}/{}, d {}, n {}, it {}, lp {}, heu {} bra {}'.format(
                self.root.lb,ub, node.depth,len(self.activeNodes), iteration,
                self._stats["lpTime"], self._stats['iterTime'], self._stats['branchTime']))
            pruned = False # store if current node can be pruned
            if node.lb >= ub-1+delta:
                node.lb = INFINITY
                pruned = True
            if not pruned:
                # upper bound calculation

                if iteration > 1 and self.runUbProviderNextIter:
                    # for first iteration this was done in setLLR
                    self.timer.start()
                    self.ubProvider.solve()
                    self._stats['iterTime'] += self.timer.stop()

                if self.ubProvider.foundCodeword and self.ubProvider.objectiveValue < ub:
                    self.solution[:] = self.ubProvider.solution[:]
                    ub = self.ubProvider.objectiveValue
                # lower bound calculation
                self.timer.start()

                if (iteration == 1 or self.runUbProviderNextIter) and self.ubProvider.foundCodeword:
                    self.lbProvider.hint = np.asarray(self.ubProvider.solution).astype(np.int)
                else:
                    self.lbProvider.hint = None
                self.lbProvider.solve(-INFINITY, ub - 1 + delta)
                node.lpObj = self.lbProvider.objectiveValue
                self._stats['lpTime'] += self.timer.stop()
                if self.lbProvider.status == Decoder.UPPER_BOUND_HIT:
                    node.lb = INFINITY
                    self._stats['prBd2'] += 1
                elif self.lbProvider.status == Decoder.INFEASIBLE:
                    node.lb = INFINITY
                    self._stats['prInf'] += 1
                elif self.lbProvider.objectiveValue > node.lb:
                    node.lb = self.lbProvider.objectiveValue
                self.branchRule.callback(node)
                if node.lb == INFINITY:
                    self._stats['prInf'] += 1
                elif self.lbProvider.foundCodeword and self.lbProvider.objectiveValue > .5:
                    # solution is integral
                    if self.lbProvider.objectiveValue < ub:
                        self.solution[:] = self.lbProvider.solution[:]
                        ub = self.lbProvider.objectiveValue
                        self._stats['prOpt'] += 1
                elif node.lb < ub-1+delta:
                    # branch
                    self.timer.start()
                    self.branchRule.computeBranchIndex(node, ub, self.lbProvider.solution.copy())
                    self._stats['branchTime'] += self.timer.stop()
                    if self.branchRule.ub < ub:
                        self.solution = self.branchRule.codeword.copy()
                        ub = self.branchRule.ub
                        logger.info('found new candidate in branching rule')
                    if self.branchRule.canPrune or node.lb >= ub - 1 + delta:
                        node.lb = INFINITY
                        self._stats['prBranch'] += 1
                    else:
                        branchIndex = self.branchRule.index
                        if branchIndex == -1:
                            node.lb = INFINITY
                            print('********** PRUNE 000000 ***************')
                        else:
                            assert not self.fixed(branchIndex), '{} {} {}'.format(self, self.branchRule, branchIndex)
                            self.activeNodes.extend(node.branch(branchIndex, self.childOrder, self, ub))
                else:
                    self._stats["prBd2"] += 1
            if node.parent is not None:
                node.parent.updateBound(node.lb, node.branchValue)
                if self.root.lb >= ub - 1 + delta:
                    self._stats["termGap"] += 1
                    break
            if len(self.activeNodes) == 0:
                self._stats["termEx"] += 1
                break
            try:
                newNode = self.selectNode(node, ub)
            except StopIteration:
                break
            move(self.lbProvider, self.ubProvider, node, newNode)
            node = newNode

        assert self.solution in self.code
        self.objectiveValue = np.rint(ub)
        print('Minimum distance of {} computed successfully'.format(self.code))
        print('Minimum codeword is:')
        print(np.asarray(self.solution))
        print('MininumDistance is: {}'.format(self.objectiveValue))
        # restore normal (decoding) mode
        self.ubProvider.excludeZero = self.minDistance = False
        if self.selectionMethod == mixed:
            self.lbProvider.objBufLim = self.objBufLimOrig
            self.lbProvider.minCutoff = self.cutoffOrig
        for i in range(self.code.blocklength):
            self.release(i)
            self.release(i)
        return self.objectiveValue

    def params(self):
        if self.selectionMethod == mixed:
            method = "mixed{}/{}".format(self.mixParam, self.mixGap)
        else:
            selectionMethodNames = { dfs: "dfs", bbs: "bbs"}
            method = selectionMethodNames[self.selectionMethod]
        parms = OrderedDict()
        parms['branchClass'] = type(self.branchRule).__name__
        parms['branchParams'] = self.branchRule.params()
        parms['selectionMethod'] = method
        if self.childOrder != b'01':
            parms['childOrder'] = self.childOrder.decode('utf8')
        if type(self.lbProvider) is not AdaptiveLPDecoder:
            parms['lpClass'] = type(self.lbProvider).__name__
        parms['lpParams'] = self.lbProvider.params()
        if type(self.ubProvider) is not IterativeDecoder:
            parms['iterClass'] = type(self.ubProvider).__name__
        parms['iterParams'] = self.ubProvider.params()
        if self.highSNRMode:
            parms['highSNR'] = True
        if self.initialReencodeOrder != 0:
            parms['initOrder'] = self.initialReencodeOrder
        if self.maxDecay != 4.0:
            parms['maxDecay'] = self.maxDecay
        if self.maxDecayDepthFactor != 2:
            parms['maxDecayDepthFactor'] = self.maxDecayDepthFactor
        if self.dfsDepthFactor != 10.0:
            parms['dfsDepthFactor'] = self.dfsDepthFactor
        parms['name'] = self.name
        return parms
