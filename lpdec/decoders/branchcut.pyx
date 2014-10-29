# -*- coding: utf-8 -*-
# Copyright 2014 Michael Helmling
# cython: embedsignature=True
# cython: boundscheck=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: wraparound=False
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation

from __future__ import division, print_function, unicode_literals

import logging
import itertools
import random
from collections import OrderedDict

from libc.math cimport fabs, fmin
import numpy as np
cimport numpy as np

from lpdec.decoders import Decoder
from lpdec.decoders cimport Decoder
from lpdec.decoders.adaptivelp import AdaptiveLPDecoder
from lpdec.decoders.iterative import IterativeDecoder
from lpdec.utils import Timer

cdef double inf = np.inf
logger = logging.getLogger(name='branchcut')

cdef enum BranchMethod:
    mostFractional, leastReliable, eiriksPaper


cdef enum SelectionMethod:
    mixed, dfs, bbs, bfs


cdef void move(Decoder lbProv, Decoder ubProv, Node node, Node newNode):
    """Moves from one brach-and-bound node to another, updating variable fixes in the upper and
    lower bound providers on the fly.
    """
    cdef list fix = []
    while node.depth > newNode.depth:
        lbProv.release(node.branchIndex)
        ubProv.release(node.branchIndex)
        node = node.parent

    while newNode.depth > node.depth:
        fix.append( (newNode.branchIndex, newNode.branchValue) )
        newNode = newNode.parent
    while node is not newNode:
        lbProv.release(node.branchIndex)
        ubProv.release(node.branchIndex)
        fix.append( (newNode.branchIndex, newNode.branchValue) )
        node = node.parent
        newNode = newNode.parent
    for var, value in fix:
        lbProv.fix(var, value)
        ubProv.fix(var, value)


cdef class Node:
    """
    A node in the branch-and-bound tree.

    .. attribute:: parent

      pointer to the parent node

    .. attribute:: branchIndex

      index of the variable on which is branched

    .. attribute:: branchValue

      value of the branched variable (0 or 1)

    .. attribute:: depth

      depth in the tree (root has depth=0)

    .. attribute:: lb

      current local lower bound of the tree below this node

    .. attribute:: lbChild0, lbChild1

      lower bounds from left and right child, respectively. Used for bound updates.
    """
    def __init__(self, **kwargs):
        self.parent = kwargs.get("parent", None)
        self.branchIndex = kwargs.get("branchIndex", -1)
        self.branchValue = kwargs.get("branchValue", -1)
        if self.parent is not None:
            self.depth = self.parent.depth + 1
            self.lb = self.parent.lb
        else:
            self.depth = 0
            self.lb = -inf
        self.lbChild0 = self.lbChild1 = -inf

    cpdef updateBound(self, double lbChild, int childValue):
        cdef double newLb, oldChild = self.lbChild0 if childValue == 0 else self.lbChild1
        if lbChild > oldChild:
            if childValue == 0:
                self.lbChild0 = lbChild
            else:
                self.lbChild1 = lbChild
        newLb = fmin(self.lbChild0, self.lbChild1)
        if newLb > self.lb:
            self.lb = newLb
            if self.parent is not None:
                self.parent.updateBound(newLb, self.branchValue)

    def printFixes(self):
        cdef Node node = self
        while node is not None:
            print('x{}={}, '.format(node.branchIndex, node.branchValue), end='')
            node = node.parent
        print()


cdef class BranchAndCutDecoder(Decoder):
    """
    Maximum-Likelihood decoder using a branch-and-cut approach. Upper bounds (i.e. valid
    solutions) are generated using the max-product algorithm from :class:`IterativeDecoder`.
    Lower bounds and cuts are generated with :class:`AdaptiveLPDecoder`.

    :param str branchMethod: Method to determine the (fractional) variable on which to branch.
        Possible values are:

        * `"mostFractional"`: choose the variable which is nearest to :math:`\\frac{1}{2}`
        * `"leastReliable"`: choose the fractional variable whose LLR value is closest to 0
        * `"eiriksPaper"`: (currently unimplemented)
    :param str selectionMethod: Method to determine the next node from the set of active nodes.
        Possible values:

        * `"bfs"`: Breadth-first search
        * `"dfs"`: Depth-first search
        * `"bbs"`: Best-bound search
        * `"mixed[-]/<a>/<b>/<c>/<d>"`: Mixed strategy. Uses depth-first search in general but
            jumps to the node with smallest lower bound every <a> iterations, but only if the
            duality gap at the current node is at least <d>. In the DFS phase, at most <b> RPC
            cut-search iterations are performed in the LP solver in each node. After a BBS jump,
            <c> is used instead.
    :param str childOrder: Determines the order in which newly created child branch-and-bound
        nodes are appended to the list of active nodes. Possible values are:

        * `"01"`: child with zero-fix is added first
        * `"10"`: child with one-fix is added first
        * `"llr"`: add first the node whose fix-value equals the hard-decision value of the fixed bit.
        * `"random"`: add in random order
    :param bool highSNR: Use optimizations for high SNR values.
    :param str name: Name of the decoder
    :param dict lpParams: Parameters for the LP decoder
    :param dict iterParams: Parameters for the iterative decoder
    """
    cdef:
        bint calcUb  # indicates whether to run the iterative decoder in the next iteration
        bint ubBB, highSNR, minDistance
        object childOrder
        object timer
        SelectionMethod selectionMethod
        BranchMethod branchMethod
        public Decoder lbProvider, ubProvider
        int mixParam, maxRPCspecial, maxRPCnormal, maxRPCorig
        double mixGap, sentObjective
        int selectCnt
        Node root

    def __init__(self, code,
                 branchMethod='mostFractional',
                 selectionMethod='dfs',
                 childOrder='01',
                 highSNR=False,
                 name='BranchAndCutDecoder',
                 lpParams=None,
                 iterParams=None):
        self.name = name
        if lpParams is None:
            lpParams = {}
        if iterParams is None:
            iterParams = {}
        self.lbProvider = AdaptiveLPDecoder(code, **lpParams)
        self.ubProvider = IterativeDecoder(code, **iterParams)
        self.highSNR = highSNR
        if branchMethod == 'mostFractional':
            self.branchMethod = mostFractional
        elif branchMethod == 'leastReliable':
            self.branchMethod = leastReliable
        else:
            assert branchMethod == 'eiriksPaper'
            raise NotImplementedError('eiriks branch method not implemented')
        self.childOrder = childOrder
        self.calcUb = True
        if selectionMethod.startswith('mixed'):
            self.selectionMethod = mixed
            if selectionMethod[5] == '-':
                self.ubBB = True
                selectionMethod = selectionMethod[7:]
            else:
                self.ubBB = False
                selectionMethod = selectionMethod[6:]
            a, b, c, d = selectionMethod.split("/")
            self.mixParam = int(a)
            self.maxRPCspecial = int(b)
            self.maxRPCnormal = int(c)
            self.mixGap = float(d)
            self.maxRPCorig = self.lbProvider.maxRPCrounds
        elif selectionMethod == 'bfs':
            self.selectionMethod = bfs
        elif selectionMethod == 'dfs':
            self.selectionMethod = dfs
        else:
            assert selectionMethod == "bbs", str(selectionMethod)
            self.selectionMethod = bbs
        self.timer = Timer()
        Decoder.__init__(self, code, name=name)


    cpdef setStats(self, stats):
        for item in "nodes", "prBd1", "prBd2", "prInf", "prOpt", "termEx", "termGap", 'lpTime', \
                    'iterTime':
            if item not in stats:
                stats[item] = 0
        if "nodesPerDepth" not in stats:
            stats["nodesPerDepth"] = {}
        if "lpStats" in stats:
            self.lbProvider.setStats(stats["lpStats"])
            del stats["lpStats"]
        else:
            self.lbProvider.setStats(dict())
        if "iterStats" in stats:
            self.ubProvider.setStats(stats["iterStats"])
            del stats["iterStats"]
        else:
            self.ubProvider.setStats(dict())
        Decoder.setStats(self, stats)


    cpdef stats(self):
        stats = self._stats.copy()
        stats['lpStats'] = self.lbProvider.stats().copy()
        stats['iterStats'] = self.ubProvider.stats().copy()
        return stats


    cdef int branchIndex(self):
        """Determines the index of the current branching variable, according to the selected
        branch method."""
        cdef:
            int index, i
            double minDiff = np.inf
            np.double_t[:] solution = self.lbProvider.solution
        if self.branchMethod == mostFractional:
            for i in range(self.code.blocklength):
                if fabs(solution[i] - .5) < minDiff:
                    index = i
                    minDiff = fabs(solution[i] - .5)
            if solution[index] < 1e-6 or solution[index] > 1-1e-6:
                #  no fractional value exists -> branch on first free position
                for index in range(self.code.blocklength):
                    if not self.fixed(index):
                        return index
                return -1
            return index
        elif self.branchMethod == leastReliable:
            for i in np.argsort(np.abs(self.llrs)):
                if fabs(.5-solution[i]) < .499:
                    return i
            return -1
        else:
            raise NotImplementedError('Eiriks method not implemented')


    cpdef setLLRs(self, np.double_t[:] llrs, np.int_t[:] sent=None):
        self.ubProvider.setLLRs(llrs, sent)
        if sent is not None:
            self.sentObjective = np.dot(sent, llrs)
        else:
            self.sentObjective = -inf
        if self.highSNR:
            self.ubProvider.foundCodeword = self.ubProvider.mlCertificate = False
        else:
            self.timer.start()
            self.ubProvider.solve()
            self._stats["iterTime"] += self.timer.stop()
            if self.ubProvider.foundCodeword:
                self.lbProvider.hint = self.ubProvider.solution.astype(np.int)
                logger.debug('init ub={}'.format(self.ubProvider.objectiveValue))
                # codeword will be used in first iteration of main algorithm; no need to copy it
                # here
        self.lbProvider.setLLRs(llrs, sent)
        Decoder.setLLRs(self, llrs)


    cpdef fix(self, int index, int value):
        self.lbProvider.fix(index, value)
        self.ubProvider.fix(index,value)


    cpdef release(self, int index):
        self.lbProvider.release(index)
        self.ubProvider.release(index)


    cpdef fixed(self, int index):
        return self.lbProvider.fixed(index)


    cpdef solve(self, double lb=-np.inf, double ub=np.inf):
        cdef:
            Node node, newNode0, newNode1, newNode
            list activeNodes = []
            np.ndarray[dtype=np.double_t, ndim=1] candidate = np.zeros(self.code.blocklength, dtype=np.double)
            int i, branchIndex
            str depthStr
        ub = 0
        #  ensure there are no leftover fixes from previous decodings
        for i in range(self.code.blocklength):
            self.lbProvider.release(i)
            self.ubProvider.release(i)
        self.foundCodeword = self.mlCertificate = True
        self.root = node = Node() #  root node
        self.selectCnt = 0 #  parameter used for the mixed node selection strategy
        self._stats["nodes"] += 1
        if self.selectionMethod == mixed:
            self.lbProvider.maxRPCrounds = self.maxRPCspecial

        for i in itertools.count(start=1):

            # statistic collection and debug output
            depthStr = str(node.depth)
            if depthStr not in self._stats["nodesPerDepth"]:
                self._stats["nodesPerDepth"][depthStr] = 0
            self._stats["nodesPerDepth"][depthStr] += 1

            # upper bound calculation
            if i > 1 and self.calcUb: # for first iteration this was done in setLLR
                self.timer.start()
                self.ubProvider.solve()
                self._stats["iterTime"] += self.timer.stop()
            if self.ubProvider.foundCodeword:
                logger.debug('ub solution={}'.format(self.ubProvider.objectiveValue))
            if self.ubProvider.foundCodeword and self.ubProvider.objectiveValue < ub:
                candidate = self.ubProvider.solution.copy()
                ub = self.ubProvider.objectiveValue
                if ub < self.sentObjective:
                    self.mlCertificate = False
                    break

            # lower bound calculation
            self.timer.start()
            if (i == 1 or self.calcUb) and self.ubProvider.foundCodeword:
                self.lbProvider.hint = self.ubProvider.solution.astype(np.int)
            else:
                self.lbProvider.hint = None
            logger.debug('lp with {} rounds'.format(self.lbProvider.maxRPCrounds))
            self.lbProvider.solve(-inf, ub)
            self._stats['lpTime'] += self.timer.stop()
            if self.lbProvider.objectiveValue > node.lb:
                node.lb = self.lbProvider.objectiveValue

            # pruning or branching
            if node.lb == np.inf:
                logger.debug("node pruned by infeasibility")
                self._stats["prInf"] += 1
            elif self.lbProvider.foundCodeword:
                # solution is integral
                logger.debug("node pruned by integrality")
                if self.lbProvider.objectiveValue < ub:
                    candidate = self.lbProvider.solution.copy()
                    ub = self.lbProvider.objectiveValue
                    logger.debug("ub improved to {}".format(ub))
                    self._stats['prOpt'] += 1
                    if ub < self.sentObjective:
                        self.mlCertificate = False
                        break
            elif node.lb < ub-1e-6:
                # branch
                branchIndex = self.branchIndex()
                newNode0 = Node(parent=node, branchIndex=branchIndex, branchValue=0)
                newNode1 = Node(parent=node, branchIndex=branchIndex, branchValue=1)
                if (self.childOrder == '10') or \
                   (self.childOrder == 'llr' and self.llrs[branchIndex] < 0) or \
                   (self.childOrder == 'random' and np.random.randint(0, 2) == 0):
                    activeNodes.append(newNode1)
                    activeNodes.append(newNode0)
                else:
                    activeNodes.append(newNode0)
                    activeNodes.append(newNode1)
                self._stats['nodes'] += 2
            else:
                self._stats["prBd2"] += 1
            if node.parent is not None:
                node.parent.updateBound(node.lb, node.branchValue)
                if self.root.lb >= ub - 1e-6:
                    self._stats["termGap"] += 1
                    break
            if len(activeNodes) == 0:
                self._stats["termEx"] += 1
                break
            newNode = self.selectNode(activeNodes, node, ub)
            move(self.lbProvider, self.ubProvider, node, newNode)
            node = newNode
        if self.selectionMethod == mixed:
            self.lbProvider.maxRPCrounds = self.maxRPCorig
        self.solution = candidate
        self.objectiveValue = ub


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
        if cyclic:
            self.fix(0, 1)
        self.setLLRs(llrs)
        self.selectCnt = 1
        self.root = node = Node()
        self.root.lb = 1
        activeNodes = []
        candidate = None
        ub = np.inf
        self._stats['nodes'] += 1
        for i in itertools.count(start=1):
            # statistic collection and debug output
            depthStr = str(node.depth)
            if i % 1000 == 0:
                logger.info('MD {}/{}, d {}, n {}, c {}, it {}, lp {}, spa {}'.format(
                    self.root.lb,ub, node.depth,len(activeNodes), self.lbProvider.numConstrs, i,
                    self._stats["lpTime"], self._stats['iterTime']))
            if depthStr not in self._stats['nodesPerDepth']:
                self._stats['nodesPerDepth'][depthStr] = 0
            self._stats["nodesPerDepth"][depthStr] += 1
            pruned = False # store if current node can be pruned
            if node.lb >= ub-1+delta:
                node.lb = np.inf
                pruned = True
            if not pruned:
                # upper bound calculation

                if i > 1 and self.calcUb: # for first iteration this was done in setLLR
                    self.timer.start()
                    self.ubProvider.solve()
                    self._stats['iterTime'] += self.timer.stop()

                if self.ubProvider.foundCodeword and self.ubProvider.objectiveValue < ub:
                    candidate = self.ubProvider.solution.copy()
                    ub = self.ubProvider.objectiveValue
                # lower bound calculation
                self.timer.start()

                if (i == 1 or self.calcUb) and self.ubProvider.foundCodeword:
                    self.lbProvider.hint = self.ubProvider.solution.astype(np.int)
                else:
                    self.lbProvider.hint = None
                self.lbProvider.solve(-inf, ub - 1 + delta)
                self._stats['lpTime'] += self.timer.stop()
                if self.lbProvider.objectiveValue > node.lb:
                    node.lb = self.lbProvider.objectiveValue
                if node.lb == np.inf:
                    self._stats['prInf'] += 1
                elif self.lbProvider.foundCodeword and self.lbProvider.objectiveValue > .5:
                    # solution is integral
                    logger.debug('node pruned by integrality')
                    if self.lbProvider.objectiveValue < ub:
                        candidate = self.lbProvider.solution.copy()
                        print('new candidate from LP with weight {}'.format(
                            self.lbProvider.objectiveValue))
                        ub = self.lbProvider.objectiveValue
                        logger.debug('ub improved to {}'.format(ub))
                        self._stats['prOpt'] += 1
                elif node.lb < ub-1+delta:
                    # branch
                    branchIndex = self.branchIndex()
                    if branchIndex == -1:
                        node.lb = np.inf
                        print('********** PRUNE 000000 ***************')
                    else:
                        newNodes = [Node(parent=node, branchIndex=branchIndex, branchValue=i) for i in (0,1) ]
                        if self.childOrder == 'random':
                            random.shuffle(newNodes)
                        elif (self.childOrder == 'llr' and self.llrs[branchIndex] < 0) or self.childOrder == '10':
                            newNodes.reverse()
                        activeNodes.extend(newNodes)
                        self._stats['nodes'] += 2
                else:
                    logger.debug("node pruned by bound 2")
                    self._stats["prBd2"] += 1
            if node.parent is not None:
                node.parent.updateBound(node.lb, node.branchValue)
                if self.root.lb >= ub - 1 + delta:
                    self._stats["termGap"] += 1
                    break
            if len(activeNodes) == 0:
                self._stats["termEx"] += 1
                break
            newNode = self.selectNode(activeNodes, node, ub)
            move(self.lbProvider, self.ubProvider, node, newNode)
            node = newNode
        self.solution = candidate
        print(candidate)
        assert self.solution in self.code
        self.objectiveValue = np.rint(ub)
        # restore normal (decoding) mode
        self.ubProvider.excludeZero = self.minDistance = False
        return self.objectiveValue

    cdef Node popMinNode(self, list activeNodes):
        cdef int i, minIndex
        cdef double minValue = np.inf
        for i in range(len(activeNodes)):
            if activeNodes[i].lb < minValue:
                minIndex = i
                minValue = activeNodes[i].lb
        return activeNodes.pop(minIndex)


    cdef Node selectNode(self, list activeNodes, Node currentNode, double ub):
        if self.selectionMethod == mixed:
            if (self.selectCnt >= self.mixParam or (self.minDistance and self.root.lb == 1)) and (ub - currentNode.lb) > self.mixGap:
                # best bound step
                newNode = self.popMinNode(activeNodes)
                self.selectCnt = 1
                self.lbProvider.maxRPCrounds = self.maxRPCspecial #np.rint(ub-newNode.lb)
                if self.ubBB:
                    self.calcUb = True
                return newNode
            else:
                self.lbProvider.maxRPCrounds = self.maxRPCnormal
                self.selectCnt += 1
                if self.ubBB:
                    self.calcUb = False
                return activeNodes.pop()
        elif self.selectionMethod == dfs:
            return activeNodes.pop()
        elif self.selectionMethod == bbs:
            return self.popMinNode(activeNodes)
        elif self.selectionMethod == bfs:
            return activeNodes.pop(0)

    cpdef params(self):
        if self.selectionMethod == mixed:
            method = "mixed{}/{}/{}/{}/{}".format('-' if self.ubBB else "",
                                                   self.mixParam,
                                                   self.maxRPCspecial,
                                                   self.maxRPCnormal,
                                                   self.mixGap)
        else:
            selectionMethodNames = { dfs: "dfs", bfs: "bfs", bbs: "bbs"}
            method = selectionMethodNames[self.selectionMethod]
        branchMethodNames = {mostFractional: "mostFractional",
                             leastReliable: "leastReliable",
                             eiriksPaper: "eiriksPaper"}
        parms = OrderedDict()
        if self.branchMethod != mostFractional:
            parms['branchMethod'] = branchMethodNames[self.branchMethod]
        parms['selectionMethod'] = method
        if self.childOrder != '01':
            parms['childOrder'] = self.childOrder
        parms['lpParams'] = self.lbProvider.params()
        parms['iterParams'] = self.ubProvider.params()
        if self.highSNR:
            parms['highSNR'] = True
        parms['name'] = self.name
        return parms
