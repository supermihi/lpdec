# -*- coding: utf-8 -*-
# Copyright 2015 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation
from __future__ import division, print_function
import numpy as np
from numpy.math cimport INFINITY
from libc.math cimport fmin, fmax

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
    def __init__(self, Node parent=None, int branchIndex=-1, int branchValue=-1):
        self.parent = parent
        self.branchIndex = branchIndex
        self.branchValue = branchValue
        if parent:
            self.depth = self.parent.depth + 1
            self.lb = self.parent.lb
            while parent is not None:
                if parent.branchLb is not None:
                    if parent.branchLb[branchIndex, branchValue] > self.lb:
                        self.lb = parent.branchLb[branchIndex, branchValue]
                        self.parent.updateBound(self.lb, branchValue)
                parent = parent.parent
        else:
            self.depth = 0
            self.lb = -INFINITY

        self.lbChild0 = self.lbChild1 = -INFINITY
        self.implicitFixes = []
        self.branchLb = None

    cdef void updateBound(self, double lbChild, int childValue):
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

    cdef list branch(self, int index, bytes childOrder, Decoder decoder, double ub):
        cdef Node node0 = None, node1 = None
        if self.branchLb is None or self.branchLb[index, 0] <= ub - 1e-6:
            node0 = Node(self, index, 0)
            decoder._stats['nodes'] += 1
        else:
            self.lbChild0 = INFINITY
        if self.branchLb is None or self.branchLb[index, 1] <= ub - 1e-6:
            node1 = Node(self, index, 1)
            decoder._stats['nodes'] += 1
        else:
            self.lbChild1 = INFINITY
        if node0 is None:
            if node1 is None:
                return []
            return [node1]
        elif node1 is None:
            return [node0]
        else:
            if childOrder == b'10' or (childOrder == b'llr' and decoder.llrs[index] < 0) or \
                    (childOrder == b'random' and np.random.randint(0, 2) == 0):
                return [node1, node0]
            else:
                return [node0, node1]

    def isDescendantOf(self, Node other):
        cdef Node current = self
        while current is not None:
            if current is other:
                return True
            current = current.parent
        return False

    def printFixes(self):
        cdef Node node = self
        while node is not None:
            print('x{}={}, '.format(node.branchIndex, node.branchValue), end='')
            node = node.parent
        print()


cdef int move(Decoder lbProv, Decoder ubProv, Node node, Node newNode) except -1:
    """Moves from one brach-and-bound node to another, updating variable fixes in the upper and
    lower bound providers on the fly.
    """
    cdef list fix = []
    while node.depth > newNode.depth:
        lbProv.release(node.branchIndex)
        ubProv.release(node.branchIndex)
        for i, _ in node.implicitFixes:
            lbProv.release(i)
            ubProv.release(i)
        node = node.parent

    while newNode.depth > node.depth:
        fix.append( (newNode.branchIndex, newNode.branchValue) )
        for i, v in newNode.implicitFixes:
            fix.append( (i, v) )
        newNode = newNode.parent
    while node is not newNode:
        lbProv.release(node.branchIndex)
        ubProv.release(node.branchIndex)
        for i, _ in node.implicitFixes:
            lbProv.release(i)
            ubProv.release(i)
        fix.append( (newNode.branchIndex, newNode.branchValue) )
        for i, v in newNode.implicitFixes:
            fix.append((i, v))
        node = node.parent
        newNode = newNode.parent
    for var, value in fix:
        lbProv.fix(var, value)
        ubProv.fix(var, value)


