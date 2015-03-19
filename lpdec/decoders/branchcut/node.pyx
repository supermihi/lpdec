# -*- coding: utf-8 -*-
# Copyright 2015 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation
from __future__ import division, print_function
from numpy.math cimport INFINITY
from libc.math cimport fmin

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
        else:
            self.depth = 0
            self.lb = -INFINITY
        self.lbChild0 = self.lbChild1 = -INFINITY

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


