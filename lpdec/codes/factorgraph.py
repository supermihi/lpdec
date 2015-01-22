# -*- coding: utf-8 -*-
# Copyright 2014-2015 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation

"""The :mod:`lpdec.codes.factorgraph` module contains classes for defining a factor graph of a
parity-check
matrix."""

import numpy as np


class FactorGraph:
    """Factor graph main class.

    .. attribute:: varNodes

      List of variable nodes of the graph.

    .. attribute:: checkNodes

      List of check nodes of the graph.
    """
    def __init__(self, varNodes, checkNodes, x=None):
        self.varNodes = varNodes
        for i, var in enumerate(varNodes):
            if not hasattr(var, 'index'):
                var.index = i
            elif var.index != i:
                raise ValueError('{}-th var nodes has index {}!={}'.format(i, var.index, i))
        self.checkNodes = checkNodes
        for i, check in enumerate(checkNodes):
            if not hasattr(check, 'index'):
                check.index = i
            elif check.index != i:
                raise ValueError('{}-th check nodes has index {}!={}'.format(i, check.index, i))
        if x is not None:
            assert set(x) <= set(varNodes)
            self.x = x
        else:
            self.x = varNodes

    @classmethod
    def fromLinearCode(cls, code):
        """Creates the factor graph according to the parity-check matrix of *code*."""
        pcm = code.parityCheckMatrix
        m, n = pcm.shape
        varNodes = [VariableNode(i) for i in range(n)]
        checkNodes = [CheckNode(j) for j in range(m)]
        for check, row in zip(checkNodes, pcm):
            for i in np.flatnonzero(row):
                check.connect(varNodes[i])
        return cls(varNodes, checkNodes)

    def parityCheckMatrix(self):
        H = np.zeros((len(self.checkNodes), len(self.varNodes)), dtype=np.int)
        for check in self.checkNodes:
            for var in check.neighbors:
                H[check.index, var.index] = 1
        return H


class FactorNode:
    """Base class for nodes of a factor graph.

    :param identifier: A free-form identifier for this node.

    .. attribute:: neighbors

      List of neighboring nodes.
    """
    def __init__(self, identifier):
        self.neighbors = []
        self.identifier = identifier

    def connect(self, other):
        """Connect *other* to this node by updating the :attr:`neighbors` lists of both nodes.
        """
        assert other not in self.neighbors
        assert self not in other.neighbors
        self.neighbors.append(other)
        other.neighbors.append(self)

    def disconnect(self, other):
        assert other in self.neighbors
        self.neighbors.remove(other)
        other.neighbors.remove(self)

    def isolate(self):
        for neigh in self.neighbors[:]:
            self.disconnect(neigh)

    @property
    def degree(self):
        """The degree of this node, i.e. number of connected nodes."""
        return len(self.neighbors)

    def __str__(self):
        return self.identifier

    def __repr__(self):
        return str(self)


class VariableNode(FactorNode):
    """:class:`FactorNode` subclass for variables."""
    def connect(self, other):
        assert isinstance(other, CheckNode), 'No check node: {}'.format(other)
        FactorNode.connect(self, other)

    def merge(self, other):
        """Merge *other* into this node. Afterwards, *other* will be isolated and all of its
        neighbors connected to *self*.
        """
        for check in other.neighbors[:]:
            if self in check.neighbors:
                # merging two variable nodes connected to the same check: in that case, the new
                # super-variable node is completely irrelevant for the check, so both edges are
                # removed
                self.disconnect(check)
                other.disconnect(check)
            else:
                other.disconnect(check)
                self.connect(check)


class CheckNode(FactorNode):
    """:class:`FactorNode` subclass for checks."""
    def connect(self, other):
        assert isinstance(other, VariableNode), 'No var node: {} ({})'.format(other, type(other))
        FactorNode.connect(self, other)
