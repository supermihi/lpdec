# -*- coding: utf-8 -*-
# Copyright 2014 Michael Helmling
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
    def __init__(self, varNodes, checkNodes):
        self.varNodes = varNodes
        self.checkNodes = checkNodes

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
        assert isinstance(other, CheckNode)
        FactorNode.connect(self, other)


class CheckNode(FactorNode):
    """:class:`FactorNode` subclass for checks."""
    def connect(self, other):
        assert isinstance(other, VariableNode)
        FactorNode.connect(self, other)
