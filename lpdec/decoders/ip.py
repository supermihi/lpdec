# -*- coding: utf-8 -*-
# Copyright 2014 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation

# -*- coding: utf-8 -*-
# Copyright 2011-2014 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation

""" This module contains integer programming (IP) decoders for binary linear block codes,
based on the formulation called 'IPD' in:
Helmling, M., Ruzika, S. and Tanatmis, A.: "Mathematical Programming Decoding of Binary Linear
Codes: Theory and Algorithms", IEEE Transactions on Information Theory, Vol. 58 (7), 2012,
pp. 4753-4769.
."""
from __future__ import division, print_function
from collections import OrderedDict
import sys
import numpy as np
from lpdec.decoders import Decoder, cplexhelpers


class CplexIPDecoder(cplexhelpers.CplexDecoder):
    """CPLEX implementation of the IPD maximum-likelihood decoder.

    For ML simulations, the decoding process can be speed up using a shortcut callback.

    .. attribute:: z

       Vector of names of the auxiliary variables
    """
    def __init__(self, code, name=None, **kwargs):
        if name is None:
            name = 'CplexIPDecoder'
        cplexhelpers.CplexDecoder.__init__(self, code, name, **kwargs)
        matrix = code.parityCheckMatrix
        self.z = ['z' + str(num) for num in range(matrix.shape[0])]
        self.cplex.variables.add(types=['I'] * matrix.shape[0], names=self.z)
        self.cplex.linear_constraints.add(
            names=['parity_check_' + str(num) for num in range(matrix.shape[0])])
        for cnt, row in enumerate(matrix):
            nonzero_indices = [(self.x[i], row[i]) for i in range(row.size) if row[i]]
            nonzero_indices.append((self.z[cnt], -2))
            self.cplex.linear_constraints.set_linear_components(
                'parity_check_{0}'.format(cnt),
                zip(*nonzero_indices))




    def minimumDistance(self, hint=None):
        """Calculate the minimum distance of :attr:`code` via integer programming.

        Compared to the decoding formulation, this adds the constraint :math:`|x| \\geq 1` and
        minimizes :math:`\\sum_{i=1}^n x`.
        """
        self.cplex.linear_constraints.add(
            names=['mindistance'],
            lin_expr=[[self.x, [1] * len(self.x)]],
            senses=['G'],
            rhs=[1]
        )
        self.cplex.parameters.mip.tolerances.absmipgap.set(1-1e-5) # all solutions are integral
        self.setLLRs(np.ones(self.code.blocklength))
        self.solve(hint)
        self.cplex.linear_constraints.delete('mindistance')
        return int(round(self.objectiveValue)), np.array(self.solution)

    def fix(self, index, value):
        if value == 0:
            self.cplex.variables.set_upper_bounds(self.x[index], 0)
        else:
            self.cplex.variables.set_lower_bounds(self.x[index], 1)

    def release(self, index):
        self.cplex.variables.set_lower_bounds(self.x[index], 0)
        self.cplex.variables.set_upper_bounds(self.x[index], 1)

    def params(self):
        params = OrderedDict(name=self.name)
        if self.shortCallback:
            params['shortCallback'] = self.shortCallback
        params['cplexParams'] = cplexhelpers.getCplexParams(self.cplex)
        return params


class GurobiMLDecoder(Decoder):

    def __init__(self, code, singleThread=False, gurobiParams=dict(), name=None):
        Decoder.__init__(self, code)
        if name is None:
            name = "GurobiMLDecoder" + ("(st)" if singleThread else "")
        self.name = name
        self.singleThread = singleThread
        from gurobipy import Model, GRB, quicksum
        matrix = code.parityCheckMatrix
        self.model = Model("ML Decoder")
        self.model.setParam("OutputFlag", 0)
        if singleThread:
            self.model.setParam("Threads", 1)
        for param, value in gurobiParams.items():
            self.model.setParam(param, value)
        self.x = [self.model.addVar(vtype=GRB.BINARY, name="x{}".format(i))
                  for i in range(code.blocklength)]
        self.z = [self.model.addVar(vtype=GRB.INTEGER, name="z{}".format(i))
                  for i in range(matrix.shape[0])]
        self.model.update()
        for z, row in zip(self.z, matrix):
            self.model.addConstr(quicksum(self.x[i] for i in np.flatnonzero(row)) - 2 * z, GRB.EQUAL, 0)
        self.model.update()
        self.mlCertificate = self.foundCodeword = True
        self.solution = np.empty(code.blocklength, dtype=np.double)

    def setStats(self, stats):
        if "nodes" not in stats:
            stats["nodes"] = 0
        Decoder.setStats(self, stats)

    def setLLRs(self, llrs):
        from gurobipy import GRB, LinExpr
        self.model.setObjective(LinExpr(llrs, self.x), GRB.MINIMIZE)
        Decoder.setLLRs(self, llrs)

    def solve(self, hint=None, lb=1):
        from gurobipy import GRB
        if hint is not None:
            for val, x in zip(hint, self.x):
                x.setAttr("Start", val)
            for row, z in zip(self.code.parityCheckMatrix, self.z):
                z.setAttr("Start", np.dot(row, np.asarray(hint)) / 2)
        self.model.optimize()
        if self.model.getAttr("Status") == GRB.INTERRUPTED:
            raise KeyboardInterrupt()
        self._stats["nodes"] += self.model.getAttr("NodeCount")
        self.objectiveValue = self.model.objVal
        for i, x in enumerate(self.x):
            self.solution[i] = x.x

    def params(self):
        params = [("name", self.name)]
        if self.singleThread:
            params.insert(0, ("singleThread", True))
        return OrderedDict(params)
