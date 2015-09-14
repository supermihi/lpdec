# -*- coding: utf-8 -*-
# Copyright 2014-2015 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation


"""
This module contains integer programming (IP) decoders for binary linear block codes,
based on the formulation called 'IPD' in :cite:`Helmling+11MathProgDecoding`.
"""
from __future__ import division, print_function
from collections import OrderedDict
import numpy as np
from lpdec import utils
from lpdec.decoders import Decoder, cplexhelpers, gurobihelpers


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
            names=['nonzero'], lin_expr=[(self.x, np.ones(len(self.x)))], senses='G', rhs=[1])
        self.cplex.parameters.mip.tolerances.absmipgap.set(1-1e-5)  # all solutions are integral
        self.setLLRs(np.ones(self.code.blocklength))
        self.solve(hint)
        self.cplex.linear_constraints.delete('nonzero')
        return int(round(self.objectiveValue))

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
        params['cplexParams'] = self.cplexParams(self.cplex)
        return params


class GurobiIPDecoder(gurobihelpers.GurobiDecoder):
    """Gurobi implementation of the IPD maximum-likelihood decoder.

    :param LinearBlockCode code: The code to decoder.
    :param dict gurobiParams: Optional dictionary of parameters; these are passed to the Gurobi
        model via :func:`gurobimh.Model.setParam`. The attributes :attr:`tuningSet1`,
        :attr:`tuningSet2` and :attr:`tuningSet3` contain three sets of parameters that were
        obtained from the Gurobi tuning tool.
    :param str gurobiVersion: Version of the Gurobi package; if supplied, an error is raised if
        the current version does not match.
    :param str name: Name of the decoder. Defaults to "GurobiIPDecoder".

    The number of nodes in Gurobi's branch-and-bound procedure is collected in the statistics.

    Example usage:
        >>> from lpdec.imports import *
        >>> code = HammingCode(3)
        >>> decoder = GurobiIPDecoder(code, gurobiParams=GurobiIPDecoder.tuningSet1, name='GurobiTuned')
        >>> result = decoder.decode([1, -1, 0, -1.5, 2, 3, 0])
        >>> print(result)

    .. attribute:: tuningSet1

    Dictionary to be passed to the constructor as *gurobiParams*; this set of parameters was
    obtained from the Gurobi tuning tool on a hard instance for the (155,93) Tanner LDPC code.

    .. attribute:: tuningSet2

    As above; second-best parameter set.

    .. attribute:: tuningSet3

    As above; third-best parameter set.
    """
    def __init__(self, code, gurobiParams=None, gurobiVersion=None, name=None):

        if name is None:
            name = 'GurobiIPDecoder'
        if utils.isStr(gurobiParams):
            # allow to specify tuning set parameters using gurobiParams='tuning1' or similar
            for i in ('1', '2', '3'):
                if i in gurobiParams:
                    gurobiParams = getattr(self, 'tuningSet' + i)
                    break
        elif gurobiParams is None:
            gurobiParams = self.tuningSet1
        gurobihelpers.GurobiDecoder.__init__(self, code, name, gurobiParams, gurobiVersion,
                                             integer=True)
        from gurobimh import GRB, quicksum
        matrix = code.parityCheckMatrix
        for i in range(code.blocklength):
            self.model.addConstr(quicksum(self.x[i, k] for k in range(1, code.q)),
                                 GRB.LESS_EQUAL, 1)
        self.z = []
        for i in range(matrix.shape[0]):
            ub = np.sum(matrix[i]) * (code.q - 1) // code.q
            self.z.append(self.model.addVar(0, ub, vtype=GRB.INTEGER, name='z{}'.format(i)))
        self.model.update()
        for z, row in zip(self.z, matrix):
            self.model.addConstr(quicksum(row[i]*k*self.x[i, k] for k in range(1, code.q) for i
                                          in np.flatnonzero(row)) - code.q * z, GRB.EQUAL, 0)
        self.model.update()
        self.mlCertificate = self.foundCodeword = True

    tuningSet1 = dict(MIPFocus=2, PrePasses=2, Presolve=2)
    tuningSet2 = dict(MIPFocus=2, VarBranch=1)
    tuningSet3 = dict(MIPFocus=2)

    def setStats(self, stats):
        if 'nodes' not in stats:
            stats['nodes'] = 0
        Decoder.setStats(self, stats)

    @staticmethod
    def callback(model, where):
        """ A callback function for Gurobi that is able to terminate the MIP solver if a solution
        which is better than the sent codeword has been found.
        """
        from gurobimh import GRB
        if where == GRB.Callback.MIPNODE:
            if model.cbGet(GRB.Callback.MIPNODE_OBJBST) < model._realObjective - 1e-6:
                model._incObj = model.cbGet(GRB.Callback.MIPNODE_OBJBST)
                model.terminate()

    def solve(self, lb=-np.inf, ub=np.inf):
        q = self.code.q
        self.model.write('ham.lp')
        from gurobimh import GRB
        self.mlCertificate = True
        if self.sent is not None:
            sent = np.asarray(self.sent)
            self.model._realObjective = 0
            for i, val in enumerate(sent):
                for k in range(1, q):
                    self.x[i, k].Start = 1 if val == k else 0
                if val != 0:
                    self.model._realObjective += self.llrs[i*(q-1)+val-1]
            zValues = np.dot(self.code.parityCheckMatrix, sent // q).tolist()
            for val, var in zip(zValues, self.z):
                var.Start = val
            self.model._incObj = None
            self.model.optimize(GurobiIPDecoder.callback)
        else:
            self.model.optimize()
        if self.model.Status == GRB.INTERRUPTED:
            if self.sent is None or self.model._incObj is None:
                raise KeyboardInterrupt()
            else:
                self.objectiveValue = self.model._incObj
                self.mlCertificate = False
        self._stats['nodes'] += self.model.NodeCount
        self.readSolution()

    def minimumDistance(self, hint=None):
        """Calculate the minimum distance of :attr:`code` via integer programming.

        Compared to the decoding formulation, this adds the constraint :math:`|x| \\geq 1` and
        minimizes :math:`\\sum_{i=1}^n x`.
        """
        from gurobimh import quicksum, GRB
        self.model.addConstr(quicksum(self.xlist), GRB.GREATER_EQUAL, 1, name='excludeZero')
        self.model.setParam('MIPGapAbs', 1-1e-5)
        self.setLLRs(np.ones(self.code.blocklength * (self.code.q - 1)))
        self.solve()
        self.model.remove(self.model.getConstrByName('excludeZero'))
        self.model.update()
        return int(round(self.objectiveValue))

