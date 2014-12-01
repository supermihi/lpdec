# -*- coding: utf-8 -*-
# Copyright 2014 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation

# -*- coding: utf-8 -*-
# distutils: libraries = ["glpk", "m"]
# cython: boundscheck=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: wraparound=False
#
# Copyright 2014 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation

from __future__ import division, print_function
from collections import OrderedDict
import logging
from gurobipy.gurobipy import Model
import numpy as np
import gurobipy as g

from lpdec.decoders.base import Decoder
from lpdec import mod2la
from lpdec.utils import Timer

logger = logging.getLogger('alp_gurobi')

class GurobiALPDecoder(Decoder):
    """
    Implements the adaptive linear programming decoder with optional generation of redundant
    parity-check (RPC) cuts.

    Uses the GLPK C-interface for solving the LP subproblems.

    :param int maxRPCrounds: Maximum number of iterations of RPC cuts generation. The value
      ``-1`` means no limit (as in the paper). If set to ``0``, no RPC cuts are generated,
      and the decoder behaves as a normal LP decoder.
    :param float minCutoff: Minimum violation of an inequality to be inserted as a cut into the
      LP model.
      Defaults to ``1e-5``. A smaller value might lead to numerical problems, i.e,, an inequality
      being inserted that is not really a cut and hence the danger of infinite loops. On the
      other hand, a larger value can make sense, in order to only use "strong" cuts.
    :param int removeInactive: Determines if and when inactive constraints are removed from the LP
      model in order to limit its size. ``0`` (the default) turns off removal of constraints. A
      value of ``-1`` means that inactive constraints should be removed after each call to the LP
      solver. A positive number leads to removal of inactive constraints as soon as the total
      number of constraints exceeds that number.
    :param bool removeAboveAverageSlack: If set to ``True``, during removal of inactive
      constraints (see above) only those with a slack above the average of all inactive
      constraints are indeed removed.
    :param bool keepCuts: If set to ``True``, inserted cuts are not remove after decoding one
      frame.

    .. attribute:: hint

        An optional "hint" codeword that is somehow believed to be near-optimal. For example,
        this might be the output of an iterative decoder. The adaptive LP decoder can use that
        hint to insert active constraints.
    """

    def __init__(self, code, maxRPCrounds=-1, minCutoff=1e-5, removeInactive=0,
                 removeAboveAverageSlack=False, keepCuts=False, insertActive=False, name=None):
        if name is None:
            name = 'GurobiALPDecoder'
        Decoder.__init__(self, code=code, name=name)
        self.maxRPCrounds = maxRPCrounds
        self.minCutoff = minCutoff
        self.removeInactive = removeInactive
        self.removeAboveAverageSlack = removeAboveAverageSlack
        self.keepCuts = keepCuts
        assert not insertActive
        # initialize Gurobi
        self.model = g.Model()
        self.model.setParam('OutputFlag', 0)
        self.x = [self.model.addVar(0, 1, name='x{}'.format(j)) for j in range(code.blocklength)]
        self.model.update()
        # initialize various structures
        self.hmat = code.parityCheckMatrix
        self.htilde = self.hmat.copy() # the copy is used for gaussian elimination
        self.solution = np.empty(code.blocklength)
        self.diffFromHalf = np.empty(code.blocklength)
        self.fixes = -np.ones(code.blocklength, dtype=np.int)
        self.numConstrs = 0
        self.timer = Timer()
        self.Nj = np.zeros(code.blocklength)
        self.setV = np.zeros(code.blocklength)

    def cutSearchAlgorithm(self, originalMatrix):
        """Runs the cut search algorithm and inserts found cuts. If ``originalHmat`` is True,
        the code-defining parity-check matrix is used for searching, otherwise :attr:`htilde`
        which is the result of Gaussian elimination on the most fractional positions of the last
        LP solution.
        :returns: The number of cuts inserted
        """
        inserted = 0
        matrix = self.hmat if originalMatrix else self.htilde
        for row in matrix:
            #  for each row, we build the set Nj = { j: matrix[row,j] == 1}
            #  and V = {j ∈ Nj: solution[j] > .5}. The variable setV will be of size Njsize and
            #  have 1 and -1 entries, depending on whether the corresponding index belongs to V or
            #  not.
            #  Indices are shifted by 1 to match GLPK's indexing.
            Nj = np.flatnonzero(row)
            if Nj.size == 0:
                # skip all-zero row
                continue
            setV = np.ones(Nj.size)
            sizeV = 0
            for i, j in enumerate(Nj):
                if self.solution[j] <= .5:
                    setV[i] = -1
                else:
                    sizeV += 1
            Xj = [self.x[i] for i in Nj]
            if sizeV % 2 == 0:
                #  V size must be odd, so add entry with minimum distance from .5
                minIndex = np.argmin(self.diffFromHalf[Nj])
                setV[minIndex] *= -1
                sizeV += setV[minIndex]
            vSum = 0 # left hand side of the induced Feldman inequality
            for ind in range(Nj.size):
                if setV[ind] == 1:
                    vSum += 1 - self.solution[Nj[ind]]
                elif setV[ind] == -1:
                    vSum += self.solution[Nj[ind]]
            if vSum < 1 - self.minCutoff:
                # inequality violated -> insert
                inserted += 1
                self.addConstr(Xj, setV, sizeV)
            if originalMatrix and vSum < 1-1e-5:
                #  in this case, we are in the "original matrix" phase and would have a cut for
                #  insertion which is declined because of minCutoff. This implies that we don't
                #  have a codeword although this method may return 0
                self.foundCodeword = self.mlCertificate = False
        if inserted > 0:
            self._stats['cuts'] += inserted
            self.update()
        return inserted

    def addConstr(self, Xj, setV, sizeV):
        lhs = g.LinExpr(setV, Xj)
        self.model.addConstr(lhs, g.GRB.LESS_EQUAL, sizeV - 1)

    def update(self):
        self.model.update()

    def optimize(self):
        self.model.optimize()

    def setStats(self, stats):
        statNames = ["cuts", "totalLPs", "totalConstraints", "ubReached", 'lpTime']
        for item in statNames:
            if item not in stats:
                stats[item] = 0
        Decoder.setStats(self, stats)

    def fix(self, i, val):
        self.x[i].lb = val
        self.x[i].ub = val
        self.fixes[i] = val

    def release(self, i):
        self.x[i].lb = 0
        self.x[i].ub = 1
        self.fixes[i] = -1

    def fixed(self, i):
        """Returns True if and only if the given index is fixed."""
        return self.fixes[i] != -1

    def setLLRs(self, llrs, sent=None):
        for j in range(self.code.blocklength):
            self.x[j].obj = llrs[j]
        Decoder.setLLRs(self, llrs, sent)

    def solve(self, lb=-np.inf, ub=np.inf):
        rpcrounds = 0
        iteration = 0
        if not self.keepCuts:
            self.removeNonfixedConstraints()
        self.foundCodeword = self.mlCertificate = False
        self.objectiveValue = -np.inf
        if self.sent is not None and ub == np.inf:
            # calculate known upper bound on the objective from sent codeword
            ub = np.dot(self.sent, self.llrs) + 2e-6
        while True:
            iteration += 1
            with self.timer:
                self.optimize()
            self._stats['lpTime'] += self.timer.duration
            self._stats["totalLPs"] += 1
            self.numConstrs = self.model.numConstrs
            self._stats["totalConstraints"] += self.numConstrs
            if self.model.Status in (g.GRB.INFEASIBLE, g.GRB.INF_OR_UNBD, g.GRB.UNBOUNDED):
                # during branch-and-bound the problem might become infeasible
                self.objectiveValue = np.inf
                self.foundCodeword = self.mlCertificate = False
                break
            elif self.model.Status != g.GRB.OPTIMAL:
                raise RuntimeError("Gurobi unknown status {}".format(i))
            newObjectiveValue = self.model.ObjVal
            if newObjectiveValue <= self.objectiveValue and iteration > self.code.blocklength:
                # prevent infinite loops in some rare cases where numerical issues cause
                # non-increasing objective value after cut generation
                print('cga: no improvement in iteration {}'.format(iteration))
                break
            self.objectiveValue = newObjectiveValue
            if self.objectiveValue >= ub - 1e-6:
                # lower bound from the LP is above known upper bound -> no need to proceed
                self.objectiveValue = np.inf
                self._stats["ubReached"] += 1
                self.foundCodeword = self.mlCertificate = False
                break
            integral = True
            # read solution from Gurobi. Round values to {0,1} that are very close
            for i in range(self.code.blocklength):
                self.solution[i] = self.x[i].X
            for i in range(self.code.blocklength):
                if self.solution[i] < 1e-6:
                    self.solution[i] = 0
                elif self.solution[i] > 1-1e-6:
                    self.solution[i] = 1
                else:
                    integral = False
            self.diffFromHalf = np.abs(self.solution-.499999)
            if self.removeInactive != 0 and self.numConstrs  >= self.removeInactive:
                self.removeInactiveConstraints()
            self.foundCodeword = self.mlCertificate = True
            numCuts = self.cutSearchAlgorithm(True)
            if numCuts > 0:
                # found cuts from original H matrix
                continue
            elif integral:
                break
            elif rpcrounds >= self.maxRPCrounds and self.maxRPCrounds != -1:
                self.foundCodeword = self.mlCertificate = False
                break
            else:
                # search for RPC cuts
                xindices = np.argsort(self.diffFromHalf)
                mod2la.gaussianElimination(self.htilde, xindices, True)
                numCuts = self.cutSearchAlgorithm(False)
                if numCuts == 0:
                    self.mlCertificate = self.foundCodeword = False
                    break
                rpcrounds += 1

    def removeInactiveConstraints(self):
        """Removes constraints which are not active at the current solution."""
        removed = 0
        #  compute average slack of constraints all constraints, if only those above the average
        # slack should be removed
        if self.removeAboveAverageSlack:
            slacks = self.model.get('Slack', self.model.getConstrs())
            if self.numConstrs == 0:
                avgSlack = 1e-5
            else:
                avgSlack = np.mean(slacks)
        else:
            avgSlack = 1e-5  # some tolerance to avoid removing active constraints
        for constr in self.model.getConstrs():
            if constr.slack > avgSlack:
                removed += 1
                self.model.remove(constr)
                self.numConstrs -= 1
        self.model.update()
        assert self.numConstrs == self.model.numConstrs

    def removeNonfixedConstraints(self):
        """Remove all but the fixed constraints from the model.

        Usually there are no fixed constraints. In case of all-zero decoding, the zero
        constraints are fixed and not removed by this function.
        """
        for constr in self.model.getConstrs():
            self.model.remove(constr)
        self.model.update()
        self.numConstrs = 0

    def params(self):
        params = OrderedDict(name=self.name)
        if self.maxRPCrounds != -1:
            params['maxRPCrounds'] = self.maxRPCrounds
        if self.minCutoff != 1e-5:
            params['minCutoff'] = self.minCutoff
        if self.removeInactive != 0:
            params['removeInactive'] = self.removeInactive
        if self.removeAboveAverageSlack:
            params['removeAboveAverageSlack'] = True
        if self.keepCuts:
            params['keepCuts'] = True
        params['name'] = self.name
        return params
