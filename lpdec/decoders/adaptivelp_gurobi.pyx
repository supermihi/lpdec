# -*- coding: utf-8 -*-
# cython: boundscheck=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: wraparound=False
# cython: initializedcheck=False
# distutils: libraries = ["gurobi60"]
#
# Copyright 2014-2015 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation

from __future__ import division, print_function, unicode_literals
from collections import OrderedDict, deque
import logging
import numpy as np
cimport numpy as np
from numpy.math cimport INFINITY
from libc.math cimport fabs
import gurobimh as g
cimport gurobimh as g
from cython.operator cimport dereference


from lpdec.mod2la cimport gaussianElimination
from lpdec.decoders.base cimport Decoder
from lpdec.decoders import UpperBoundHit, ProblemInfeasible, LimitHit
from lpdec.utils import Timer

logger = logging.getLogger('alp_gurobi')

cdef class AdaptiveLPDecoderGurobi(Decoder):
    """
    Implements the adaptive linear programming decoder with optional generation of redundant
    parity-check (RPC) cuts.

    Uses the gurobimh cython interface.

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
    :param double maxActiveAngle: Maximum angle between LLR vector and an active inequality for
      it to be inserted during the "insert active" step.
    :param bool allZero: If set to ``True``, constraints that are active at the all-zero codeword
      are inserted into the model prior to any decoding and stay there all the time.

    .. attribute:: hint

        An optional "hint" codeword that is somehow believed to be near-optimal. For example,
        this might be the output of an iterative decoder. The adaptive LP decoder can use that
        hint to insert active constraints.
    """

    cdef public bint removeAboveAverageSlack, keepCuts, allZero
    cdef int solveCalls, objBufSize
    cdef public double maxActiveAngle, minCutoff, objBufLim
    cdef public int removeInactive, numConstrs, maxRPCrounds
    cdef np.int_t[:,::1] hmat, htilde
    cdef public g.Model model
    cdef double[::1] diffFromHalf, newSolution, setV
    cdef int[::1] Nj, fixes
    cdef public object timer, xlist
    cdef double[:] objBuff


    def __init__(self, code,
                 maxRPCrounds=-1,
                 minCutoff=1e-5,
                 removeInactive=0,
                 removeAboveAverageSlack=False,
                 keepCuts=False,
                 maxActiveAngle=1,
                 allZero=False,
                 objBufSize=8,
                 objBufLim=0.0,
                 name=None):
        if name is None:
            name = 'ALPDecoderGurobi'
        Decoder.__init__(self, code=code, name=name)
        self.maxRPCrounds = maxRPCrounds
        self.minCutoff = minCutoff
        self.removeInactive = removeInactive
        self.removeAboveAverageSlack = removeAboveAverageSlack
        self.keepCuts = keepCuts
        self.maxActiveAngle=maxActiveAngle
        self.allZero = allZero
        self.model = g.Model('ALP-GUROBIMH')
        self.model.setParam('OutputFlag', 0)
        self.xlist = []
        for i in range(self.code.blocklength):
            self.xlist.append(self.model.addVar(0, 1, g.GRB_CONTINUOUS))
        self.model.update()
        # initialize various structures
        self.hmat = code.parityCheckMatrix
        self.htilde = self.hmat.copy() # the copy is used for gaussian elimination
        self.newSolution = np.empty(code.blocklength)
        self.diffFromHalf = np.empty(code.blocklength)
        self.setV = np.empty(code.blocklength, dtype=np.double)
        self.Nj = np.empty(code.blocklength, dtype=np.intc)
        self.fixes = -np.ones(code.blocklength, dtype=np.intc)
        self.numConstrs = 0
        self.timer = Timer()
        self.objBuff = np.ones(objBufSize, dtype=np.double)
        self.objBufLim = objBufLim

    cdef int cutSearchAlgorithm(self, bint originalHmat) except -3:
        """Runs the cut search algorithm and inserts found cuts. If ``originalHmat`` is True,
        the code-defining parity-check matrix is used for searching, otherwise :attr:`htilde`
        which is the result of Gaussian elimination on the most fractional positions of the last
        LP solution.
        :returns: The number of cuts inserted
        """
        cdef:
            double[::1] setV = self.setV
            int[::1] Nj = self.Nj
            double[::1] solution = self.solution
            np.int_t[:,::1] matrix
            int inserted = 0, row, j, ind, setVsize, minDistIndex, Njsize
            double minDistFromHalf, dist, vSum
        matrix = self.hmat if originalHmat else self.htilde
        for row in range(matrix.shape[0]):
            #  for each row, we build the set Nj = { j: matrix[row,j] == 1}
            #  and V = {j ∈ Nj: solution[j] > .5}. The variable setV will be of size Njsize and
            #  have 1 and -1 entries, depending on whether the corresponding index belongs to V or
            #  not.
            Njsize = 0
            setVsize = 0
            minDistFromHalf = 1
            minDistIndex = -1
            for j in range(matrix.shape[1]):
                if matrix[row, j] == 1:
                    Nj[Njsize] = j
                    if solution[j] > .5:
                        setV[Njsize] = 1
                        setVsize += 1
                    else:
                        setV[Njsize] = -1
                    if self.diffFromHalf[j] < minDistFromHalf:
                        minDistFromHalf = self.diffFromHalf[j]
                        minDistIndex = Njsize
                    elif minDistIndex == -1:
                        minDistIndex = Njsize
                    Njsize += 1
            if Njsize == 0:
                # skip all-zero rows (might occur due to Gaussian elimination)
                continue
            if setVsize % 2 == 0:
                #  V size must be odd, so add entry with minimum distance from .5
                setV[minDistIndex] *= -1
                setVsize += <int>setV[minDistIndex]
            vSum = 0 # left hand side of the induced Feldman inequality
            for ind in range(Njsize):
                if setV[ind] == 1:
                    vSum += 1 - solution[Nj[ind]]
                elif setV[ind] == -1:
                    vSum += solution[Nj[ind]]
            if vSum < 1 - self.minCutoff:
                # inequality violated -> insert
                inserted += 1
                self.model.fastAddConstr2(setV[:Njsize], Nj[:Njsize], g.GRB_LESS_EQUAL, setVsize - 1)
            elif originalHmat and vSum < 1-1e-5:
                #  in this case, we are in the "original matrix" phase and would have a cut for
                #  insertion which is declined because of minCutoff. This implies that we don't
                #  have a codeword although this method may return 0
                self.foundCodeword = self.mlCertificate = False
        if inserted > 0:
            self._stats['cuts'] += inserted
            self.model.update()
        return inserted

    def setStats(self, object stats):
        statNames = ["cuts", "totalLPs", "totalConstraints", "ubReached", 'lpTime', 'simplexIters',
                     'objBufHit']
        for item in statNames:
            if item not in stats:
                stats[item] = 0
        Decoder.setStats(self, stats)

    cpdef fix(self, int i, int val):
        if self.fixes[i] != -1:
            self.release(i)
        if val == 1:
            self.model.setElementDblAttr(b'LB', i, 1)
        else:
            self.model.setElementDblAttr(b'UB', i, 0)
        self.fixes[i] = val

    cpdef release(self, int i):
        self.model.setElementDblAttr(b'LB', i, 0)
        self.model.setElementDblAttr(b'UB', i, 1)
        self.fixes[i] = -1

    def fixed(self, int i):
        """Returns True if and only if the given index is fixed."""
        return self.fixes[i] != -1

    cpdef setLLRs(self, double[::1] llrs, np.int_t[::1] sent=None):
        self.model.fastSetObjective(0, llrs.size, llrs)
        Decoder.setLLRs(self, llrs, sent)
        self.removeNonfixedConstraints()
        self.model.update()

    @staticmethod
    cdef int callbackFunction(g.GRBmodel *model, void *cbdata, int where, void *userdata):
        """Terminates the simplex algorithm if upper bound is hit."""
        cdef double ub = dereference(<double*>userdata)
        cdef double value
        if where == g.GRB_CB_SIMPLEX:
            g.GRBcbget(cbdata, where, g.GRB_CB_SPX_OBJVAL, <void*> &value)
            if value > ub + 1e-6:
                g.GRBterminate(model)

    cpdef solve(self, double lb=-INFINITY, double ub=INFINITY):
        cdef double[::1] solution = self.solution
        cdef double newObjectiveValue
        cdef int i, iteration = 0, rpcrounds = 0
        if not self.keepCuts:
            self.removeNonfixedConstraints()
        self.foundCodeword = self.mlCertificate = False
        self.objectiveValue = -INFINITY
        self.objBuff[:] = -INFINITY
        if self.sent is not None and ub == INFINITY:
            # calculate known upper bound on the objective from sent codeword
            ub = np.dot(self.sent, self.llrs) + 2e-6
        while True:
            iteration += 1
            with self.timer:
                if ub < INFINITY:
                    g.GRBsetcallbackfunc(self.model.model, self.callbackFunction, <void*>&ub)
                self.model.optimize() #self.callback if ub < INFINITY else None)
                if ub < INFINITY:
                    g.GRBsetcallbackfunc(self.model.model, NULL, NULL)
            self._stats['lpTime'] += self.timer.duration
            self._stats["totalLPs"] += 1
            self._stats['simplexIters'] += self.model.IterCount
            self.numConstrs = self.model.NumConstrs
            self._stats["totalConstraints"] += self.numConstrs
            if self.model.Status ==  g.GRB_INFEASIBLE:
                self.objectiveValue = INFINITY
                self.foundCodeword = self.mlCertificate = False
                raise ProblemInfeasible()
            elif self.model.Status == g.GRB_INTERRUPTED and ub < INFINITY:
                self.objectiveValue = ub
                self.foundCodeword = self.mlCertificate = False
                self._stats['ubReached'] += 1
                raise UpperBoundHit()
            elif self.model.Status == g.GRB_ITERATION_LIMIT:
                self.model.fastGetX(0, self.code.blocklength, self.newSolution)
                self.objectiveValue = np.dot(self.llrs, self.newSolution)
                raise LimitHit()
            elif self.model.Status != g.GRB_OPTIMAL:
                raise RuntimeError("Unknown Gurobi status {}".format(self.model.Status))
            newObjectiveValue = self.model.ObjVal
            # if newObjectiveValue <= self.objectiveValue:
            #     print(newObjectiveValue, self.objectiveValue, iteration)
            #     if iteration > self.code.blocklength:
            #         # prevent infinite loops in some rare cases where numerical issues cause
            #         # non-increasing objective value after cut generation
            #         print('cga: no improvement in iteration {}'.format(iteration))
            #         break
            self.objectiveValue = newObjectiveValue
            if self.objectiveValue >= ub - 1e-6:
                # lower bound from the LP is above known upper bound -> no need to proceed
                self.objectiveValue = INFINITY
                self._stats["ubReached"] += 1
                self.foundCodeword = self.mlCertificate = False
                raise UpperBoundHit()
            self.objBuff = np.roll(self.objBuff, 1)
            self.objBuff[0] = self.objectiveValue
            if self.objectiveValue - self.objBuff[self.objBuff.size - 1] < self.objBufLim:
                self.mlCertificate = self.foundCodeword = False
                self._stats['objBufHit'] += 1
                return
            integral = True
            self.model.fastGetX(0, self.newSolution.size, self.newSolution)
            solution[:] = self.newSolution
            for i in range(self.solution.size):
                if solution[i] < 1e-6:
                    solution[i] = 0
                elif solution[i] > 1-1e-6:
                    solution[i] = 1
                else:
                    integral = False
                self.diffFromHalf[i] = fabs(solution[i]-.499999)
            if self.removeInactive != 0 \
                    and self.numConstrs >= self.removeInactive:
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
                gaussianElimination(self.htilde, xindices, True)
                numCuts = self.cutSearchAlgorithm(False)
                if numCuts == 0:
                    self.mlCertificate = self.foundCodeword = False
                    break
                rpcrounds += 1

    cdef void removeInactiveConstraints(self):
        """Removes constraints which are not active at the current solution."""
        cdef int i, removed = 0
        cdef double avgSlack, slack
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
            if constr.Slack > avgSlack:
                removed += 1
                self.model.remove(constr)
                self.numConstrs -= 1
        if removed:
            self.model.update()
        assert self.numConstrs == self.model.numConstrs


    cdef void removeNonfixedConstraints(self):
        """Remove all but the fixed constraints from the model.

        Usually there are no fixed constraints. In case of all-zero decoding, the zero
        constraints are fixed and not removed by this function.
        """
        cdef g.Constr constr
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
        if self.maxActiveAngle != 1:
            params['maxActiveAngle'] = self.maxActiveAngle
        if self.allZero:
            params['allZero'] = True
        if self.objBufLim != 0.0:
            if self.objBufSize != 8:
                params['objBufSize'] = self.objBufSize
            params['objBufLim'] = self.objBufLim
        params['name'] = self.name
        return params
