# -*- coding: utf-8 -*-
# distutils: libraries = ["gurobi65"]
# cython: cdivision=True
# cython: wraparound=False
# cython: language_level=3
# Copyright 2014-2015 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation

from __future__ import division, print_function, unicode_literals
from collections import OrderedDict
import logging
import numpy as np
cimport numpy as np
from numpy.math cimport INFINITY
from libc.math cimport fabs, sqrt, fmin, fmax, atanh, isnan, tanh
import gurobimh as g
cimport gurobimh as g
from cython.operator cimport dereference

from lpdec.gfqla cimport gaussianElimination
from lpdec.decoders.base cimport Decoder
from lpdec.decoders import gurobihelpers
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
    :param gurobiParams: Additional parameters passed to the gurobi model.
    """

    cdef public bint removeAboveAverageSlack, keepCuts, rejected, noSkipOrig
    cdef int objBufSize, fixedConstrs
    cdef public double minCutoff, objBufLim, iterationLimit, sdMin, sdMax, sdX
    cdef public int removeInactive, maxRPCrounds, superDual, cutLimit
    cdef np.int_t[:,::1] hmat, htilde
    cdef np.int_t[::1] row
    cdef np.intp_t[::1] successfulCols
    cdef public g.Model model
    cdef double[::1] setV, fractionality
    cdef int[::1] Nj, fixes
    cdef public object xlist, dualDecoder
    cdef object timer, grbParams, dual4
    cdef double[:] objBuff


    def __init__(self, code,
                 name=None,
                 **kwargs):
        if name is None:
            name = 'ALPDecoder(Gurobi {})'.format('.'.join(str(v) for v in g.gurobi.version()))
        Decoder.__init__(self, code=code, name=name)
        self.maxRPCrounds = kwargs.get('maxRPCrounds', -1)
        self.minCutoff = kwargs.get('minCutoff', 1e-5)
        self.removeInactive = kwargs.get('removeInactive', 0)
        self.removeAboveAverageSlack = kwargs.get('removeAboveAverageSlack', False)
        self.keepCuts = kwargs.get('keepCuts', False)
        gurobiParams = kwargs.get('gurobiParams', {})
        self.model = gurobihelpers.createModel(name, kwargs.get('gurobiVersion'), **gurobiParams)
        self.grbParams = gurobiParams.copy()
        self.superDual = kwargs.get('superDual', 0)
        self.sdMin = kwargs.get('sdMin', .25)
        self.sdMax = kwargs.get('sdMax', .45)
        self.sdX = kwargs.get('sdX', -1)
        self.noSkipOrig = kwargs.get('noSkipOrig', False)
        if self.superDual & 2:
            from lpdec.codes import BinaryLinearBlockCode
            from lpdec.decoders.ip import GurobiIPDecoder
            dualCode = BinaryLinearBlockCode(parityCheckMatrix=code.generatorMatrix, name='dual')
            dualParams = kwargs.get('dualParams', {})
            self.dualDecoder = SuperDualDecoder(dualCode, self, **dualParams)
            if self.superDual & 4:
                self.dual4 = {}
                self.dual4['code'] = dualCode
                model = self.dual4['model'] = g.Model('DUAL')
                model.setParam('OutputFlag', 0)
                xv =  [model.addVar(0, 1, 0.0, vtype=g.GRB.BINARY, name='xv{}'.format(i)) for i in range(code.blocklength)]
                xvb = [model.addVar(0, 1, 0.0, vtype=g.GRB.BINARY, name='xvb{}'.format(i)) for i in range(code.blocklength)]
                z = [model.addVar(0, code.blocklength, 0.0, vtype=g.GRB.INTEGER, name='z{}'.format(i)) for i in range(dualCode.parityCheckMatrix.shape[0])]
                zv = model.addVar(0, code.blocklength, 0.0, vtype=g.GRB.INTEGER, name='zv')
                self.dual4['zv'] = zv
                model.update()
                for i in range(code.blocklength):
                    model.addConstr(xv[i] + xvb[i] <= 1)
                print(g.quicksum(xv) - 2*zv)
                model.addConstr(g.quicksum(xv) - 2*zv, g.GRB.EQUAL, 1)
                for z, row in zip(z, dualCode.parityCheckMatrix):
                    model.addConstr(g.quicksum(xv[i] + xvb[i] for i in np.flatnonzero(row)) - 2*z, g.GRB.EQUAL, 0)
                self.dual4['xv'] = xv
                self.dual4['xvb'] = xvb

        self.xlist = []
        for i in range(self.code.blocklength):
            self.xlist.append(self.model.addVar(0, 1, 0.0, g.GRB_CONTINUOUS))
        self.model.update()
        # initialize various structures
        self.hmat = code.parityCheckMatrix
        self.htilde = self.hmat.copy() # the copy is used for gaussian elimination
        self.fractionality = np.empty(code.blocklength)
        self.setV = np.empty(code.blocklength, dtype=np.double)
        self.Nj = np.empty(code.blocklength, dtype=np.intc)
        self.fixes = -np.ones(code.blocklength, dtype=np.intc)
        self.timer = Timer()
        self.objBufSize = kwargs.get('objBufSize', 8)
        self.objBuff = np.ones(self.objBufSize, dtype=np.double)
        self.objBufLim = kwargs.get('objBufLim', 0.0)
        self.iterationLimit = INFINITY
        self.row = np.zeros(code.blocklength, dtype=np.int)
        self.successfulCols = np.empty(code.blocklength, dtype=np.intp)
        self.cutLimit = kwargs.get('cutLimit', 0)
        self.fixedConstrs = 0

    def searchCut(self, np.int_t[::1] dual):
        return self.searchCutFromDualCodeword(dual)

    cdef int searchCutFromDualCodeword(self, np.int_t[::1] dual) except -2:
        """Search for a cut in the given dual codeword and insert it if its cutoff exceeds
        self.minCutoff.

        Returns:
          1 in cut found and inserted, -1 if cut fouINFINITind but rejected by min cutoff rule, 0 else
        """
        cdef int Njsize = 0, setVsize = 0, maxFracIndex = -1
        cdef int j, ind
        cdef double cutoff, vSum = 0, maxFractionality = 0
        for j in range(dual.shape[0]):
            if dual[j] == 1:
                self.Nj[Njsize] = j
                if self.solution[j] > .5:
                    self.setV[Njsize] = 1
                    setVsize += 1
                else:
                    self.setV[Njsize] = -1
                if self.fractionality[j] > maxFractionality:
                    maxFractionality = self.fractionality[j]
                    maxFracIndex = Njsize
                elif maxFracIndex == -1:
                    maxFracIndex = Njsize
                Njsize += 1
        if Njsize == 0:
            # skip all-zero rows (might occur due to Gaussian elimination)
            return 0
        if setVsize % 2 == 0:
            #  V size must be odd, so add entry with maximum fractionality
            self.setV[maxFracIndex] *= -1
            setVsize += <int>self.setV[maxFracIndex]
        for ind in range(Njsize):
            if self.setV[ind] == 1:
                vSum += self.solution[self.Nj[ind]]
            elif self.setV[ind] == -1:
                vSum -= self.solution[self.Nj[ind]]
        cutoff = (vSum - setVsize + 1) / sqrt(Njsize)
        if cutoff > self.minCutoff:
            # inequality violated -> insert
            self.model.fastAddConstr2(self.setV[:Njsize], self.Nj[:Njsize], b'<', setVsize - 1)
            return 1
        elif cutoff > 1e-5:
            return -1
        return 0

    cdef int cutSearchAlgorithm(self, np.int_t[:,::1] matrix, bint originalHmat) except -1:
        """Runs the cut search algorithm and inserts found cuts. If ``originalHmat`` is True,
        the code-defining parity-check matrix is used for searching, otherwise :attr:`htilde`
        which is the result of Gaussian elimination on the most fractional positions of the last
        LP solution.
        :returns: The number of cuts inserted
        """
        cdef int inserted = 0, ans, row
        for row in range(matrix.shape[0]):
            ans = self.searchCutFromDualCodeword(matrix[row, :])
            if ans == 1:
                inserted += 1
                if originalHmat:
                    self._stats['cutsFromOrig'] += 1
            elif ans == -1:
                self._stats['minCutoffFailed'] += 1
                self.rejected = True
        if inserted > 0:
            self._stats['cuts'] += inserted
            self.model.update()
        return inserted

    def setStats(self, object stats):
        statNames = ["cuts", "totalLPs", "totalConstraints", "ubReached", 'ubReachedC', 'lpTime', 'simplexIters',
                     'objBufHit', 'infeasible', 'iterLimitHit', 'minCutoffFailed', 'rpcRounds',
                     'cutsFromOrig', 'sdFound', 'sdSearch', 'cutLimitHit', 'sd2Found']
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
        if self.fixes[i] == -1:
            return
        self.model.setElementDblAttr(b'LB', i, 0)
        self.model.setElementDblAttr(b'UB', i, 1)
        self.fixes[i] = -1

    cpdef fixed(self, int i):
        return self.fixes[i] != -1

    cpdef setLLRs(self, double[::1] llrs, np.int_t[::1] sent=None):
        self.model.fastSetObjective(0, llrs.size, llrs)
        self.fixedConstrs = 0
        self.removeNonfixedConstraints()
        Decoder.setLLRs(self, llrs, sent)

        self.model.update()

    @staticmethod
    cdef int callbackFunction(g.GRBmodel *model, void *cbdata, int where, void *userdata):
        """Terminates the simplex algorithm if upper bound is hit."""
        cdef double ub = dereference(<double*>userdata)
        cdef double value
        if where == g.GRB_CB_SIMPLEX:
            g.GRBcbget(cbdata, where, g.GRB_CB_SPX_OBJVAL, <void*> &value)
            if value > ub - 1e-6:
                g.GRBterminate(model)

    cpdef solve(self, double lb=-INFINITY, double ub=INFINITY):
        cdef double[::1] solution = self.solution
        cdef np.intp_t[::1] unitCols
        cdef np.intp_t[:] xindices
        cdef double spxIters = 0
        cdef int i, iteration = 0, rpcrounds = 0, numCuts, totalCuts = 0
        if not self.keepCuts:
            self.removeNonfixedConstraints()
        self.foundCodeword = self.mlCertificate = False
        self.objectiveValue = -INFINITY
        self.objBuff[:] = -INFINITY
        self.status = Decoder.OPTIMAL
        while True:
            iteration += 1
            self.timer.start()
            if ub < INFINITY:
                g.GRBsetcallbackfunc(self.model.model, self.callbackFunction, <void*>&ub)
            self.model.optimize()
            if ub < INFINITY:
                g.GRBsetcallbackfunc(self.model.model, NULL, NULL)
            self._stats['lpTime'] += self.timer.stop()
            self._stats["totalLPs"] += 1
            self._stats['simplexIters'] += self.model.IterCount
            spxIters += self.model.IterCount
            self._stats['totalConstraints'] += self.model.NumConstrs
            self.model.fastGetX(0, self.solution.shape[0], self.solution)
            if spxIters > self.iterationLimit:
                self.foundCodeword = self.mlCertificate = (self.solution in self.code)
                self._stats['iterLimitHit'] += 1
                self.status = Decoder.LIMIT_HIT
                return
            if self.model.Status == g.GRB_INFEASIBLE:
                self.objectiveValue = INFINITY
                self.foundCodeword = self.mlCertificate = False
                self._stats['infeasible'] += 1
                self.status = Decoder.INFEASIBLE
                return
            elif self.model.Status == g.GRB_INTERRUPTED and ub < INFINITY:
                # interrupted by callback
                self.objectiveValue = ub
                self._stats['ubReachedC'] += 1
                self.foundCodeword = self.mlCertificate = (self.solution in self.code)
                self.status = Decoder.UPPER_BOUND_HIT
                return
            elif self.model.Status == g.GRB_ITERATION_LIMIT:
                self.objectiveValue = np.dot(self.llrs, self.solution)
                # self.foundCodeword = self.mlCertificate = (self.solution in self.code)
                # self._stats['iterLimitHit'] += 1
                # self.status = Decoder.LIMIT_HIT
                # return
            elif self.model.Status == g.GRB_OPTIMAL:
                self.objectiveValue = self.model.ObjVal
            else:
                raise RuntimeError("Unknown Gurobi status {}".format(self.model.Status))

            if self.objectiveValue > ub - 1e-6:
                # lower bound from the LP is above known upper bound -> no need to proceed
                #self.objectiveValue = ub
                self._stats["ubReached"] += 1
                self.foundCodeword = self.mlCertificate = (self.solution in self.code)
                self.status = Decoder.UPPER_BOUND_HIT
                return
            if self.objBufLim != 0.0 and self.objBufSize > 1:
                self.objBuff = np.roll(self.objBuff, 1)
                self.objBuff[0] = self.objectiveValue
                if self.objectiveValue - self.objBuff[self.objBuff.shape[0] - 1] < self.objBufLim:
                    self.mlCertificate = self.foundCodeword = (self.solution in self.code)
                    self._stats['objBufHit'] += 1
                    return
            if self.cutLimit != 0 and totalCuts > self.cutLimit:
                self._stats['cutLimitHit'] += 1
                self.foundCodeword = self.mlCertificate = (self.solution in self.code)
                self.status = Decoder.LIMIT_HIT
                return
            integral = True
            for i in range(self.solution.shape[0]):
                if solution[i] < 1e-6:
                    solution[i] = 0
                elif solution[i] > 1-1e-6:
                    solution[i] = 1
                else:
                    integral = False
                self.fractionality[i] = .5 - fabs(solution[i] - .499999)
            if self.removeInactive != 0 \
                    and self.model.NumConstrs >= self.removeInactive:
                self.removeInactiveConstraints()
            self.foundCodeword = self.mlCertificate = True
            self.rejected = False
            numCuts = self.cutSearchAlgorithm(self.hmat, True)
            if self.noSkipOrig and self.rejected and numCuts == 0:
                origCut = self.minCutoff
                self.minCutoff = 1e-5
                numCuts = self.cutSearchAlgorithm(self.hmat, True)
                self.minCutoff = origCut
            if numCuts > 0:
                # found cuts from original H matrix
                totalCuts += numCuts
                continue
            elif integral:
                self.foundCodeword = self.mlCertificate = (self.solution in self.code)
                break
            elif rpcrounds >= self.maxRPCrounds and self.maxRPCrounds != -1:
                self.foundCodeword = self.mlCertificate = False
                break
            else:
                # search for RPC cuts
                self._stats['rpcRounds'] += 1
                xindices = np.argsort(self.fractionality)[::-1]

                unitCols = gaussianElimination(self.htilde, xindices, True, self.successfulCols)
                numCuts = self.cutSearchAlgorithm(self.htilde, False)
                totalCuts += numCuts
                if numCuts == 0:
                    if self.superDual & 1:
                        numCuts = self.superDualSearch(unitCols, xindices)
                        if numCuts > 0:
                            totalCuts += numCuts
                            continue
                    if self.superDual & 2:
                        dLLRs = np.zeros(self.llrs.shape[0])
                        for i in range(self.solution.size):
                            dLLRs[i] = .5 - fabs(.50001 - self.solution[i])
                        self.dualDecoder.setLLRs(dLLRs)
                        found = self.dualDecoder.search()
                        if found > 0:
                            self._stats['sd2Found'] += found
                            continue
                    if self.superDual & 4:
                        model = self.dual4['model']
                        model.setObjective(g.LinExpr(np.asarray(self.solution), self.dual4['xvb'])
                                                         + g.LinExpr(1 - np.asarray(self.solution), self.dual4['xv']),g.GRB.MINIMIZE)
                        model.optimize()
                        if model.Status == g.GRB.OPTIMAL and model.ObjVal < 1-1e-5:
                            #print(self.dual4['model'].ObjVal)
                            xvVal = [int(x.X) for x in self.dual4['xv']]
                            xvbVal = [int(x.X) for x in self.dual4['xvb']]
                            #print(np.add(xvVal, xvbVal))
                            #print([x.X for x in self.dual4['xv']])
                            #print([x.X for x in self.dual4['xvb']])
                            #print(np.dot(self.solution, xvVal))
                            if self.searchCutFromDualCodeword(np.add(xvVal, xvbVal)) == 1:
                                print('found SD4!')
                                continue
                    self.mlCertificate = self.foundCodeword = False
                    break
                rpcrounds += 1

    cdef int superDualSearch(self, np.intp_t[::1] unitCols, np.intp_t[:] xindices) except -1:
        cdef int found = 0
        cdef int minInd = -1, maxInd = unitCols.shape[0] - 1
        cdef int i, j, k
        cdef double sigma = 0, estimate
        if self.sdX != -1:
            i = unitCols[unitCols.shape[0] - 1]
            for j in range(xindices.shape[0]):
                if xindices[j] == i:
                    i = j + 1
                    break
            # now xindices[i] is the first entry that is not a unit column
            for k in range(i, xindices.shape[0]):
                if self.fractionality[xindices[k]] > 1e-6:
                    sigma += self.fractionality[xindices[k]]
                else:
                    break
            for i in range(unitCols.shape[0] - 1, 0, -1):
                estimate = sigma * .35 + self.fractionality[unitCols[i]]
                if estimate > self.sdX:
                    break
                self.row[:] = self.htilde[i, :]
                for j in range(i - 1, -1, -1):
                    if estimate + self.fractionality[unitCols[j]] > self.sdX:
                        break
                    for k in range(self.row.shape[0]):
                        self.row[k] ^= self.htilde[j, k]
                    self._stats['sdSearch'] += 1
                    ans = self.searchCutFromDualCodeword(self.row)
                    if ans == 1:
                        found += 1
                    if j > 0:
                        for k in range(self.row.shape[0]):
                            self.row[k] ^= self.htilde[j, k]
        else:
            for i in range(unitCols.shape[0]):
                if minInd == -1 and fabs(.5 - self.solution[unitCols[i]]) >= self.sdMin:
                    minInd = i
                if fabs(.5 - self.solution[unitCols[i]]) >= self.sdMax:
                    maxInd = i
                    break
            for i in range(minInd, maxInd):
                self.row[:] = self.htilde[i, :]
                for j in range(i+1, maxInd + 1):
                    for k in range(self.row.shape[0]):
                        self.row[k] ^= self.htilde[j, k]
                    self._stats['sdSearch'] += 1
                    ans = self.searchCutFromDualCodeword(self.row)
                    if ans == 1:
                        found += 1
                    if j < maxInd:
                        for k in range(self.row.shape[0]):
                            self.row[k] ^= self.htilde[j, k]
        if found:
            self._stats['cuts'] += found
            self._stats['sdFound'] += found
            self.model.update()
        return found


    cdef void removeInactiveConstraints(self):
        """Removes constraints which are not active at the current solution."""
        cdef int i, removed = 0
        cdef double avgSlack, slack
        cdef g.Constr constr
        #  compute average slack of constraints all constraints, if only those above the average
        # slack should be removed
        if self.removeAboveAverageSlack:
            slacks = self.model.get('Slack', self.model.getConstrs()[self.fixedConstrs:])
            if self.model.NumConstrs == 0:
                avgSlack = 1e-5
            else:
                avgSlack = np.mean(slacks)
        else:
            avgSlack = 1e-5  # some tolerance to avoid removing active constraints
        for constr in self.model.getConstrs()[self.fixedConstrs:]:
            if self.model.getElementDblAttr(b'Slack', constr.index) > avgSlack:
                removed += 1
                self.model.remove(constr)
        if removed:
            self.model.optimize()


    cdef void removeNonfixedConstraints(self):
        """Remove all but the fixed constraints from the model.

        Usually there are no fixed constraints. In case of all-zero decoding, the zero
        constraints are fixed and not removed by this function.
        """
        cdef g.Constr constr
        for constr in self.model.getConstrs()[self.fixedConstrs:]:
            self.model.remove(constr)
        self.model.update()
                   
    def fixCurrentConstrs(self):
        self.removeInactiveConstraints()
        self.fixedConstrs = self.model.NumConstrs

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
        if self.cutLimit != 0:
            params['cutLimit'] = self.cutLimit
        if self.objBufLim != 0.0:
            if self.objBufSize != 8:
                params['objBufSize'] = self.objBufSize
            params['objBufLim'] = self.objBufLim
        if self.superDual:
            params['superDual'] = self.superDual
            if self.sdX != -1:
                params['sdX'] = self.sdX
            else:
                if self.sdMin != .25:
                    params['sdMin'] = self.sdMin
                if self.sdMax != .45:
                    params['sdMax'] = self.sdMax
            if self.superDual & 2:
                if len(self.dualDecoder.params()) > 0:
                    params['dualParams'] = self.dualDecoder.params()
        if self.noSkipOrig:
            params['noSkipOrig'] = True
        if len(self.grbParams):
            params['gurobiParams'] = self.grbParams
        params['gurobiVersion'] = '.'.join(str(v) for v in g.gurobi.version())
        params['name'] = self.name
        return params

cdef class SuperDualDecoder(Decoder):

    cdef:
        int[:]    checkNodeSatStates
        double[:] varSoftBits
        np.int_t[:]    varHardBits
        int[:]    varNodeDegree
        int[:]    checkNodeDegree
        np.intp_t[:,::1]  varNeighbors
        np.intp_t[:,::1]  checkNeighbors
        double[:,:]  varToChecks
        double[:,:]  checkToVars
        double[:] fP, bP
        double[:] fixes
        int            iterations
        int            reencodeOrder
        # helpers for the order-i reprocessing
        int[:]    syndrome, varDeg2, fixSyndrome
        np.int_t[::1] candidate
        np.intp_t[:]   indices, pool
        np.intp_t[:,:] varNeigh2
        int            maxRange
        double         reencodeRange
        np.int_t  [:,::1] matrix
        Decoder alpDecoder

    def __init__(self, code, alpDecoder, iterations=100, reencodeOrder=2, reencodeRange=1):
        name = 'X'
        Decoder.__init__(self, code, name)
        self.name = name
        self.alpDecoder = alpDecoder
        self.reencodeRange = reencodeRange
        self.iterations = iterations
        self.reencodeOrder = reencodeOrder
        mat = self.code.parityCheckMatrix
        k, n = code.parityCheckMatrix.shape
        self.syndrome = np.zeros(code.blocklength, dtype=np.intc)
        self.candidate = np.zeros(code.blocklength, dtype=np.int)
        self.indices = np.zeros(reencodeOrder, dtype=np.intp)
        self.matrix = mat.copy()
        self.pool = np.zeros(code.blocklength, dtype=np.intp)
        self.varNeigh2 = np.zeros((n, k), dtype=np.intp)
        self.varDeg2 = np.zeros(code.blocklength, dtype=np.intc)
        self.fixSyndrome = np.zeros(code.blocklength, dtype=np.intc)
        self.fixes = np.zeros(n, dtype=np.double)
        self.checkNodeSatStates = np.empty(k, dtype=np.intc)
        self.varSoftBits = np.empty(n, dtype=np.double)
        self.varHardBits = np.empty(n, dtype=np.int)
        self.varNodeDegree = np.empty(n, dtype=np.intc)
        self.checkNodeDegree = np.empty(k, dtype=np.intc)
        self.varToChecks = np.empty( (n, k), dtype=np.double)
        self.checkToVars = np.empty( (k, n), dtype=np.double)
        self.checkNeighbors = np.empty( (k, n), dtype=np.intp)
        self.varNeighbors = np.empty( (n, k), dtype=np.intp)
        self.fP = np.empty(n+1, dtype=np.double)
        self.bP = np.empty(n+1, dtype=np.double)

        for j in range(n):
            self.varNodeDegree[j] = 0
            for i in range(k):
                if mat[i, j]:
                    self.varNeighbors[j, self.varNodeDegree[j]] = i
                    self.varNodeDegree[j] += 1
        for i in range(k):
            self.checkNodeDegree[i] = 0
            for j in range(n):
                if mat[i, j]:
                    self.checkNeighbors[i, self.checkNodeDegree[i]] = j
                    self.checkNodeDegree[i] += 1

    def setStats(self, object stats):
        for param in 'iterations', 'noncodewords':
            if param not in stats:
                stats[param] = 0
        Decoder.setStats(self, stats)

    cpdef fix(self, int index, int val):
        """Variable fixing is implemented by adding :attr:`fixes` to the LLRs. This vector
        contains :math:`\infty` for a fix to zero and :math:`-\infty` for a fix to one.
        """
        self.fixes[index] = (.5 - val) * INFINITY

    cpdef release(self, int index):
        self.fixes[index] = 0

    def search(self):
        cdef:
            int[:]      checkNodeSatStates = self.checkNodeSatStates
            np.int_t[:]      varHardBits = self.varHardBits
            int[:]      varNodeDegree = self.varNodeDegree
            int[:]      checkNodeDegree = self.checkNodeDegree
            np.intp_t[:,:]    varNeighbors = self.varNeighbors, checkNeighbors = self.checkNeighbors
            double[:,:] varToChecks = self.varToChecks, checkToVars = self.checkToVars
            double[:]   varSoftBits = self.varSoftBits, bP = self.bP, fP = self.fP
            int i, j, deg, iteration, checkIndex, varIndex
            int numVarNodes = self.code.blocklength
            int numCheckNodes = self.code.parityCheckMatrix.shape[0], foundCuts = 0
            bint codeword, sign

        # reset messages
        for j in range(numVarNodes):
            for i in range(varNodeDegree[j]):
                varToChecks[j, varNeighbors[j, i]] = 0
        for i in range(numCheckNodes):
            for j in range(checkNodeDegree[i]):
                checkToVars[i, checkNeighbors[i, j]] = 0
        # reset satisfy state of check nodes
        for i in range(numCheckNodes):
            checkNodeSatStates[i] = False

        iteration = 0
        while iteration < self.iterations:
            iteration += 1
            # variable node processing
            for i in range(numVarNodes):
                varSoftBits[i] = self.llrs[i] + self.fixes[i]
                for j in range(varNodeDegree[i]):
                    varSoftBits[i] += checkToVars[varNeighbors[i,j], i]
                    if isnan(varSoftBits[i]):
                        # this might happen if contradicting bits are fixed
                        return 0
                varHardBits[i] = ( varSoftBits[i] <= 0 )
                for j in range(varNodeDegree[i]):
                    checkIndex = varNeighbors[i,j]
                    varToChecks[i, checkIndex] = varSoftBits[i] - checkToVars[checkIndex, i]
                    checkNodeSatStates[checkIndex] ^= varHardBits[i]
            # check node processing
            codeword = True
            for i in range(numCheckNodes):
                deg = checkNodeDegree[i]
                if checkNodeSatStates[i]:
                    codeword = checkNodeSatStates[i] = False # reset for next iteration
                fP[0] = bP[deg] = 1
                for j in range(deg):
                    varIndex = checkNeighbors[i,j]
                    fP[j+1] = fP[j]*tanh(varToChecks[varIndex, i]/2)
                    varIndex = checkNeighbors[i, deg-j-1]
                    bP[deg-1-j] = bP[deg-j]*tanh(varToChecks[varIndex, i]/2)
                for j in range(deg):
                    checkToVars[i, checkNeighbors[i,j]] = fmax(-1e9, fmin(2*atanh(fP[j]*bP[
                            j+1]), 1e9)) # avoid infinity values
            if codeword:
                if self.alpDecoder.searchCut(varHardBits) == 1:
                    foundCuts += 1
                break
        foundCuts += self.reprocess()
        return foundCuts

    cdef int reprocess(self) except -2:
        """Perform order-i reprocessing, where i is given by :attr:`self.reencodeOrder`.
        """
        cdef int mod2sum, i, j, index, order, poolSize = 0, poolRange, foundCuts = 0
        cdef double objVal
        cdef np.intp_t[:] sorted = np.argsort(np.abs(self.varSoftBits))
        cdef np.intp_t[:] indices = self.indices, pool = self.pool
        cdef np.int_t[::1] candidate = self.candidate
        cdef int[:] syndrome = self.syndrome, fixSyndrome = self.fixSyndrome
        cdef np.int_t[:] varHardBits = self.varHardBits
        cdef int[:] varDeg = self.varDeg2
        cdef np.int_t[:,::1] matrix = self.matrix
        cdef np.intp_t[:,:] varNeigh = self.varNeigh2
        cdef double[:] fixes = self.fixes
        cdef np.intp_t[:] unit = gaussianElimination(matrix, sorted, True)

        for j in unit:
            if fixes[j] != 0:
                # unit column is fixed -> not enough "free" unit columns -> no reprocessing possible
                return 0
        # first, data structures are built up.
        # pool: contains the variable indices that are neither part of the unit matrix nor fixed,
        #   ie, the pool of indices that can be flipped during order-i reprocessing. poolSize
        #   indicates what part of pool is valid.
        # fixSyndrome: per check, stores the syndrome (mod-2 sum) of the fixed columns of H.
        for j in range(matrix.shape[0]):
            fixSyndrome[j] = 0
        for i in range(self.code.blocklength):
            j = sorted[i]
            if j not in unit:
                if fixes[j] == 0:
                    pool[poolSize] = j
                    poolSize += 1
                elif fixes[j] == -INFINITY:
                    for row in range(matrix.shape[0]):
                        fixSyndrome[row] ^= matrix[row, j]
        # poolRange: one plus maximum pool index for flipping due to reencodeRange limitation.
        poolRange = int(poolSize * self.reencodeRange)
        # varDeg: record variable degrees of indices in pool (wrt Gaussed matrix)
        # varNeigh: record neighborhood of variables in pool (wrt Gaussed matrix)
        for j in range(poolSize):
            varDeg[j] = 0
            for i in range(matrix.shape[0]):
                if matrix[i, pool[j]] == 1:
                    varNeigh[j, varDeg[j]] = i
                    varDeg[j] += 1

        for order in range(0, self.reencodeOrder+1):
            # we need at least `order` flippable positions in the allowed pool range!
            if order > poolRange:
                break
            # reset candidate and syndrome
            for j in range(self.code.blocklength):
                candidate[j] = <int>varHardBits[j]
            for row in range(matrix.shape[0]):
                syndrome[row] = fixSyndrome[row]
            for j in range(poolSize):
                if candidate[pool[j]]:
                    for i in range(varDeg[j]):
                        syndrome[varNeigh[j, i]] ^= 1
            # now, syndrome reflects the syndrome of the matrix except for the unit part

            # this is inspired by the example implementation of itertools.combinations in the
            # python docs
            # First, flip bits 0, ..., order-1, and reencode this configuration.
            for i in range(order):
                indices[i] = i
                candidate[pool[indices[i]]] ^= 1
                for j in range(varDeg[indices[i]]):
                    syndrome[varNeigh[indices[i], j]] ^= 1
            # reencode
            for row in range(unit.shape[0]):
                candidate[unit[row]] = syndrome[row]
            if self.alpDecoder.searchCut(candidate) == 1:
                foundCuts += 1
            # now, move the `order` positions through all configurations among the feasible pool.
            while True:
                for i in range(order - 1, -1, -1):
                    if indices[i] != i + poolRange - order:
                        break
                else:
                    break
                index = pool[indices[i]]
                candidate[pool[indices[i]]] ^= 1
                for j in range(varDeg[indices[i]]):
                    syndrome[varNeigh[indices[i], j]] ^= 1
                indices[i] += 1
                index = pool[indices[i]]
                candidate[pool[indices[i]]] ^= 1
                for j in range(varDeg[indices[i]]):
                    syndrome[varNeigh[indices[i], j]] ^= 1
                for j in range(i + 1, order):
                    candidate[pool[indices[j]]] ^= 1
                    for index in range(varDeg[indices[j]]):
                        syndrome[varNeigh[indices[j], index]] ^= 1
                    indices[j] = indices[j-1] + 1
                    candidate[pool[indices[j]]] ^= 1
                    for index in range(varDeg[indices[j]]):
                        syndrome[varNeigh[indices[j], index]] ^= 1
                # reencode
                for row in range(unit.shape[0]):
                    candidate[unit[row]] = syndrome[row]
                if self.alpDecoder.searchCut(candidate) == 1:
                    foundCuts += 1
        return foundCuts

    def params(self):
        parms = OrderedDict()
        if self.iterations != 100:
            parms['iterations'] = self.iterations
        if self.reencodeOrder != 2:
            parms['reencodeOrder'] = self.reencodeOrder
        if self.reencodeRange != 1:
            parms['reencodeRange'] = self.reencodeRange
        return parms
