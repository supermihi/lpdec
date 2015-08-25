# -*- coding: utf-8 -*-
# distutils: libraries = ["glpk", "m"]
# cython: boundscheck=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: wraparound=False
#
# Copyright 2014-2015 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation

from __future__ import division, print_function
from collections import OrderedDict
import logging
import numpy as np
cimport numpy as np
from numpy.math cimport INFINITY
from libc.math cimport fabs, sqrt

from lpdec.decoders cimport glpk

from lpdec.gfqla cimport gaussianElimination
from lpdec.decoders.base cimport Decoder
from lpdec.utils import Timer

logger = logging.getLogger('alp')

cdef class AdaptiveLPDecoder(Decoder):
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

    cdef public bint removeAboveAverageSlack, keepCuts, variableFixing
    cdef bint glpkVerbose
    cdef int nrFixedConstraints, insertActive, solveCalls, objBufSize
    cdef public double maxActiveAngle, minCutoff, objBufLim
    cdef public int removeInactive, numConstrs, maxRPCrounds
    cdef np.int_t[:,::1] hmat, htilde
    cdef glpk.glp_prob *prob
    cdef glpk.glp_smcp parm
    cdef double[::1] setV, fractionality
    cdef int[::1] Nj, fixes
    cdef public object timer, erasureDecoder
    cdef double[:] objBuff

    def __init__(self, code, name=None, **kwargs):
        if name is None:
            name = 'AdaptiveLPDecoder'
        Decoder.__init__(self, code=code, name=name)

        self.parseArgs(**kwargs)
        self.initStructures()

        # initialize GLPK
        self.prob = NULL
        self.initGLPK()

    def parseArgs(self, **kwargs):
        self.maxRPCrounds = kwargs.get('maxRPCrounds', -1)
        self.minCutoff = kwargs.get('minCutoff', 1e-5)

        self.keepCuts = kwargs.get('keepCuts', False)
        self.removeInactive = kwargs.get('removeInactive', 0)
        self.removeAboveAverageSlack = kwargs.get('removeAboveAverageSlack', False)

        self.objBufSize = kwargs.get('objBufSize', 8)
        self.objBuff = np.ones(self.objBufSize, dtype=np.double)
        self.objBufLim = kwargs.get('objBufLim', 0.0)

        self.glpkVerbose = kwargs.get('glpkVerbose', False)

    def initStructures(self):
        self.hmat = self.code.parityCheckMatrix
        self.htilde = self.hmat.copy() # the copy is used for gaussian elimination
        self.solution = np.empty(self.code.blocklength)
        self.fractionality = np.empty(self.code.blocklength)
        self.setV = np.empty(1 + self.code.blocklength, dtype=np.double)
        self.Nj = np.empty(1 + self.code.blocklength, dtype=np.intc)
        self.fixes = -np.ones(self.code.blocklength, dtype=np.intc)
        self.numConstrs = 0
        self.timer = Timer()

    def initGLPK(self):
        if self.prob != NULL:
            glpk.glp_erase_prob(self.prob)
        else:
            self.prob = glpk.glp_create_prob()
        glpk.glp_init_smcp(&self.parm)
        if not self.glpkVerbose:
            self.parm.msg_lev = glpk.GLP_MSG_OFF
        self.parm.meth = glpk.GLP_DUAL # use dual simplex method
        # self.parm.tol_bnd = 1e-9 never change tolerances! :)
        # self.parm.tol_dj = 1e-9
        glpk.glp_add_cols(self.prob, self.code.blocklength)
        for j in range(self.code.blocklength):
            glpk.glp_set_col_bnds(self.prob, 1+j, glpk.GLP_DB, 0.0, 1.0)
        glpk.glp_set_obj_dir(self.prob, glpk.GLP_MIN)

    cdef int cutSearchAlgorithm(self, np.int_t[:,::1] matrix) except -1:
        """Runs the cut search algorithm and inserts found cuts.
        :returns: The number of cuts inserted
        """
        cdef int ans, row, inserted = 0
        for row in range(matrix.shape[0]):
            ans = self.searchCutFromDualCodeword(matrix[row, :])
            if ans == 1:
                inserted += 1
            elif ans == -1:
                self._stats['minCutoffFailed'] += 1
        if inserted > 0:
            self._stats['cuts'] += inserted
        return inserted

    cdef int searchCutFromDualCodeword(self, np.int_t[::1] dual) except -2:
        """Search for a cut in the given dual codeword and insert it if its cutoff exceeds
        self.minCutoff.

        Returns:
          1 in cut found and inserted, -1 if cut found but rejected by min cutoff rule, 0 else
        """
        cdef int Njsize = 0, setVsize = 0, maxFracIndex = -1
        cdef int j, ind
        cdef double cutoff, vSum = 0, maxFractionality = 0
        for j in range(dual.shape[0]):
            if dual[j] == 1:
                self.Nj[1 + Njsize] = j
                if self.solution[j] > .5:
                    self.setV[1 + Njsize] = 1
                    setVsize += 1
                else:
                    self.setV[1 + Njsize] = -1
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
            self.setV[1 + maxFracIndex] *= -1
            setVsize += <int>self.setV[1 + maxFracIndex]
        for ind in range(Njsize):
            if self.setV[1 + ind] == 1:
                vSum += self.solution[self.Nj[1 + ind]]
            elif self.setV[1 + ind] == -1:
                vSum -= self.solution[self.Nj[1 + ind]]
        cutoff = (vSum - setVsize + 1) / sqrt(Njsize)
        if cutoff > self.minCutoff:
            # inequality violated -> insert
            ind = glpk.glp_add_rows(self.prob, 1)
            for j in range(Njsize):
                self.Nj[1 + j] += 1 # GLPK indexing: shift indexes by one
            glpk.glp_set_row_bnds(self.prob, ind, glpk.GLP_UP, 0.0, setVsize - 1)
            glpk.glp_set_mat_row(self.prob, ind, Njsize, &(self.Nj[0]), &(self.setV[0]))
            return 1
        elif cutoff > 1e-5:
            return -1
        return 0

    def setStats(self, object stats):
        statNames = ['cuts', 'totalLPs', 'totalConstraints', 'minCutoffFailed', 'ubReached',
                     'objBufHit', 'lpTime', 'infeasible', 'rpcRounds']
        for item in statNames:
            if item not in stats:
                stats[item] = 0
        Decoder.setStats(self, stats)

    cpdef fix(self, int i, int val):
        glpk.glp_set_col_bnds(self.prob, 1 + i, glpk.GLP_FX, val, val)
        self.fixes[i] = val

    cpdef release(self, int i):
        glpk.glp_set_col_bnds(self.prob, 1 + i, glpk.GLP_DB, 0.0, 1.0)
        self.fixes[i] = -1

    cpdef fixed(self, int i):
        return self.fixes[i] != -1

    cpdef setLLRs(self, double[::1] llrs, np.int_t[::1] sent=None):
        cdef int j
        self.solveCalls = 0
        for j in range(self.code.blocklength):
            glpk.glp_set_obj_coef(self.prob, 1+j, llrs[j])
        Decoder.setLLRs(self, llrs, sent)

    cpdef solve(self, double lb=-np.inf, double ub=np.inf):
        cdef int removed, error, status, numCuts, rpcrounds = 0, iteration = 0, totalCuts = 0, i
        cdef double[::1] solution = self.solution
        cdef np.intp_t[::1] unitCols
        cdef np.intp_t[:] xindices
        cdef bint integral

        if not self.keepCuts:
            self.removeNonfixedConstraints()

        self.foundCodeword = self.mlCertificate = False
        self.objectiveValue = -INFINITY
        self.objBuff[:] = -INFINITY
        self.status = Decoder.OPTIMAL

        while True:
            iteration += 1

            # solve LP
            self.timer.start()
            self.parm.obj_ul = ub
            error = glpk.glp_simplex(self.prob, &self.parm)
            self._stats['lpTime'] += self.timer.stop()
            if error not in  (0, glpk.GLP_EOBJUL):
                raise RuntimeError('GLPK Simplex Error ({}) {}'
                                   .format(error, glpk.glp_get_num_rows(self.prob)))
            self._stats['totalLPs'] += 1

            for i in range(self.code.blocklength):
                solution[i] = glpk.glp_get_col_prim(self.prob, 1 + i)
            self.numConstrs = glpk.glp_get_num_rows(self.prob)
            self._stats['totalConstraints'] += self.numConstrs

            status = glpk.glp_get_status(self.prob)
            # evaluate GLPK status
            if error == glpk.GLP_EOBJUL:
                self.objectiveValue = ub
                self._stats['ubReached'] += 1
                self.foundCodeword = self.mlCertificate = (self.solution in self.code)
                self.status = Decoder.UPPER_BOUND_HIT
                return
            elif status == glpk.GLP_NOFEAS:
                self.objectiveValue = INFINITY
                self.foundCodeword = self.mlCertificate = False
                self._stats['infeasible'] += 1
                self.status = Decoder.INFEASIBLE
                return
            elif status == glpk.GLP_OPT:
                self.objectiveValue = glpk.glp_get_obj_val(self.prob)
            else:
                raise RuntimeError('Unknown GLPK status {}'.format(status))

            if self.objectiveValue > ub - 1e-6:
                self._stats['ubReached'] += 1
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

            integral = True
            for i in range(solution.shape[0]):
                if solution[i] < 1e-6:
                    solution[i] = 0
                elif solution[i] > 1-1e-6:
                    solution[i] = 1
                else:
                    integral = False
                self.fractionality[i] = .5 - fabs(solution[i] - .499999)
            if self.removeInactive != 0 and self.numConstrs >= self.removeInactive:
                self.removeInactiveConstraints()
            self.foundCodeword = self.mlCertificate = True
            numCuts = self.cutSearchAlgorithm(self.hmat)
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
                unitCols = gaussianElimination(self.htilde, xindices, True)
                numCuts = self.cutSearchAlgorithm(self.htilde)
                totalCuts += numCuts
                if numCuts == 0:
                    self.mlCertificate = self.foundCodeword = False
                    break
                rpcrounds += 1

    cdef void removeInactiveConstraints(self):
        """Removes constraints which are not active at the current solution."""
        cdef int i, removed = 0, ind
        cdef double avgSlack, slack
        cdef np.ndarray[dtype=int, ndim=1] indices
        #  compute average slack of constraints all constraints, if only those above the average
        # slack should be removed
        if self.removeAboveAverageSlack:
            avgSlack = 0
            for i in range(self.numConstrs):
                slack = glpk.glp_get_row_ub(self.prob, 1+i) - glpk.glp_get_row_prim(self.prob, 1+i)
                avgSlack += slack
            if self.numConstrs == 0:
                avgSlack = 1e-5
            else:
                avgSlack /= self.numConstrs
        else:
            avgSlack = 1e-5 # some tolerance to avoid removing active constraints
        for i in range(self.numConstrs):
            slack = glpk.glp_get_row_ub(self.prob, 1+i) - glpk.glp_get_row_prim(self.prob, 1+i)
            if slack > avgSlack:
                removed += 1
        if removed > 0:
            indices = np.empty(1 + removed, dtype=np.intc)
            ind = 1
            for i in range(self.numConstrs):
                if glpk.glp_get_row_ub(self.prob, 1+i) \
                        - glpk.glp_get_row_prim(self.prob, 1+i) > avgSlack:
                    indices[ind] = 1+i
                    ind += 1
            glpk.glp_del_rows(self.prob, removed, <int*>indices.data)
            self.numConstrs -= removed
            assert self.numConstrs == glpk.glp_get_num_rows(self.prob)


    
    cdef void removeNonfixedConstraints(self):
        """Remove all but the fixed constraints from the model.

        Usually there are no fixed constraints. In case of all-zero decoding, the zero
        constraints are fixed and not removed by this function.
        """
        cdef np.ndarray[dtype=int, ndim=1] indices
        self.numConstrs = glpk.glp_get_num_rows(self.prob)
        if self.numConstrs > 0:
            indices = np.arange(1 + self.numConstrs, dtype=np.intc)
            glpk.glp_del_rows(self.prob, self.numConstrs, <int*>indices.data)
            glpk.glp_std_basis(self.prob)
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
        if self.objBufLim != 0.0:
            if self.objBufSize != 8:
                params['objBufSize'] = self.objBufSize
            params['objBufLim'] = self.objBufLim
        params['name'] = self.name
        return params
