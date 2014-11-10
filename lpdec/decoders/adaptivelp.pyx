# -*- coding: utf-8 -*-
# distutils: libraries = ["glpk", "m"]
# cython: embedsignature=True
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
import numpy as np
cimport numpy as np
from libc.math cimport fabs, sqrt

from lpdec.decoders cimport glpk

from lpdec.mod2la cimport gaussianElimination
from lpdec.codes cimport BinaryLinearBlockCode
from lpdec.decoders cimport Decoder
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
    :param int insertActive: Determine if and when cuts that are active at the sent or hinted
      codeword are inserted. The value is an OR-combination of:

      * ``0``: never insert active constraints (default)
      * ``1``: insert constraints active at the sent codewords during :func:`setLLRs`,
        it that was given.
      * ``2``: insert constraints active at :attr:`hint`, if it exists, at the beginning of
        :func:`solve`.
    :param double maxActiveAngle: Maximum angle between LLR vector and an active inequality for
      it to be inserted during the "insert active" step.
    :param bool allZero: If set to ``True``, constraints that are active at the all-zero codeword
      are inserted into the model prior to any decoding and stay there all the time.

    .. attribute:: hint

        An optional "hint" codeword that is somehow believed to be near-optimal. For example,
        this might be the output of an iterative decoder. The adaptive LP decoder can use that
        hint to insert active constraints.
    """

    cdef public bint removeAboveAverageSlack, keepCuts, allZero, variableFixing
    cdef int nrFixedConstraints, insertActive
    cdef public double maxActiveAngle, minCutoff
    cdef public int removeInactive, numConstrs, maxRPCrounds
    cdef np.int_t[:,:] hmat, htilde
    cdef glpk.glp_prob *prob
    cdef glpk.glp_smcp parm
    cdef np.double_t[:] diffFromHalf
    cdef np.ndarray setV, Nj
    cdef public np.ndarray hint
    cdef np.int_t[:] fixes
    cdef public object timer, erasureDecoder

    def __init__(self, BinaryLinearBlockCode code,
                 maxRPCrounds=-1,
                 minCutoff=1e-5,
                 removeInactive=0,
                 removeAboveAverageSlack=False,
                 keepCuts=False,
                 insertActive=0,
                 maxActiveAngle=1,
                 allZero=False,
                 variableFixing=False,
                 name=None):
        if name is None:
            name = 'AdaptiveLPDecoder'
        Decoder.__init__(self, code=code, name=name)
        self.maxRPCrounds = maxRPCrounds
        self.minCutoff = minCutoff
        self.removeInactive = removeInactive
        self.removeAboveAverageSlack = removeAboveAverageSlack
        self.keepCuts = keepCuts
        self.insertActive = insertActive
        self.maxActiveAngle=maxActiveAngle
        self.allZero = allZero
        self.variableFixing = variableFixing
        if variableFixing:
            from lpdec.decoders.erasure import ErasureDecoder
            self.erasureDecoder = ErasureDecoder(self.code)
        # initialize GLPK
        self.prob = glpk.glp_create_prob()
        glpk.glp_init_smcp(&self.parm)
        self.parm.msg_lev = glpk.GLP_MSG_OFF
        self.parm.meth = glpk.GLP_DUAL # use dual simplex method
        glpk.glp_add_cols(self.prob, code.blocklength)
        for j in range(code.blocklength):
            glpk.glp_set_col_bnds(self.prob, 1+j, glpk.GLP_DB, 0.0, 1.0)
        glpk.glp_set_obj_dir(self.prob, glpk.GLP_MIN)

        # initialize various structures
        self.hmat = code.parityCheckMatrix
        self.htilde = self.hmat.copy() # the copy is used for gaussian elimination
        self.solution = np.empty(code.blocklength)
        self.diffFromHalf = np.empty(code.blocklength)
        self.setV = np.empty(1+code.blocklength, dtype=np.double)
        self.Nj = np.empty(1+code.blocklength, dtype=np.intc)
        self.fixes = -np.ones(code.blocklength, dtype=np.int)
        self.numConstrs = 0
        self.timer = Timer()
        if allZero:
            self.insertZeroConstraints()
        self.nrFixedConstraints = glpk.glp_get_num_rows(self.prob)

    cdef int cutSearchAlgorithm(self, bint originalHmat):
        """Runs the cut search algorithm and inserts found cuts. If ``originalHmat`` is True,
        the code-defining parity-check matrix is used for searching, otherwise :attr:`htilde`
        which is the result of Gaussian elimination on the most fractional positions of the last
        LP solution.
        :returns: The number of cuts inserted
        """
        cdef:
            np.ndarray[dtype=double, ndim=1] setV = self.setV
            np.ndarray[dtype=int, ndim=1] Nj = self.Nj
            np.double_t[:] solution = self.solution
            np.double_t[:] diffFromHalf = self.diffFromHalf
            np.int_t[:,:] matrix
            int inserted = 0, row, j, ind, setVsize, minDistIndex, Njsize
            double minDistFromHalf, dist, vSum
        if originalHmat:
            matrix = self.hmat
        else:
            matrix = self.htilde

        for row in range(matrix.shape[0]):
            #  for each row, we build the set Nj = { j: matrix[row,j] == 1}
            #  and V = {j ∈ Nj: solution[j] > .5}. The variable setV will be of size Njsize and
            #  have 1 and -1 entries, depending on whether the corresponding index belongs to V or
            #  not.
            #  Indices are shifted by 1 to match GLPK's indexing.
            Njsize = 0
            setVsize = 0
            minDistFromHalf = 1
            minDistIndex = -1
            for j in range(matrix.shape[1]):
                if matrix[row, j] == 1:
                    Nj[1 + Njsize] = j
                    if solution[j] > .5:
                        setV[1 + Njsize] = 1
                        setVsize += 1
                    else:
                        setV[1 + Njsize] = -1
                    if diffFromHalf[j] < minDistFromHalf:
                        minDistFromHalf = diffFromHalf[j]
                        minDistIndex = Njsize
                    elif minDistIndex == -1:
                        minDistIndex = Njsize
                    Njsize += 1
            if Njsize == 0:
                # skip all-zero rows (might occur due to Gaussian elimination)
                continue
            if setVsize % 2 == 0:
                #  V size must be odd, so add entry with minimum distance from .5
                setV[1 + minDistIndex] *= -1
                setVsize += <int>setV[1 + minDistIndex]
            vSum = 0 # left hand side of the induced Feldman inequality
            for ind in range(Njsize):
                if setV[1 + ind] == 1:
                    vSum += 1 - solution[Nj[1 + ind]]
                elif setV[1 + ind] == -1:
                    vSum += solution[Nj[1 + ind]]
            if vSum < 1 - self.minCutoff:
                # inequality violated -> insert
                inserted += 1
                ind = glpk.glp_add_rows(self.prob, 1)
                for j in range(Njsize):
                    Nj[1 + j] += 1 # GLPK indexing: shift indexes by one
                glpk.glp_set_row_bnds(self.prob, ind, glpk.GLP_UP, 0.0, setVsize-1)
                glpk.glp_set_mat_row(self.prob, ind, Njsize, <int*>Nj.data, <double*>setV.data)
            if originalHmat and vSum < 1-1e-5:
                #  in this case, we are in the "original matrix" phase and would have a cut for
                #  insertion which is declined because of minCutoff. This implies that we don't
                #  have a codeword although this method may return 0
                self.foundCodeword = self.mlCertificate = False
        if inserted > 0:
            self._stats['cuts'] += inserted
        return inserted

    cpdef setStats(self, object stats):
        statNames = ["cuts", "totalLPs", "totalConstraints", "ubReached", 'lpTime']
        if self.insertActive != 0:
            statNames.extend(['activeCuts'])
        for item in statNames:
            if item not in stats:
                stats[item] = 0
        Decoder.setStats(self, stats)

    cpdef fix(self, int i, int val):
        glpk.glp_set_col_bnds(self.prob, 1+i, glpk.GLP_FX, val, val)
        self.fixes[i] = val
        if self.variableFixing:
            self.doVariableFixing()

    def doVariableFixing(self):
        for i in range(self.code.blocklength):
            if self.fixes[i] == -1:
                self.erasureDecoder.llrs[i] = 0
            else:
                self.erasureDecoder.llrs[i] = 1 - 2*self.fixes[i]
        self.erasureDecoder.solve()
        for i in range(self.code.blocklength):
            val = self.erasureDecoder.solution[i]
            if val == -1:
                glpk.glp_set_col_bnds(self.prob, 1+i, glpk.GLP_DB, 0.0, 1.0)
            else:
                glpk.glp_set_col_bnds(self.prob, 1+i, glpk.GLP_FX, val, val)

    cpdef release(self, int i):
        glpk.glp_set_col_bnds(self.prob, 1+i, glpk.GLP_DB, 0.0, 1.0)
        self.fixes[i] = -1
        if self.variableFixing:
            self.doVariableFixing()

    def fixed(self, int i):
        """Returns True if and only if the given index is fixed."""
        return self.fixes[i] != -1

    cpdef setLLRs(self, np.double_t[:] llrs, np.int_t[:] sent=None):
        cdef int j
        cdef np.ndarray[dtype=np.int_t, ndim=1] hint
        for j in range(self.code.blocklength):
            glpk.glp_set_obj_coef(self.prob, 1+j, llrs[j])
        Decoder.setLLRs(self, llrs, sent)

        if self.insertActive & 1:
            hint = self.hint if self.hint is not None else np.asarray(sent)
            if hint is None:
                return
            self.removeNonfixedConstraints()
            if self.allZero and np.all(hint) == 0:
                return #  zero-active constraints are already in the model
            self.insertActiveConstraints(hint)


    cpdef solve(self, double lb=-np.inf, double ub=np.inf):
        cdef int i, removed, error, numCuts, rpcrounds = 0, iteration = 0
        cdef np.double_t[:] diffFromHalf = self.diffFromHalf
        cdef np.ndarray[dtype=double, ndim=1] solution = self.solution
        if not self.keepCuts:
            self.removeNonfixedConstraints()
        if self.insertActive & 2 and self.hint is not None:
            self.insertActiveConstraints(self.hint)
        self.foundCodeword = self.mlCertificate = False
        self.objectiveValue = -np.inf
        if self.sent is not None and ub == np.inf:
            # calculate known upper bound on the objective from sent codeword
            ub = np.dot(self.sent, self.llrs) + 2e-6
        while True:
            iteration += 1
            with self.timer:
                error = glpk.glp_simplex(self.prob, &self.parm)
            self._stats['lpTime'] += self.timer.duration
            if error != 0:
                raise RuntimeError("GLPK Simplex Error ({}) {}"
                                   .format(i, glpk.glp_get_num_rows(self.prob)))
            self._stats["totalLPs"] += 1
            self.numConstrs = glpk.glp_get_num_rows(self.prob)
            self._stats["totalConstraints"] += self.numConstrs
            i = glpk.glp_get_status(self.prob)
            if i == glpk.GLP_NOFEAS or i == glpk.GLP_UNBND:
                # during branch-and-bound the problem might become infeasible
                self.objectiveValue = np.inf
                self.foundCodeword = self.mlCertificate = False
                break
            elif i != glpk.GLP_OPT:
                raise RuntimeError("GLPK error {}".format(i))
            newObjectiveValue = glpk.glp_get_obj_val(self.prob)
            if newObjectiveValue <= self.objectiveValue:
                # prevent infinite loops in some rare cases where numerical issues cause
                # non-increasing objective value after cut generation
                print('cga: no improvement')
                break
            self.objectiveValue = newObjectiveValue
            self.objectiveValue = glpk.glp_get_obj_val(self.prob)
            if self.objectiveValue >= ub - 1e-6:
                # lower bound from the LP is above known upper bound -> no need to proceed
                self.objectiveValue = np.inf
                self._stats["ubReached"] += 1
                self.foundCodeword = self.mlCertificate = False
                break
            integral = True
            # read solution from GLPK. Round values to {0,1} that are very close
            for i in range(self.code.blocklength):
                solution[i] = glpk.glp_get_col_prim(self.prob, 1+i)
            for i in range(self.code.blocklength):
                if solution[i] < 1e-6:
                    solution[i] = 0
                elif solution[i] > 1-1e-6:
                    solution[i] = 1
                else:
                    integral = False
                diffFromHalf[i] = fabs(solution[i]-.499999)
            if self.removeInactive != 0 \
                    and self.numConstrs - self.nrFixedConstraints >= self.removeInactive:
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
                xindices = np.argsort(diffFromHalf)
                gaussianElimination(self.htilde, xindices, True)
                if not self.cutSearchAlgorithm(False):
                    self.mlCertificate = self.foundCodeword = False
                    break
                rpcrounds += 1
        self.hint = None
 
    cdef void removeInactiveConstraints(self):
        """Removes constraints which are not active at the current solution."""
        cdef int i, removed = 0, ind
        cdef double avgSlack, slack
        cdef np.ndarray[dtype=int, ndim=1] indices
        #  compute average slack of constraints all constraints, if only those above the average
        # slack should be removed
        if self.removeAboveAverageSlack:
            avgSlack = 0
            for i in range(self.nrFixedConstraints, self.numConstrs):
                slack = glpk.glp_get_row_ub(self.prob, 1+i) - glpk.glp_get_row_prim(self.prob, 1+i)
                avgSlack += slack
            if self.numConstrs == self.nrFixedConstraints:
                avgSlack = 1e-5
            else:
                avgSlack /= (self.numConstrs - self.nrFixedConstraints)
        else:
            avgSlack = 1e-5 # some tolerance to avoid removing active constraints
        for i in range(self.nrFixedConstraints, self.numConstrs):
            slack = glpk.glp_get_row_ub(self.prob, 1+i) - glpk.glp_get_row_prim(self.prob, 1+i)
            if slack > avgSlack:
                removed += 1
        if removed > 0:
            indices = np.empty(1+removed, dtype=np.intc)
            ind = 1
            for i in range(self.nrFixedConstraints, self.numConstrs):
                if glpk.glp_get_row_ub(self.prob, 1+i) \
                        - glpk.glp_get_row_prim(self.prob, 1+i) > avgSlack:
                    indices[ind] = 1+i
                    ind += 1
            glpk.glp_del_rows(self.prob, removed, <int*>indices.data)
            self.numConstrs -= removed
            assert self.numConstrs == glpk.glp_get_num_rows(self.prob)

    cdef void insertActiveConstraints(self, np.int_t[:] codeword):
        """Inserts constraints that are active at the given codeword."""
        cdef np.ndarray[ndim=1, dtype=double] coeff = self.setV, llrs = self.llrs
        cdef np.ndarray[ndim=1, dtype=int] Nj = self.Nj
        cdef int ind, i, j, absG, Njsize, rowIndex
        cdef np.int_t[:,:] hmat = self.hmat
        cdef double lambdaSum, normDenom, absLambda = np.linalg.norm(llrs)
        for i in range(hmat.shape[0]):
            Njsize = 0
            absG = 0
            lambdaSum = 0
            for j in range(hmat.shape[1]):
                if hmat[i,j] == 1:
                    Nj[1+Njsize] = 1+j
                    if codeword[j] == 1:
                        coeff[1+Njsize] = 1
                        absG += 1
                        lambdaSum += llrs[Njsize]
                    else:
                        coeff[1+Njsize] = -1
                        lambdaSum -= llrs[Njsize]
                    Njsize += 1
            normDenom = absLambda * Njsize
            for ind in range(Njsize):
                if coeff[1+ind] == 1:
                    if (lambdaSum-2*llrs[Nj[ind+1]-1]) / normDenom  < self.maxActiveAngle:
                        coeff[1+ind] = -1
                        rowIndex = glpk.glp_add_rows(self.prob, 1)
                        glpk.glp_set_row_bnds(self.prob, rowIndex, glpk.GLP_UP, 0.0, absG-2)
                        glpk.glp_set_mat_row(self.prob, rowIndex, Njsize,
                                             <int*>Nj.data, <double*>coeff.data)
                        coeff[1+ind] = 1
                        self._stats["activeCuts"] += 1
                        self.numConstrs += 1
                else:
                    if (lambdaSum+2*llrs[Nj[ind+1]-1]) / normDenom < self.maxActiveAngle:
                        coeff[1+ind] = 1
                        rowIndex = glpk.glp_add_rows(self.prob, 1)
                        glpk.glp_set_row_bnds(self.prob, rowIndex, glpk.GLP_UP, 0.0, absG)
                        glpk.glp_set_mat_row(self.prob, rowIndex, Njsize,
                                             <int*>Nj.data, <double*>coeff.data)
                        coeff[1+ind] = -1
                        self._stats["activeCuts"] += 1
                        self.numConstrs += 1

    cdef void insertZeroConstraints(self):
        """Inserts constraints that are active at the zero codeword. This can be used in case of
        all-zero decoding to avoid frequent adaptive insertion of the same constraints.
        """
        cdef np.ndarray[ndim=1, dtype=double] coeff = self.setV
        cdef np.ndarray[ndim=1, dtype=int] Nj = self.Nj
        cdef int ind, i, j, Njsize, rowIndex
        cdef np.int_t[:,:] hmat = self.hmat
        for i in range(hmat.shape[0]):
            Njsize = 0
            for j in range(hmat.shape[1]):
                if hmat[i,j] == 1:
                    Nj[1+Njsize] = 1+j
                    coeff[1+Njsize] = -1
                    Njsize += 1            
            for ind in range(Njsize):
                coeff[1+ind] = 1
                rowIndex = glpk.glp_add_rows(self.prob, 1)
                glpk.glp_set_row_bnds(self.prob, rowIndex, glpk.GLP_UP, 0.0, 0.0)
                glpk.glp_set_mat_row(self.prob, rowIndex, Njsize, <int*>Nj.data, <double*>coeff.data)
                coeff[1+ind] = -1
                self.numConstrs += 1
    
    cdef void removeNonfixedConstraints(self):
        """Remove all but the fixed constraints from the model.

        Usually there are no fixed constraints. In case of all-zero decoding, the zero
        constraints are fixed and not removed by this function.
        """
        cdef np.ndarray[dtype=int, ndim=1] indices
        self.numConstrs = glpk.glp_get_num_rows(self.prob)
        if self.numConstrs > self.nrFixedConstraints:
            indices = np.arange(self.nrFixedConstraints, 1+self.numConstrs, dtype=np.intc)
            glpk.glp_del_rows(self.prob, self.numConstrs - self.nrFixedConstraints,
                              <int*>indices.data)
            glpk.glp_std_basis(self.prob)
            self.numConstrs = self.nrFixedConstraints
                   
    cpdef params(self):
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
        if self.insertActive != 0:
            params['insertActive'] = self.insertActive
        if self.maxActiveAngle != 1:
            params['maxActiveAngle'] = self.maxActiveAngle
        if self.allZero:
            params['allZero'] = True
        params['name'] = self.name
        return params
