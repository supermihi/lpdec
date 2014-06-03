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
    :param float minCutoff: Minimum cutoff for a generated cut to be inserted into the LP model.
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
    :param bool excludeZero: Forbid decoding to the all-zero codeword by adding the constraint
      :math:`\sum_{i=1}^n x_i \geq 1`. Used in minimum distance computation.

    .. attribute:: hint

        An optional "hint" codeword that is somehow believed to be near-optimal. For example,
        this might be the output of an iterative decoder. The adaptive LP decoder can use that
        hint to insert active constraints.
    """

    cdef public bint removeAboveAverageSlack, keepCuts, allZero, excludeZero
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
    cdef public object timer

    def __init__(self, BinaryLinearBlockCode code,
                 maxRPCrounds=-1,
                 minCutoff=1e-5,
                 removeInactive=0,
                 removeAboveAverageSlack=False,
                 keepCuts=False,
                 insertActive=0,
                 maxActiveAngle=1,
                 allZero=False,
                 excludeZero=False,
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
        self.excludeZero = excludeZero

        self.hmat = code.parityCheckMatrix
        self.htilde = self.hmat.copy()
        self.prob = glpk.glp_create_prob()
        glpk.glp_init_smcp(&self.parm)
        self.parm.msg_lev = glpk.GLP_MSG_OFF
        self.parm.meth = glpk.GLP_DUAL
        glpk.glp_add_cols(self.prob, code.blocklength)
        for j in range(code.blocklength):
            glpk.glp_set_col_bnds(self.prob, 1+j, glpk.GLP_DB, 0.0, 1.0)
        glpk.glp_set_obj_dir(self.prob, glpk.GLP_MIN)
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
        if self.excludeZero:
            # add the constraint \sum x >= 1
            rowIndex = glpk.glp_add_rows(self.prob, 1)
            glpk.glp_set_row_bnds(self.prob, rowIndex, glpk.GLP_LO, 1.0, 0.0)
            self.setV = np.ones(code.blocklength+1, dtype=np.double)
            self.Nj = np.arange(code.blocklength+1, dtype=np.intc)
            glpk.glp_set_mat_row(self.prob, rowIndex, code.blocklength, <int*>self.Nj.data, <double*>self.setV.data)
            self.nrFixedConstraints += 1

    
    def fixed(self, int i):
        return self.fixes[i] != -1
    
    def numFixedOnes(self):
        num = 0
        for i in range(self.code.infolength):
            if self.fixes[i] != -1 and self.fixes[i] % 2 == 1:
                num += 1
        return num

    cdef int cutSearchAlgorithm(self, bint originalHmat):
        cdef np.ndarray[dtype=double, ndim=1] setV = self.setV
        cdef np.ndarray[dtype=int, ndim=1] Nj = self.Nj
        cdef np.double_t[:] solution = self.solution
        cdef np.double_t[:] diffFromHalf = self.diffFromHalf
        cdef np.int_t[:,:] mat
        cdef int found = 0
        cdef int i, j, ind, setVsize, minDistIndex, Njsize
        cdef double minDistFromHalf, dist, vSum
        if originalHmat:
            mat = self.hmat
        else:
            mat = self.htilde

        for i in range(mat.shape[0]):
            Njsize = 0
            setVsize = 0
            minDistFromHalf = 1
            minDistIndex = -1
            for j in range(mat.shape[1]):
                if mat[i,j] == 1:
                    Nj[1+Njsize] = j
                    if solution[j] > .5:
                        setV[1+Njsize] = 1
                        setVsize += 1
                    else:
                        setV[1+Njsize] = -1
                    if diffFromHalf[j] < minDistFromHalf:
                        minDistFromHalf = diffFromHalf[j]
                        minDistIndex = Njsize
                    elif minDistIndex == -1:
                        minDistIndex = Njsize
                    Njsize += 1
            if Njsize == 0: # all-zero row
                continue
            #print('row {}: {}'.format(i, [index for index in Nj[1:Njsize+1]]))
            if setVsize % 2 == 0:
                setV[1+minDistIndex] *= -1
                setVsize += <int>setV[1+minDistIndex]
            vSum = 0
            for ind in range(Njsize):
                if setV[1+ind] == 1:
                    vSum += 1 - solution[Nj[1+ind]]
                elif setV[1+ind] == -1:
                    vSum += solution[Nj[1+ind]]
            if vSum < 1-self.minCutoff:
                found += 1
                ind = glpk.glp_add_rows(self.prob, 1)
                for j in range(Njsize):
                    Nj[1+j] += 1 # GLPK indexing: shift indexes by one
                glpk.glp_set_row_bnds(self.prob, ind, glpk.GLP_UP, 0.0, setVsize-1)
                #print('add {} {}: {}'.format(i, Njsize, [(index-1, value) for index, value in zip(Nj[1:Njsize+1], setV[1:Njsize+1])]))
                glpk.glp_set_mat_row(self.prob, ind, Njsize, <int*>Nj.data, <double*>setV.data)
            if originalHmat and vSum < 1-1e-5:
                self.foundCodeword = self.mlCertificate = False
        if found > 0:
            self._stats["cuts"] += found
        return found


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

    cpdef release(self, int i):
        glpk.glp_set_col_bnds(self.prob, 1+i, glpk.GLP_DB, 0.0, 1.0)
        self.fixes[i] = -1
    
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
                #  zero-active constraints are already in the model
                return
            logger.debug('insert active for {}'.format(hint))
            self.insertActiveConstraints(hint)


    cpdef solve(self, double lb=-np.inf, double ub=np.inf):
        cdef int i, removed, error, rpcrounds = 0, iteration = 0
        cdef np.double_t[:] diffFromHalf = self.diffFromHalf
        cdef np.ndarray[dtype=double, ndim=1] solution = self.solution
        if not self.keepCuts:
            self.removeNonfixedConstraints()
        if self.insertActive & 2 and self.hint is not None:
            self.insertActiveConstraints(self.hint)
        self.foundCodeword = self.mlCertificate = False
        self.objectiveValue = -np.inf
        while True:
            iteration += 1
            with self.timer:
                i = glpk.glp_simplex(self.prob, &self.parm)
            self._stats['lpTime'] += self.timer.duration
            if i != 0:
                raise RuntimeError("GLPK Simplex Error ({}) {}".format(i, glpk.glp_get_num_rows(self.prob)))
            self._stats["totalLPs"] += 1
            self.numConstrs = glpk.glp_get_num_rows(self.prob)
            self._stats["totalConstraints"] += self.numConstrs
            i = glpk.glp_get_status(self.prob)
            if i == glpk.GLP_NOFEAS or i == glpk.GLP_UNBND:
                self.objectiveValue = np.inf
                self.foundCodeword = self.mlCertificate = False
                break
            elif i != glpk.GLP_OPT:
                raise RuntimeError("GLPK error {}".format(i))
            self.objectiveValue = glpk.glp_get_obj_val(self.prob)
            #logger.debug('solved to {} with {} constraints'.format(self.objectiveValue,
            # self.numConstrs))
            if self.objectiveValue >= ub - 1e-6:
                logger.debug('reached ub: {}'.format(self.objectiveValue))
                self.objectiveValue = np.inf
                self._stats["ubReached"] += 1
                self.foundCodeword = self.mlCertificate = False
                break
            integral = True
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
            if self.removeInactive != 0 and self.numConstrs - self.nrFixedConstraints >= self.removeInactive:
                self.removeInactiveConstraints()
            self.foundCodeword = self.mlCertificate = True
            numCuts = self.cutSearchAlgorithm(True)
            if numCuts > 0:
                continue
            elif integral:
                break
            elif rpcrounds >= self.maxRPCrounds and self.maxRPCrounds != -1:
                self.foundCodeword = self.mlCertificate = False
                break
            else:
                xindices = np.argsort(diffFromHalf)
                gaussianElimination(self.htilde, xindices, True)
                if not self.cutSearchAlgorithm(False):
                    self.mlCertificate = self.foundCodeword = False
                    break
                rpcrounds += 1
        self.hint = None
 
 
    cdef void removeInactiveConstraints(self):
        cdef int i, removed = 0, ind
        cdef double avgSlack, slack
        cdef np.ndarray[dtype=int, ndim=1] indices
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
            avgSlack = 1e-5
        for i in range(self.nrFixedConstraints, self.numConstrs):
            slack = glpk.glp_get_row_ub(self.prob, 1+i) - glpk.glp_get_row_prim(self.prob, 1+i)
            if slack > avgSlack:
                removed += 1
        if removed > 0:
            indices = np.empty(1+removed, dtype=np.intc)
            ind = 1
            for i in range(self.nrFixedConstraints, self.numConstrs):
                if glpk.glp_get_row_ub(self.prob, 1+i) - glpk.glp_get_row_prim(self.prob, 1+i) > avgSlack:
                    indices[ind] = 1+i
                    ind += 1
            glpk.glp_del_rows(self.prob, removed, <int*>indices.data)
            self.numConstrs -= removed
            assert self.numConstrs == glpk.glp_get_num_rows(self.prob)
        
           
    cdef void insertActiveConstraints(self, np.int_t[:] codeword):
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
                        #print('init1 {}: {}'.format(Njsize, [(index, value) for index, value in zip(Nj[1:Njsize+1], coeff[1:Njsize+1])]))
                        glpk.glp_set_row_bnds(self.prob, rowIndex, glpk.GLP_UP, 0.0, absG-2)
                        glpk.glp_set_mat_row(self.prob, rowIndex, Njsize, <int*>Nj.data, <double*>coeff.data)
                        coeff[1+ind] = 1
                        self._stats["activeCuts"] += 1
                        self.numConstrs += 1
                else:
                    if (lambdaSum+2*llrs[Nj[ind+1]-1]) / normDenom < self.maxActiveAngle:
                        coeff[1+ind] = 1
                        rowIndex = glpk.glp_add_rows(self.prob, 1)
                        #print('init2  {} {}: {}'.format(i, Njsize, [(index, value) for index, value in zip(Nj[1:Njsize+1], coeff[1:Njsize+1])]))
                        glpk.glp_set_row_bnds(self.prob, rowIndex, glpk.GLP_UP, 0.0, absG)
                        glpk.glp_set_mat_row(self.prob, rowIndex, Njsize, <int*>Nj.data, <double*>coeff.data)
                        coeff[1+ind] = -1
                        self._stats["activeCuts"] += 1
                        self.numConstrs += 1

    cdef void insertZeroConstraints(self):
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
                #print('init2  {} {}: {}'.format(i, Njsize, [(index, value) for index, value in zip(Nj[1:Njsize+1], coeff[1:Njsize+1])]))
                glpk.glp_set_row_bnds(self.prob, rowIndex, glpk.GLP_UP, 0.0, 0.0)
                glpk.glp_set_mat_row(self.prob, rowIndex, Njsize, <int*>Nj.data, <double*>coeff.data)
                coeff[1+ind] = -1
                self.numConstrs += 1
    
    cdef void removeNonfixedConstraints(self):
        cdef np.ndarray[dtype=int, ndim=1] indices
        self.numConstrs = glpk.glp_get_num_rows(self.prob)
        if self.numConstrs > self.nrFixedConstraints:
            indices = np.arange(self.nrFixedConstraints, 1+self.numConstrs, dtype=np.intc)
            glpk.glp_del_rows(self.prob, self.numConstrs-self.nrFixedConstraints, <int*>indices.data)
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
        if self.excludeZero:
            params['excludeZero'] = True
        params['name'] = self.name
        return params
