# -*- coding: utf-8 -*-
# Copyright 2015 Michael Helmling
# cython: boundscheck=False, nonecheck=False, initializedcheck=False, wraparound=False, cdivision=True
# cython: language_level=3
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation

from __future__ import division, print_function
from collections import OrderedDict
import numpy as np
cimport numpy as np
from numpy.math cimport INFINITY

from lpdec.decoders.base cimport Decoder
from lpdec.decoders.base import Decoder
from lpdec.codes.polar import PolarCode


class PolarSCDecoder(Decoder):
    """Implements a successive cancellation decoder for polar codes."""
    def __init__(self, code, name=None):
        if name is None:
            name = 'PolarSCDecoder'
        assert isinstance(code, PolarCode)
        Decoder.__init__(self, code=code, name=name)
        m = code.n # ok this isn't great, but it's called m in the decoding paper ...
        self.P = np.empty((m+1, 2**m, 2), dtype=np.double)
        self.C = np.empty((m+1, 2**m, 2), dtype=np.int)
        self.solution = np.zeros(code.blocklength)

    def solve(self, lb=-INFINITY, ub=INFINITY):
        n = self.code.blocklength
        m = self.code.n
        P = self.P
        C = self.C
        for beta in range(n):
            P[0,beta,1] = 1./(np.exp(self.llrs[beta])+1)
            P[0,beta,0] = 1 - P[0,beta,1]

        def recursivelyCalcP(lam, phi):
            if lam == 0:
                return
            psi = phi // 2
            sigma = 0
            if phi % 2 == 0:
                recursivelyCalcP(lam - 1, psi)
            for beta in range(2**(m-lam)):
                if phi % 2 == 0:
                    for u in (0,1):
                        P[lam,beta,u] = sum(.5*P[lam-1,2*beta,u^upp]*P[lam-1,2*beta+1,upp]
                                            for upp in (0,1))
                        sigma = max(sigma, P[lam, beta, u])
                else:
                    u = C[lam,beta,0]
                    for upp in (0,1):
                        P[lam,beta,upp] = .5*P[lam-1,2*beta,u^upp]*P[lam-1,2*beta+1,upp]
                        sigma = max(sigma, P[lam, beta, upp])
            P[lam, :2**(m-lam), :] /= sigma

        def recursivelyUpdateC(lam, phi):
            psi = phi // 2
            for beta in range(2**(m-lam)):
                C[lam-1,2*beta,psi % 2] = C[lam,beta,0] ^ C[lam, beta, 1]
                C[lam-1,2*beta+1, psi % 2] = C[lam,beta,1]
            if psi % 2 == 1:
                recursivelyUpdateC(lam-1, psi)
        for phi in range(n):
            recursivelyCalcP(m, phi)
            if phi in self.code.frozen:
                C[m,0,phi % 2] = 0
            elif P[m,0,0] > P[m,0,1]:
                C[m,0,phi % 2] = 0
            else:
                C[m,0,phi % 2] = 1
            if phi % 2 == 1:
                recursivelyUpdateC(m, phi)
        for beta in range(n):
            self.solution[beta] = C[0,beta,0]
        self.objectiveValue = np.dot(self.solution, self.llrs)

    def params(self):
        return OrderedDict(name=self.name)


cdef class PolarSCListDecoder(Decoder):
    """Implements a successive cancellation list decoder for polar codes."""

    cdef:
        public int L
        int m
        public list inactivePathIndices, inactiveArrayIndices
        int[::1] activePath, fixes
        np.intp_t[:, ::1] pathIndexToArrayIndex
        int[:, ::1] arrayReferenceCount
        double[:, :, :, ::1] P
        int[:, :, :, ::1] C
        np.ndarray probForks, contForks
        public bint excludeZero

    def __init__(self, code, L=4, name=None):
        if name is None:
            name = 'PolarSCListDecoder(L={})'.format(L)
        Decoder.__init__(self, code=code, name=name)
        assert isinstance(code, PolarCode)
        self.m = m = code.n # ok this isn't great, but it's called m in the decoding paper ...
        self.L = L
        self.activePath = np.empty(L, dtype=np.intc)
        self.P = np.empty((m+1, L, 2**m, 2), dtype=np.double)
        self.C = np.empty((m+1, L, 2**m, 2), dtype=np.intc)
        self.pathIndexToArrayIndex = np.empty((m+1, L), dtype=np.intp)
        self.arrayReferenceCount = np.zeros((m+1, L), dtype=np.intc)
        self.probForks = np.empty((L, 2), np.double)
        self.contForks = np.empty((L, 2), np.intc)
        self.fixes = -np.ones(code.blocklength, dtype=np.intc)
        self.inactivePathIndices = self.inactiveArrayIndices = None
        self.excludeZero = False


    cdef int clonePath(self, int l):
        cdef int lprime, s, lam
        lprime = self.inactivePathIndices.pop()
        self.activePath[lprime] = True
        for lam in range(self.m + 1):
            s = self.pathIndexToArrayIndex[lam, l]
            self.pathIndexToArrayIndex[lam, lprime] = s
            self.arrayReferenceCount[lam, s] += 1
        return lprime

    cdef void killPath(self, int l):
        cdef int lam, s
        self.activePath[l] = False
        self.inactivePathIndices.append(l)
        for lam in range(self.m + 1):
            s = self.pathIndexToArrayIndex[lam, l]
            self.arrayReferenceCount[lam, s] -= 1
            if self.arrayReferenceCount[lam, s] == 0:
                self.inactiveArrayIndices[lam].append(s)

    cdef int getArrayPointer(self, int lam, int l):
        cdef int s = self.pathIndexToArrayIndex[lam, l]
        cdef int sprime
        if self.arrayReferenceCount[lam, s] == 1:
            return s
        # else copy data
        sprime = self.inactiveArrayIndices[lam].pop()
        self.P[lam, sprime, :2**(self.m-lam), :] = self.P[lam, s, :2**(self.m-lam), :]
        self.C[lam, sprime, :2**(self.m-lam), :] = self.C[lam, s, :2**(self.m-lam), :]
        self.arrayReferenceCount[lam, s] -= 1
        self.arrayReferenceCount[lam, sprime] = 1
        self.pathIndexToArrayIndex[lam, l] = sprime
        return sprime

    cdef void recursivelyCalcP(self, int lam, int phi):
        #cdef int psi = phi // 2
        cdef int beta, u, pLam, pLam_, upp, l
        cdef double sigma = 0
        if lam == 0:
            return
        if phi % 2 == 0:
            self.recursivelyCalcP(lam - 1, phi // 2)

        for l in range(self.L):
            if not self.activePath[l]:
                continue
            pLam = self.getArrayPointer(lam, l)
            pLam_ = self.getArrayPointer(lam - 1, l)
            for beta in range(2**(self.m-lam)):
                if phi % 2 == 0:
                    for u in range(2):
                        self.P[lam, pLam, beta, u] = .5 * (
                            self.P[lam-1, pLam_, 2*beta, u] * self.P[lam-1, pLam_, 2*beta+1, 0]
                          + self.P[lam-1, pLam_, 2*beta, u^1] * self.P[lam-1, pLam_, 2*beta+1, 1])
                        sigma = max(sigma, self.P[lam, pLam, beta, u])
                else:
                    u = self.C[lam, pLam, beta, 0]
                    for upp in range(2):
                        self.P[lam, pLam, beta, upp] = .5 * self.P[lam-1, pLam_, 2*beta, u ^ upp]\
                                                     * self.P[lam-1, pLam_, 2*beta+1, upp]
                        sigma = max(sigma, self.P[lam, pLam, beta, upp])
        for l in range(self.L):
            if self.activePath[l]:
                pLam = self.getArrayPointer(lam, l)
                for beta in range(2**(self.m-lam)):
                    for u in range(2):
                        self.P[lam, pLam, beta, u] /= sigma

    cdef void recursivelyUpdateC(self, int lam, int phi):
        cdef int psi = phi // 2
        cdef int l, beta
        for l in range(self.L):
            if not self.activePath[l]:
                continue
            pLam  = self.getArrayPointer(lam, l)
            pLam_ = self.getArrayPointer(lam-1, l)
            for beta in range(2**(self.m-lam)):
                self.C[lam-1, pLam_, 2*beta, psi % 2] = self.C[lam, pLam, beta, 0] ^ self.C[lam, pLam, beta, 1]
                self.C[lam-1, pLam_, 2*beta+1, psi % 2] = self.C[lam, pLam, beta, 1]
        if psi % 2 == 1:
            self.recursivelyUpdateC(lam-1, psi)

    cdef int continuePaths(self, int phi) except -1:
        """Implementation of continuePaths_UnfrozenBit (Algorithm 18)"""
        cdef int l, lp, Pm, Cm, rho, i = 0
        cdef np.ndarray[ndim=2, dtype=double] probForks = self.probForks
        cdef np.ndarray[ndim=2, dtype=bint] contForks = self.contForks
        cdef np.intp_t[:] strd

        for l in range(self.L):
            if not self.activePath[l]:
                probForks[l, 0] = probForks[l, 1] = -1
            else:
                Pm = self.getArrayPointer(self.m, l)
                probForks[l, 0] = self.P[self.m, Pm, 0, 0]
                probForks[l, 1] = self.P[self.m, Pm, 0, 1]
                i += 1
        rho = min(2 * i, self.L)
        # implements line 14
        srtd = np.argsort(probForks, None)
        contForks[:, :] = 0
        contForks.flat[srtd[srtd.size-rho:]] = 1
        for l in range(self.L):
            if not self.activePath[l]:
                continue
            if (not contForks[l, 0]) and (not contForks[l, 1]):
                self.killPath(l)
        for l in range(self.L):
            if not self.activePath[l]:
                continue
            if  contForks[l, 0] == 0 and contForks[l, 1] == 0:
                continue
            Cm = self.getArrayPointer(self.m, l)
            if contForks[l, 0] == 1 and contForks[l, 1] == 1:
                self.C[self.m, Cm, 0, phi % 2] = 0
                lp = self.clonePath(l)
                Cmp = self.getArrayPointer(self.m, lp)
                self.C[self.m, Cmp, 0, phi % 2] = 1
            elif contForks[l, 0] == 1:
                self.C[self.m, Cm, 0, phi % 2] = 0
            else:
                self.C[self.m, Cm, 0, phi % 2] = 1

    cpdef solve(self, double lb=-INFINITY, double ub=INFINITY):
        cdef:
            int n = self.code.blocklength
            int phi, lp, l, lam, s
            double pp
            double[:,:,:,::1] P = self.P
            int[:,:,:,::1] C = self.C
            np.intp_t[:, ::1] pathIndexToArrayIndex = self.pathIndexToArrayIndex
            int[::1] activePath = self.activePath

        # initialize data structures
        self.arrayReferenceCount[:, :] = 0
        self.activePath[:] = 0
        self.inactivePathIndices = list(range(self.L))
        self.inactiveArrayIndices = [list(range(self.L)) for _ in range(self.m+1)]

        # assign initial path
        l = self.inactivePathIndices.pop()
        self.activePath[l] = True
        for lam in range(self.m + 1):
            s = self.inactiveArrayIndices[lam].pop()
            pathIndexToArrayIndex[lam, l] = s
            self.arrayReferenceCount[lam, s] = 1

        p0 = self.getArrayPointer(0, l)
        for beta in range(n):
            if self.fixes[beta] == -1:
                P[0, p0, beta, 1] = 1 / (np.exp(self.llrs[beta])+1)
                P[0, p0, beta, 0] = 1 - P[0, p0, beta, 1]
            else:
                P[0, p0, beta, 0] = 1 - self.fixes[beta]
                P[0, p0, beta, 1] = self.fixes[beta]
        # main loop
        for phi in range(n):
            self.recursivelyCalcP(self.m, phi)
            if phi in self.code.frozen:
                # continuePaths_FrozenBit (inlined)
                for l in range(self.L):
                    if self.activePath[l]:
                        Cm = self.getArrayPointer(self.m, l)
                        C[self.m, Cm, 0, phi % 2] = 0
            else:
                self.continuePaths(phi)
            if phi % 2 == 1:
                self.recursivelyUpdateC(self.m, phi)
        # find most probable path
        lp = -1
        pp = 0
        for l in range(self.L):
            if not self.activePath[l]:
                continue
            point = self.getArrayPointer(self.m, l)
            if pp < P[self.m, point, 0, C[self.m, point, 0, 1]]:
                if self.excludeZero:
                    C0 = self.getArrayPointer(0, l)
                    if np.all(np.equal(C[0, C0, :, 0], 0)):
                        continue
                lp = l
                pp = P[self.m, point, 0, C[self.m, point, 0, 1]]
        if lp == -1:
            self.objectiveValue = INFINITY
            self.status = Decoder.INFEASIBLE
            self.foundCodeword = False
            self.solution[:] = .5
            return
        self.status = Decoder.OPTIMAL
        C0 = self.getArrayPointer(0, lp)
        self.objectiveValue = 0
        for beta in range(n):
            self.solution[beta] = C[0, C0, beta, 0]
            if self.solution[beta]:
                self.objectiveValue += self.llrs[beta]
        if self.excludeZero:
            assert np.sum(self.solution) > 0
            assert self.objectiveValue > 0
        self.foundCodeword = self.solution in self.code

    cpdef release(self, int index):
        self.fixes[index] = -1

    cpdef fix(self, int index, int val):
        self.fixes[index] = val

    def params(self):
        return OrderedDict([('L', self.L), ('name', self.name)])