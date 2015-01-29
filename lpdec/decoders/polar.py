# -*- coding: utf-8 -*-
# Copyright 2015 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation

from __future__ import division, print_function
from collections import OrderedDict
import numpy as np

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
        self.P = np.empty((m+1, 2**m, 2))
        self.C = np.empty((m+1, 2**m, 2), dtype=np.int)
        self.solution = np.zeros(code.blocklength)

    def solve(self, lb=-np.inf, ub=np.inf):
        n = self.code.blocklength
        m = self.code.n
        P = self.P
        C = self.C
        for beta in range(n):
            P[0,beta,1] = 1/(np.exp(self.llrs[beta])+1)
            P[0,beta,0] = 1 - P[0,beta,1]

        def recursivelyCalcP(lam, phi):
            if lam == 0:
                return
            psi = phi // 2
            if phi % 2 == 0:
                recursivelyCalcP(lam - 1, psi)
            for beta in range(2**(m-lam)):
                if phi % 2 == 0:
                    for u in (0,1):
                        P[lam,beta,u] = sum(.5*P[lam-1,2*beta,u^upp]*P[lam-1,2*beta+1,upp]
                                            for upp in (0,1))
                else:
                    u = C[lam,beta,0]
                    assert u in (0,1)
                    for upp in (0,1):
                        P[lam,beta,upp] = .5*P[lam-1,2*beta,u^upp]*P[lam-1,2*beta+1,upp]

        def recursivelyUpdateC(lam, phi):
            assert phi % 2 == 1
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


class PolarSCListDecoder(Decoder):
    """Implements a successive cancellation list decoder for polar codes."""
    def __init__(self, code, L=4, name=None):
        if name is None:
            name = 'PolarSCListDecoder'
        assert isinstance(code, PolarCode)
        Decoder.__init__(self, code=code, name=name)
        m = code.n # ok this isn't great, but it's called m in the decoding paper ...
        self.L = L
        self.activePath = np.zeros(L, dtype=np.bool)
        self.P = np.empty((m+1, L, 2**m, 2), dtype=np.double)
        self.C = np.empty((m+1, L, 2**m, 2), dtype=np.bool)
        self.pathIndexToArrayIndex = np.empty((m+1, L), dtype=np.intp)
        self.arrayReferenceCount = np.empty((m+1, L), dtype=np.int)
        self.solution = np.empty(code.blocklength)

    def solve(self, lb=-np.inf, ub=np.inf):
        n = self.code.blocklength
        m = self.code.n
        P = self.P
        C = self.C
        L = self.L
        inactivePathIndices = list(range(L))
        inactiveArrayIndices = [list(range(L)) for _ in range(m+1)]
        self.activePath[:] = 0
        self.arrayReferenceCount[:, :] = 0
        self.pathIndexToArrayIndex[:, :] = -1
        P[:, :] = -1
        C[:, :] = -1


        def assignInitialPath():
            l = inactivePathIndices.pop()
            self.activePath[l] = True
            for lam in range(m+1):
                s = inactiveArrayIndices[lam].pop()
                assert isinstance(s, int), str(type(s))
                self.pathIndexToArrayIndex[lam, l] = s
                self.arrayReferenceCount[lam, s] = 1
            return l

        def clonePath(l):
            assert self.activePath[l]
            lprime = inactivePathIndices.pop()
            self.activePath[lprime] = True
            for lam in range(m+1):
                s = self.pathIndexToArrayIndex[lam, l]
                assert s != -1
                self.pathIndexToArrayIndex[lam, lprime] = s
                self.arrayReferenceCount[lam, s] += 1
            return l

        def killPath(l):
            assert self.activePath[l]
            self.activePath[l] = False
            inactivePathIndices.append(l)
            for lam in range(m+1):
                s = self.pathIndexToArrayIndex[lam, l]
                assert s != -1
                assert self.arrayReferenceCount[lam, s] >= 1
                self.arrayReferenceCount[lam, s] -= 1
                if self.arrayReferenceCount[lam, s] == 0:
                    inactiveArrayIndices[lam].append(s)

        def getArrayPointer(lam, l):
            assert self.activePath[l]
            s = self.pathIndexToArrayIndex[lam, l]
            assert self.arrayReferenceCount[lam, s] > 0
            if self.arrayReferenceCount[lam, s] == 1:
                return s
            # else copy data
            sprime = inactiveArrayIndices[lam].pop()
            self.P[lam, sprime, :2**(m-lam), :] = self.P[lam, s, :2**(m-lam), :]
            self.C[lam, sprime, :2**(m-lam), :] = self.C[lam, s, :2**(m-lam), :]
            self.arrayReferenceCount[lam, s] -= 1
            assert self.arrayReferenceCount[lam, sprime] == 0
            self.arrayReferenceCount[lam, sprime] = 1
            self.pathIndexToArrayIndex[lam, l] = sprime
            return sprime

        def recursivelyCalcP(lam, phi):
            if lam == 0:
                return
            psi = phi // 2
            if phi % 2 == 0:
                recursivelyCalcP(lam - 1, psi)
            sigma = 0
            for l in range(L):
                if not self.activePath[l]:
                    continue
                pLam = getArrayPointer(lam, l)
                pLam_ = getArrayPointer(lam - 1, l)
                for beta in range(2**(m-lam)):
                    if phi % 2 == 0:
                        for u in (0,1):
                            assert P[lam-1, pLam_, 2*beta, u] != -1
                            assert P[lam-1, pLam_, 2*beta, u^1] != -1
                            assert P[lam-1, pLam_, 2*beta+1, 0] != -1
                            assert P[lam-1, pLam_, 2*beta+1, 1] != -1
                            P[lam, pLam, beta, u] = sum(.5 * P[lam-1, pLam_, 2*beta, u ^ upp]
                                                           * P[lam-1, pLam_, 2*beta+1, upp]
                                                        for upp in (0, 1))
                            sigma = max(sigma, P[lam, pLam, beta, u])
                    else:
                        u = C[lam, pLam, beta, 0]
                        assert u != -1
                        for upp in (0, 1):
                            assert P[lam-1, pLam_, 2*beta, u^upp] != -1
                            assert P[lam-1, pLam_, 2*beta+1, upp] != -1
                            P[lam, pLam, beta, upp] = .5 * P[lam-1, pLam_, 2*beta, u ^ upp]\
                                                         * P[lam-1, pLam_, 2*beta+1, upp]
                            sigma = max(sigma, P[lam, pLam, beta, upp])
            assert sigma != 0
            for l in range(L):
                if self.activePath[l]:
                    pLam = getArrayPointer(lam, l)
                    for beta in range(2**(m-lam)):
                        for u in (0, 1):
                            P[lam, pLam, beta, u] /= sigma

        def recursivelyUpdateC(lam, phi):
            assert phi % 2 == 1
            psi = phi // 2
            for l in range(L):
                if not self.activePath[l]:
                    continue
                pLam  = getArrayPointer(lam, l)
                pLam_ = getArrayPointer(lam-1, l)
                for beta in range(2**(m-lam)):
                    C[lam-1, pLam_, 2*beta, psi % 2] = C[lam, pLam, beta, 0] ^ C[lam, pLam, beta, 1]
                    C[lam-1, pLam_, 2*beta+1, psi % 2] = C[lam, pLam, beta, 1]
            if psi % 2 == 1:
                recursivelyUpdateC(lam-1, psi)

        probForks = np.empty((L, 2), np.double)
        contForks = np.empty((L, 2), np.bool)

        def continuePaths(phi):
            """Implementation of continuePaths_UnfrozenBit (Algorithm 18)"""
            i = 0
            for l in range(self.L):
                if not self.activePath[l]:
                    probForks[l, 0] = probForks[l, 1] = -1
                else:
                    Pm = getArrayPointer(m, l)
                    assert P[m, Pm, 0, 0] != -1
                    assert P[m, Pm, 0, 1] != -1
                    probForks[l, 0] = P[m, Pm, 0, 0]
                    probForks[l, 1] = P[m, Pm, 0, 1]
                    i += 1
            rho = min(2*i, L)
            # implements line 14
            srtd = np.argsort(probForks, None)
            contForks[:, :] = 0
            contForks.flat[srtd[-rho:]] = 1
            for l in range(L):
                if not self.activePath[l]:
                    continue
                if (not contForks[l, 0]) and (not contForks[l, 1]):
                    killPath(l)
            for l in range(L):
                if not self.activePath[l]:
                    continue
                if (not contForks[l, 0]) and (not contForks[l, 1]):
                    continue
                Cm = getArrayPointer(m, l)
                if contForks[l, 0] and contForks[l, 1]:
                    C[m, Cm, 0, phi % 2] = 0
                    lp = clonePath(l)
                    Cm = getArrayPointer(m, lp)
                    C[m, Cm, 0, phi % 2] = 1
                elif contForks[l, 0]:
                    C[m, Cm, phi % 2] = 0
                else:
                    C[m, Cm, phi % 2] = 1

        def findMostProbablePath():
            lp = -1
            pp = 0
            for l in range(self.L):
                if not self.activePath[l]:
                    continue
                point = getArrayPointer(m, l)
                if pp < P[m, point, 0, C[m, point, 0, 1]]:
                    lp = l
                    pp = P[m, point, 0, C[m, point, 0, 1]]
            assert lp != -1
            return lp

        l = assignInitialPath()
        p0 = getArrayPointer(0, l)
        for beta in range(n):
            P[0, p0, beta, 1] = 1 / (np.exp(self.llrs[beta])+1)
            P[0, p0, beta, 0] = 1 - P[0, p0, beta, 1]

        for phi in range(n):
            recursivelyCalcP(m, phi)
            if phi in self.code.frozen:
                for l in range(self.L):
                    if self.activePath[l]:
                        Cm = getArrayPointer(m, l)
                        C[m, Cm, 0, phi % 2] = 0
            else:
                continuePaths(phi)
            if phi % 2 == 1:
                recursivelyUpdateC(m, phi)
        l = findMostProbablePath()
        C0 = getArrayPointer(0, l)
        for beta in range(n):
            assert C[0, C0, beta, 0] != -1
            self.solution[beta] = C[0, C0, beta, 0]
        self.objectiveValue = np.dot(self.solution, self.llrs)
        print(self.solution in self.code)

    def params(self):
        return OrderedDict(name=self.name)