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
