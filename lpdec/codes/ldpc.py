# -*- coding: utf-8 -*-
# Copyright 2014 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation
from __future__ import division
from collections import OrderedDict
import numpy as np
from lpdec.codes import BinaryLinearBlockCode


class ArrayLDPCCode(BinaryLinearBlockCode):
    """An array LDPC code is defined by an odd prime :math:`q` and a number :math:`m \leq q`.

    For the formal definition see:
    J. L. Fan, "Array codes as low-density parity-check codes", in Proc.
    2nd Int. Symp. Turbo Codes & Rel. Topics, Brest, France, Sep. 2000, pp. 543–546.
    """
    def __init__(self, q, m):
        assert m <= q
        self.q, self.m = q, m
        hmatrix = np.empty((q * m, q * q), dtype=np.int)
        P = np.zeros((q, q))
        P[0, q-1] = 1
        for i in range(q - 1):
            P[i + 1, i] = 1
        for i in range(m):
            for j in range(q):
                hmatrix[q * i:q * (i + 1), q * j:q * (j + 1)] = np.linalg.matrix_power(P, i * j)
        BinaryLinearBlockCode.__init__(self, parityCheckMatrix=hmatrix,
                                       name="({0},{1}) Array LDPC Code".format(q, m))

    def params(self):
        return OrderedDict([('q', self.q), ('m', self.m)])
