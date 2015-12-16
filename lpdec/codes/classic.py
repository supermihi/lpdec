# -*- coding: utf-8 -*-
# Copyright 2014-2015 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation
from __future__ import division
import itertools
import numpy as np
from collections import OrderedDict
from lpdec.codes import BinaryLinearBlockCode
from lpdec.codes.nonbinary import NonbinaryLinearBlockCode


class HammingCode(BinaryLinearBlockCode):
    """A class for Hamming codes, both standard and extended.

    For given :math:`r \geq 2`, creates the Hamming code of block length :math:`2^r-1` or the
    extended Hamming code of block length :math:`2^r`; the information length of both types ist
    :math:`2^r-r-1`.

    :param int r: determines the size of the code
    :param bool extended: set to ``True`` to create the extended Hamming code instead of the
        "normal" one.
    """

    def __init__(self, r, extended=False):
        blocklength = 2 ** r - (0 if extended else 1)
        infolength = 2 ** r - r - 1
        name = '({},{}) {}Hamming Code'.format(blocklength, infolength,
                                               'Extended ' if extended else '')
        pcm = np.zeros((blocklength - infolength, blocklength), dtype=np.int)
        colIndex = 0
        for numOnes in range(1, r + 1):
            for positions in itertools.combinations(range(r), numOnes):
                column = [0] * r + ([1] if extended else [])
                for pos in positions:
                    column[pos] = 1
                pcm[:, colIndex] = column
                colIndex += 1
        if extended:
            pcm[r, blocklength - 1] = 1
        BinaryLinearBlockCode.__init__(self, name=name, parityCheckMatrix=pcm)
        self.r = r
        self.extended = extended

    def params(self):
        ans = OrderedDict(r=self.r)
        if self.extended:
            ans['extended'] = True
        return ans


class ReedMullerCode(BinaryLinearBlockCode):
    """Reed-Muller code using the "polar code like" construction with :math:`F^{\otimes 2}` as
    generator matrix.

    Parameters
    ----------
    r : int
        Order of the RM code.
    m : int
        2-logarithm of the code's length: the code's length will be :math:`2^m`.

    """

    def __init__(self, m, r=None, infolength=None):
        F = np.array([[1, 0], [1, 1]]) # polar kernel matrix
        Fkron = np.ones((1, 1))
        self.m = m
        self.r = r
        # take Kronecker product m times
        for i in range(m):
            Fkron = np.kron(Fkron, F)
        assert Fkron.shape == (2**m, 2**m)
        # filter rows that have weight < 2**(m-r)
        if r is not None:
            assert r <= m
            assert infolength is None
            Fkron = Fkron[Fkron.sum(1) >= 2**(m-r)]
            name = 'RM({},{})'.format(r, m)
        else:
            weight = 2
            while Fkron.shape[0] > infolength:
                Fkron = Fkron[Fkron.sum(1) >= weight]
                weight *= 2
            cutIndices = (Fkron.sum(1) == weight)[:(Fkron.shape[0] - infolength)]
            Fkron = Fkron[[i for i in range(Fkron.shape[0]) if i not in cutIndices]]
            name = '({},{}) RM'.format(2**m, infolength)

        BinaryLinearBlockCode.__init__(self, generatorMatrix=Fkron, name=name)

    def params(self):
        ans = OrderedDict()
        if self.r:
            ans['r'] = self.r
        else:
            ans['infolength'] = self.infolength
        ans['m'] = self.m
        return ans


class TernaryGolayCode(NonbinaryLinearBlockCode):

    def __init__(self):
        K = np.array([[1, 1, 1, 2, 2, 0],
                      [1, 1, 2, 1, 0, 2],
                      [1, 2, 1, 0, 1, 2],
                      [1, 2, 0, 1, 2, 1],
                      [1, 0, 2, 2, 1, 1]])
        H = np.hstack((K, np.eye(5, dtype=np.int)))
        NonbinaryLinearBlockCode.__init__(self, parityCheckMatrix=H, name='TernaryGolayCode', q=3)

    def params(self):
        return OrderedDict()

class NonbinarySPCCode(NonbinaryLinearBlockCode):

    """Creates all-ones (or, if `value` != 1, all-`value`) single parity-check (SPC) code of length `length`.

    Note: To create random-valued SPC codes, use the codes.random module.
    """
    def __init__(self, q, length, value=1):
        assert value < q and value > 0
        H = np.ones((1, length), dtype=np.int)
        if value != 1:
            H[:, :] = value
        NonbinaryLinearBlockCode.__init__(self, parityCheckMatrix=H, name='SPC', q=q)


    def params(self):
        params = OrderedDict(name=self.name)
        params['q'] = self.q
        if not self.parityCheckMatrix[0, 0] != 1:
            params['value'] = int(self.parityCheckMatrix[0, 0])
        return params