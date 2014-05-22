# -*- coding: utf-8 -*-
# Copyright 2014 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation
from __future__ import division
import itertools
import numpy as np
from collections import OrderedDict
from lpdec.codes import BinaryLinearBlockCode


class HammingCode(BinaryLinearBlockCode):
    """A class for Hamming codes, both standard and extended.

    For given :math:`r \geq 2`, creates the Hamming code of block length :math:`2^r-1` or the
    extended Hamming code of block length :math:`2^r`; the information length of both types ist
    :math:`2^r-r-1`.

    :param int r: determines the size of the code
    :param bool extended: set to ``True`` to create the extended Hamming code instead of the
        "normal" one.
    """

    # noinspection PyArgumentList
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

