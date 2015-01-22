# -*- coding: utf-8 -*-
# Copyright 2014-2015 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation

import itertools
import numpy as np
from lpdec.decoders import Decoder
from lpdec.codes.factorgraph import FactorGraph


class ErasureDecoder(Decoder):
    """A simple decoder for the binary erasure channel.

    The decoder behaves a little different than others with respect to the interpretation of
    LLRs, objective value and solution:

    * A zero LLR is interpreted as an erasure, any positive value as a guaranteed 0,
      and any negative value as a guaranteed 1. This has the same effect as multiplying the LLR
      with :math:`+\infty` in the normal meaning.
    * The objective value is :math:`-n`, where :math:`n` is the number of filled out erasures.
      An objective value of :math:`\infty` indicates a contradictory setting at a check, i.e.,
      a degree-2 check with neighbors fixed to 0 and 1, respectively. This can not occur on the
      BEC but, for instance, in the context of branch-and-bound.
    * The solution contains entries :math:`-1` at positions that could not be filled out.

    .. attribute:: corrected

      Contains 0s and 1s in the positions that were filled out
      by the algorithm, and -1s otherwise.
    """
    def __init__(self, code, name=None):
        if name is None:
            name = 'ErasureDecoder'
        Decoder.__init__(self, code, name)
        self.fg = FactorGraph.fromLinearCode(code)
        self.solution = np.zeros(code.blocklength)
        self.corrected = np.zeros(code.blocklength)

    def setStats(self, stats):
        if 'iterations' not in stats:
            stats['iterations'] = 0
        Decoder.setStats(self, stats)

    def solve(self, lb=-np.inf, ub=np.inf):
        self.corrected.setfield(-1, np.double)
        self.objectiveValue = 0
        for val, varNode in zip(self.llrs, self.fg.varNodes):
            # 0 -> -1, pos -> 0, neg -> 1
            varNode.value = -1 + (val != 0) + (val < 0)
        for check in self.fg.checkNodes:
            check.value = sum(1 for varNode in check.neighbors if varNode.value == 1)
            check.finished = False

        for iteration in itertools.count(1):
            action = False
            for check in self.fg.checkNodes:
                if check.finished:
                    continue
                erasedNeighbors = [var for var in check.neighbors if var.value == -1]
                if len(erasedNeighbors) == 1:
                    action = True
                    filledInVar = erasedNeighbors[0]
                    filledInVar.value = self.corrected[filledInVar.identifier] = check.value % 2
                    self.objectiveValue -= 1
                    for check2 in erasedNeighbors[0].neighbors:
                        check2.value += 1
                    check.finished = True
                elif len(erasedNeighbors) == 0:
                    check.finished = True
                    action = True
                    if check.value % 2 != 0:
                        self.objectiveValue = np.inf
            if not action:
                break
        self._stats['iterations'] += iteration
        for i, var in enumerate(self.fg.varNodes):
            self.solution[i] = var.value


if __name__ == '__main__':
    from lpdec.imports import *
    a = BinaryLinearBlockCode(parityCheckMatrix=[[1, 1, 0],
                                                 [0, 1, 1],
                                                 [1, 0, 1]], name='Test')
    llr = np.array([1, 0,1], dtype=np.double)
    dec = ErasureDecoder(a)
    dec.decode(llr)
    print(dec.solution)
    print(dec.objectiveValue)
    print(dec.stats())
