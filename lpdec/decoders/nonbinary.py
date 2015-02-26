# -*- coding: utf-8 -*-
# Copyright 2015 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation

from __future__ import division, print_function
import numpy as np
from collections import OrderedDict
from lpdec.utils import Timer
import itertools
from lpdec.decoders import Decoder
from lpdec.codes import nonbinary
import gurobimh as gu


class FlanaganLPDecoder(Decoder):
    """LP Decoder for non-binary code based on :cite:`Flanagan+09NonBinary`.

    The formulation generalizes the first LP formulation for binary codes in
    :cite:`Feldman+05LPDecoding`. In fact, this decoder may be used for binary codes as well,
    in which case it reduces to the binary LP decoder with auxiliary variables.

    :param ml: When *True*, the LP is solved with integer contstraints, resulting in ML decoding.
    """

    def __init__(self, code, gurobiParams=dict(), gurobiVersion=None, ml=False, name=None):
        if name is None:
            name = 'Flanagan{}Decoder'.format('ML' if ml else 'LP')
        self.ml = ml
        Decoder.__init__(self, code, name)
        self.timer = Timer()
        self.model = gu.Model()
        self.model.setParam('OutputFlag', 0)
        self.model.setParam('Threads', 1)
        self.f = OrderedDict()
        for i in range(code.blocklength):
            for k in range(1, code.q):
                var = self.model.addVar(0, 1, vtype=gu.GRB.BINARY if ml else gu.GRB.CONTINUOUS,
                                        name='f{},{}'.format(i, k))
                self.f[i, k] = var
        self.model.update()
        for j, row in enumerate(code.parityCheckMatrix):
            # generate all local codewords
            nonzeros = np.flatnonzero(row)
            spc = row[nonzeros]
            numNonzeros = nonzeros.size
            codewords = []
            vars = []
            for i, localword in enumerate(itertools.product(list(range(code.q)),
                                          repeat=numNonzeros-1)):
                modq = np.dot(localword, spc[:-1]) % code.q
                localword += (-modq * nonbinary.inv(spc[-1], code.q) % code.q,)
                codewords.append(localword)
                var = self.model.addVar(0, 1, name='w_{},{}'.format(j, i))
                vars.append(var)
            self.model.update()
            self.model.addConstr(gu.quicksum(vars), gu.GRB.LESS_EQUAL, 1)
            for i, ij in enumerate(nonzeros):
                for alpha in range(1, code.q):
                    if any(codewords[j][i] == alpha for j in range(len(vars))):
                        self.model.addConstr(gu.quicksum(vars[j] for j in range(len(vars)) if
                                                         codewords[j][i] == alpha), gu.GRB.EQUAL,
                                             self.f[ij, alpha])
                    else:
                        print('no {} {} {}'.format(alpha, j, ij))
                        print(self.code.parityCheckMatrix[j])
                        raise NotImplementedError()
        self.model.update()
        self.xvars = list(self.f.values())
        self.solution = np.empty(code.blocklength)

    def setStats(self, stats):
        if 'lpTime' not in stats:
            stats['lpTime'] = 0.0
        if 'simplexIters' not in stats:
            stats['simplexIters'] = 0
        Decoder.setStats(self, stats)

    def solve(self, lb=-np.inf, ub=np.inf):
        self.mlCertificate = self.foundCodeword = True
        self.model.setObjective(gu.LinExpr(self.llrs, self.xvars))
        with self.timer:
            self.model.optimize()
        self._stats['lpTime'] += self.timer.duration
        self._stats['simplexIters'] += self.model.IterCount
        if self.model.Status == gu.GRB.OPTIMAL:
            self.objectiveValue = self.model.ObjVal
            for i in range(self.code.blocklength):
                self.solution[i] = 0
                for k in range(1, self.code.q):
                    if self.f[i, k].X > 1e-5:
                        if self.solution[i] != 0:
                            self.mlCertificate = self.foundCodeword = False
                            self.solution[:] = .5  # error
                            return
                        else:
                            self.solution[i] = k

        else:
            raise RuntimeError()

    def params(self):
        ret = OrderedDict(ml=self.ml)
        ret['name'] = self.name
        return ret


if __name__ == '__main__':
    from lpdec.codes.classic import TernaryGolayCode
    from lpdec.imports import *

    code = TernaryGolayCode()
    decML = FlanaganLPDecoder(code, ml=True)
    decLP = FlanaganLPDecoder(code, ml=False)
    print(code.parityCheckMatrix)
    print(code.blocklength)
    print(code.infolength)
    from lpdec import simulation
    simulation.ALLOW_DIRTY_VERSION = True
    simulation.ALLOW_VERSION_MISMATCH = True
    db.init('sqlite:///:memory:')
    for snr in frange(0, 0.1, .5):
        channel = AWGNC(snr, code.rate, seed=8374, q=3)
        simulator = Simulator(code, channel, [decLP], 'ternary')
        simulator.maxSamples = 100000
        simulator.maxErrors = 200
        simulator.wordSeed = 1337
        simulator.outputInterval = 1
        simulator.dbStoreTimeInterval = 10
        simulator.revealSent = True
        simulator.concurrent = True
        simulator.run()
    print(decLP.stats())


