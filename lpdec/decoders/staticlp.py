# -*- coding: utf-8 -*-
# Copyright 2015 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation

from __future__ import division, print_function
import itertools

import numpy as np

from lpdec import utils, gfqla
from lpdec.decoders.gurobihelpers import GurobiDecoder
import gurobimh as gu


class ExplicitLPDecoder(GurobiDecoder):
    """LP Decoder using the static explicit LP formulation (no auxiliary variables) from
    :cite:`Feldman+05LPDecoding`.
    """

    def __init__(self, code, gurobiParams=None, gurobiVersion=None, ml=False, name=None):
        if name is None:
            name = 'Explicit{}Decoder'.format('ML' if ml else 'LP')
        self.ml = ml
        GurobiDecoder.__init__(self, code, name, gurobiParams, gurobiVersion, integer=ml)
        self.timer = utils.Timer()
        assert code.q == 2, 'only binary codes are supported'
        from lpdec.polytopes import feldmanInequalities
        A, b = feldmanInequalities(code.parityCheckMatrix)
        for i in range(len(b)):
            self.model.addConstr(gu.LinExpr(A[i], self.xlist), gu.GRB.LESS_EQUAL, b[i])
        self.model.update()

    def setStats(self, stats):
        if 'lpTime' not in stats:
            stats['lpTime'] = 0.0
        if 'simplexIters' not in stats:
            stats['simplexIters'] = 0
        GurobiDecoder.setStats(self, stats)

    def solve(self, lb=-np.inf, ub=np.inf):
        with self.timer:
            self.model.optimize()
        self._stats['lpTime'] += self.timer.duration
        self._stats['simplexIters'] += self.model.IterCount
        if self.model.Status == gu.GRB.OPTIMAL:
            self.mlCertificate = self.foundCodeword = self.readSolution()
        else:
            raise RuntimeError()

    def params(self):
        ret = GurobiDecoder.params(self)
        ret['ml'] = self.ml
        return ret


class StaticLPDecoder(GurobiDecoder):
    """LP Decoder using the static LP formulation with auxiliary variables from
    :cite:`Feldman+05LPDecoding`. Also supports the linear-sized cascaded version from
    :cite:`Yang+08NewLP` and nonbinary versions of both due to :cite:`Flanagan+09NonBinary`.

    :param ml: When *True*, the LP is solved with integer contstraints, resulting in ML decoding.
    :param cascade: When *True*, use the cascaded formulation.
    """

    def __init__(self, code, gurobiParams=None, gurobiVersion=None, ml=False,
                 cascade=False, name=None):
        if name is None:
            name = '{}{}Decoder'.format('Cascaded' if cascade else 'Static', 'ML' if ml else 'LP')
        self.ml = ml
        self.cascade = cascade
        GurobiDecoder.__init__(self, code, name, gurobiParams, gurobiVersion, integer=ml)
        self.timer = utils.Timer()
        self.numChiVars = 0
        self.numWvars = 0
        for j, row in enumerate(code.parityCheckMatrix):
            nonzeros = np.flatnonzero(row)
            h = row[nonzeros]
            d = h.size
            if (self.code.q**d > 1e6):
                raise ValueError('Code too dense!')
            if cascade and d > 3:
                L = list(range(1, d - 2))
                chi = {}
                for i in L:
                    for alpha in range(1, code.q):
                        chi[i, alpha] = self.model.addVar(0, 1,
                                                          name='chi^{}_{},{}'.format(j, i, alpha))
                        self.numChiVars += 1
                xvars1 = {}
                for alpha in range(1, code.q):
                    xvars1[0, alpha] = self.x[nonzeros[0], alpha]
                    xvars1[1, alpha] = self.x[nonzeros[1], alpha]
                    xvars1[2, alpha] = chi[1, alpha]
                self.createLocalCodePolytope('{}/0'.format(j), np.array([h[0], h[1], 1]), xvars1)
                for l in range(1, d - 3):
                    xvarsl = {}
                    for alpha in range(1, code.q):
                        xvarsl[0, alpha] = chi[l, alpha]
                        xvarsl[1, alpha] = self.x[nonzeros[l+1], alpha]
                        xvarsl[2, alpha] = chi[l+1, alpha]
                    self.createLocalCodePolytope('{}/{}'.format(j, l),
                                                 np.array([code.q-1, h[l+1], 1]),
                                                 xvarsl)
                xvarsd = {}
                for alpha in range(1, code.q):
                    xvarsd[0, alpha] = chi[d - 3, alpha]
                    xvarsd[1, alpha] = self.x[nonzeros[-2], alpha]
                    xvarsd[2, alpha] = self.x[nonzeros[-1], alpha]
                self.createLocalCodePolytope('{}/{}'.format(j, d - 3),
                                             np.array([code.q - 1, h[-2], h[-1]]),
                                             xvarsd)
            else:
                vars = {(ii, alpha): self.x[i, alpha] for ii, i in enumerate(nonzeros) for alpha
                         in range(1, code.q)}
                self.createLocalCodePolytope(j, h, vars)
        self.model.update()

    def createLocalCodePolytope(self, jname, h, xvars):
        """Adds auxiliary variables and constraints for a local code polytope, given by the
        local parity-check row *h* and the according variable dictionary *xvars*.
        """
        d =h.size
        codewords = []
        auxVars = []
        q = self.code.q
        for i, localword in enumerate(itertools.product(list(range(q)),
                                      repeat=d-1)):
            modq = np.dot(localword, h[:-1]) % q
            localword += (-modq * gfqla.inv(h[-1], q) % q,)
            codewords.append(localword)
            var = self.model.addVar(0, 1, name='w_{},{}'.format(jname, i))
            auxVars.append(var)
            self.numWvars += 1
        self.model.update()
        self.model.addConstr(gu.quicksum(auxVars),
                             gu.GRB.LESS_EQUAL, 1, name='auxSum{}'.format(jname))
        for i in range(d):
            for alpha in range(1, q):
                if any(codeword[i] == alpha for codeword in codewords):
                    self.model.addConstr(gu.quicksum(auxVars[k] for k in range(len(auxVars)) if
                                                     codewords[k][i] == alpha), gu.GRB.EQUAL,
                                         xvars[i, alpha], name='consis_{}_{}_{}'.format(jname, i,
                                                                                        alpha))
                else:
                    print('no {} {} {}'.format(alpha, jname, i))
                    raise NotImplementedError()

    def setStats(self, stats):
        if 'lpTime' not in stats:
            stats['lpTime'] = 0.0
        if 'simplexIters' not in stats:
            stats['simplexIters'] = 0
        GurobiDecoder.setStats(self, stats)

    def solve(self, lb=-np.inf, ub=np.inf):
        with self.timer:
            self.model.optimize()
        self._stats['lpTime'] += self.timer.duration
        self._stats['simplexIters'] += self.model.IterCount
        if self.model.Status == gu.GRB.OPTIMAL:
            self.mlCertificate = self.foundCodeword = self.readSolution()
        else:
            raise RuntimeError()

    def params(self):
        ret = GurobiDecoder.params(self)
        ret['ml'] = self.ml
        ret['cascade'] = self.cascade
        return ret

if __name__ == '__main__':
    from lpdec.imports import *
    code = NonbinaryLinearBlockCode(parityCheckMatrix="~/UNI/LP4SOC/Codes/Nonbinary/Nonbinary_PCM_GF5_155_93.txt")
    decoder = StaticLPDecoder(code, cascade=True)
    print(np.sum(code.parityCheckMatrix != 0, 1))
    print(code.parityCheckMatrix.shape)
    print(decoder.numChiVars)
    print(decoder.numWvars)
    print(len(decoder.xlist))
