# -*- coding: utf-8 -*-
# Copyright 2014 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation

from collections import OrderedDict
import numpy as np
from lpdec import utils
from lpdec.codes import BinaryLinearBlockCode
from lpdec.decoders import Decoder
from lpdec.codes.factorgraph import FactorGraph, VariableNode, CheckNode
from lpdec.codes.polar_helpers import BMSChannel


class PolarCode(BinaryLinearBlockCode):
    """Class for representing polar codes (see :cite:`Arikan09Polarization`).

    :param int n: Exponent of the block length, which will then be :math:`2^n`.
    :param iterable frozen: Indices of the frozen input bits.
    :param str name: Code name. Defaults to *PolarCode(n=<n>, frozen=<frozen>)*.

    The code's information length computes as ```2**n-len(frozen)```.
    The parity-check matrix is generated as described in :cite:`Goela+10LPPolar`.

    .. attribute:: factorGraph

      Factor graph of the polar code, containing auxiliary variables, as depicted in Fig. 1 of
      :cite:`TaranalliSiegel14ALPPolar`. Created on-the-fly on first access. See
      :class:`PolarFactorGraph` for details.
    """
    def __init__(self, n, frozen, name=None):
        frozen = tuple(sorted(frozen))
        if name is None:
            name = 'PolarCode(n={}, frozen={})'.format(n, repr(list(frozen)))
        BinaryLinearBlockCode.__init__(self, name=name)
        self.n = n
        self.blocklength = 2 ** n
        self.infolength = self.blocklength - len(frozen)
        self.frozen = list(frozen)

    @property
    def parityCheckMatrix(self):
        if self._parityCheckMatrix is None:
            # compute parity-check matrix by using the identity G_N = B_N F^{\otimes n} (see eq.
            # (70) of Arikan's paper)
            F = np.array([[1, 0], [1, 1]]) # polar kernel matrix
            Fkron = np.ones((1, 1))
            for i in range(self.n):
                Fkron = np.kron(Fkron, F)
            # reorder rows by bit-reversal
            G = np.empty(Fkron.shape, dtype=np.int)
            for i in range(self.blocklength):
                G[i] = Fkron[int(np.binary_repr(i, self.n)[::-1], 2)]
            # construct parity-check matrix
            self._parityCheckMatrix = G.T[self.frozen]
        return self._parityCheckMatrix

    @property
    def factorGraph(self):
        if not hasattr(self, '_factorGraph'):
            self._factorGraph = PolarFactorGraph(self.n)
            for index in self.frozen:
                self._factorGraph.u[index].frozen = True
        return self._factorGraph

    def params(self):
        ret = OrderedDict(n=self.n)
        ret['frozen'] = self.frozen
        ret['name'] = self.name
        return ret


def computeFrozenIndices(BMSC, n, mu, threshold=None, rate=None):
    """Compute frozen bit indices by the method presented in Tal and Vardy: How to Construct
    Polar Codes. There are two ways to determine the set of frozen bits, either by giving a
    threshold on the bit-channel's error probability or by specifying a target rate.

    :param BMSChannel BMSC: Initial :class:`BMSChannel` to start with
    :param int n: The blocklength is determined as :math:`N=2^n`.
    :param int mu: Granularity of channel degrading. A higher value means higher running time but
        closer approximation. Must be an even number.
    :param double threshold: If given, all bit-channels for which the approximate error probability
        :math:`P` satisfies :math:`P >` *threshold* are frozen.
    :param double rate: If given, the channels with highest error probability are frozen until
        the target rate is achieved.
    """
    def bitChannelDegrading(i):
        """Bit-Channel degrading function to compute degraded version of *i*-th bit channel."""
        Q = BMSC.degradingMerge(mu)
        i_binary = np.binary_repr(i, n)
        for j in range(n):
            if i_binary[j] == '0':
                W = Q.arikanTransform1()
            else:
                assert i_binary[j] == '1'
                W = Q.arikanTransform2()
            Q = W.degradingMerge(mu)
        return Q
    N = 2**n
    P = np.zeros(N)
    for i in range(N):
        print('computing channel {} of {}'.format(i, N))
        channel = bitChannelDegrading(i)
        P[i] = channel.errorProbability()
    if threshold:
        ind = [i for i in range(N) if P[i] > threshold]
    else:
        sortedP = np.argsort(P)
        targetLength = (1-rate)*N
        ind = sortedP[-targetLength:].tolist()
    return ind


class PolarFactorGraph(FactorGraph):
    """Sparse factor graph of a polar code, as defined in e.g. :cite:`TaranalliSiegel14ALPPolar`.

    :class:`PolarFactorGraph` has additional attributes compared to :class:`.FactorGraph`:

    .. attribute:: polarVars

    Array of variable nodes with shape (N, n+1). `polarVars[i,j]` equals :math:`s_{i,j}` in the
    paper.

    .. attribute:: polarChecks

    Array of check nodes with shape (N, n). `polarChecks[i,j]` is the check right of
    :math:`s_{i,j}` in the paper.

    Nodes in the polar factor graph have additional attributes *column* and *row*.
    """
    def __init__(self, n):
        N = 2 ** n
        self.n = n
        self.N = N
        bitReversed = lambda num: int(np.binary_repr(num, n)[::-1], 2)
        # create variables
        polarVars = np.empty((n+1, N), dtype=object)
        for column in range(n+1):
            for row in range(N):
                if column == 0:
                    # x (codeword) variables are in the 0-th column
                    identifier = 'x{}'.format(row)
                elif column == n:
                    # u (information) variables are in the n-th column
                    identifier = 'u{}'.format(bitReversed(row))
                else:
                    identifier = 's{},{}'.format(column, row)
                var = VariableNode(identifier)
                var.column = column
                var.row = row
                var.frozen = False
                polarVars[column, row] = var
        # create checks
        polarChecks = np.empty((n, N), dtype=object)
        for column in range(n):
            for row in range(N):
                polarChecks[column, row] = CheckNode('c{},{}'.format(column, row))
                polarChecks[column, row].column = column
                polarChecks[column, row].row = row
                # each check is connected to its according variable in the same column and row
                polarChecks[column, row].connect(polarVars[column, row])
        self.zStructures = []
        # stores the "z structures". Entries are tuples of the form:
        # ( (upper left var, upper check, upper right var),
        #   (lower left var, lower check, lower right var) )
        for column in range(1, n+1):
            structs = []
            k = 2**column
            for row in range(N):
                var = polarVars[column, row]
                check = polarChecks[column-1, row]
                var.connect(check)
                if row % k >= k/2:
                    rowT = row-k/2
                    checkT = polarChecks[column-1, rowT]
                    var.connect(checkT)
                    varT = polarVars[column, rowT]
                    varTPrev = polarVars[column-1, rowT]
                    varPrev = polarVars[column-1, row]
                    structs.append(((varT, checkT, varTPrev), (var, check, varPrev)))
            self.zStructures.extend(structs[::-1])
        self.u = [polarVars[n, bitReversed(i)] for i in range(N)]
        x = polarVars[0].tolist()
        vars = polarVars.flatten().tolist()
        checks = polarChecks.flatten().tolist()
        FactorGraph.__init__(self, vars, checks, x)
        self.polarVars = polarVars
        self.polarChecks = polarChecks

    def sparsify(self):
        """Makes the graph smaller by eliminating frozen variables and removing degree-2 checks
        and degree-1 auxiliary variables. The result is still a sparse graph (each check node has
        degree at most 3), but usually much smaller than the original one.

        See :cite:`TaranalliSiegel14ALPPolar` for details on the construction.
        """
        # 1. sparsify z-structures
        for structure in reversed(self.zStructures):
            (vul, cu, vur), (vll, cl, vlr) = structure
            if vul.frozen and vll.frozen:
                vul.isolate()
                vll.isolate()
                vlr.merge(vll)
                vur.frozen = vlr.frozen = True
            elif vul.frozen ^ vll.frozen:
                assert vul.frozen, str(vul) + ', ' + str(vll)
                vlr.merge(vll)
                vul.isolate()
        # remove degree-2 checks
        for column in range(self.n - 1, -1, -1):
            for row in range(self.N):
                check = self.polarChecks[column, row]
                if check.degree == 2:
                    neigh1, neigh2 = check.neighbors
                    neigh1.merge(neigh2)
        # # remove degree-1 variables
        for v in self.varNodes:
            if v.degree == 1 and v not in self.x:
                for check in v.neighbors[:]:
                    check.isolate()
                v.isolate()
        # remove isolated nodes and restore indices
        self.varNodes   = [v for v in self.varNodes   if v.degree > 0]
        self.checkNodes = [c for c in self.checkNodes if c.degree > 0]
        for i, v in enumerate(self.varNodes):
            v.index = i
        for i, c in enumerate(self.checkNodes):
            c.index = i

    def parityCheckMatrix(self):
        H = FactorGraph.parityCheckMatrix(self)
        Hunfrozen = H[:, [i for i in range(H.shape[1]) if not self.varNodes[i].frozen]]
        return Hunfrozen


class SparsePolarDecoder(Decoder):
    """Decoder wrapper class for decoding a polar code based on the sparsified
    :class:`PolarFactorGraph`. It acts as a decoder of the (smaller) original polar codes,
    filling LLR values etc. with zeros.
    """
    def __init__(self, code, decoderCls, **decoderParams):
        code.factorGraph.sparsify()
        self.longCode = BinaryLinearBlockCode(
            parityCheckMatrix=code.factorGraph.parityCheckMatrix(),
            name=code.name + '(Sparse)')
        if utils.isStr(decoderCls):
            import lpdec.imports
            decoderCls = lpdec.imports.__dict__[decoderCls]
        self.decoder = decoderCls(code=self.longCode, **decoderParams)
        self.decoderCls = decoderCls
        name = self.decoder.name
        if not name.endswith('(PolarSparse)'):
            name += '(PolarSparse)'
        Decoder.__init__(self, code, name=name)
        self.longLLR = np.random.normal(loc=0, scale=1e-9, size=self.longCode.blocklength)
        self.longSent = np.zeros(self.longCode.blocklength, dtype=np.int)

    def setStats(self, stats):
        self.decoder.setStats(stats)

    def stats(self):
        return self.decoder.stats()

    def setLLRs(self, llrs, sent=None):
        self.longLLR[:llrs.size] = llrs
        if sent is not None:
            self.longSent[:sent.size] = sent
            self.decoder.setLLRs(self.longLLR, sent=self.longSent)
        else:
            self.decoder.setLLRs(self.longLLR)

    def solve(self, lb=-np.inf, ub=np.inf):
        self.decoder.solve(lb, ub)
        self.solution = self.decoder.solution[:self.code.blocklength]
        self.objectiveValue = self.decoder.objectiveValue
        self.mlCertificate = self.decoder.mlCertificate
        self.foundCodeword = self.decoder.foundCodeword

    def params(self):
        ans = OrderedDict(decoderCls=self.decoderCls.__name__)
        for key, value in self.decoder.params().items():
            if key == 'name':
                ans[key] = self.name
            else:
                ans[key] = value
        return ans
