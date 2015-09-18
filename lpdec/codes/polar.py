# -*- coding: utf-8 -*-
# Copyright 2014-2015 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation

from collections import OrderedDict
import logging
import numpy as np
from lpdec.codes import BinaryLinearBlockCode
from lpdec.codes.factorgraph import FactorGraph, VariableNode, CheckNode
from lpdec.codes.polar_helpers import BMSChannel


class PolarCode(BinaryLinearBlockCode):
    """Class for representing polar codes (see :cite:`Arikan09Polarization`).

    The code's information length computes as ``2**n-len(frozen)``.
    The parity-check matrix is generated as described in :cite:`Goela+10LPPolar`.

    Parameters
    ----------
    n : int
        Determines the block length of the polar code as :math:`2^n`.
    name : str
        The code's name. Defaults to ``PolarCode(n=<n>, frozen=<frozen>)``.
    frozen : iterable
        Indices of the frozen input bits;  the code's information length will be
        ``2**n-len(frozen)``. For convenience, the constructor also to specify *mu*, *SNR*, *rate*,
        and *SNR_is_SNRb* (see below), which calls :func:`computeFrozenIndices` for the AWGN
        channel to compute frozen indices on the fly.
    mu : int
        Passed to :func:`computeFrozenIndices`.
    SNR : float
        Signal-to-noise ratio of the AWGN channel, in dB.
    SNR_is_SNRb : bool
        Whether the SNR is treated as information-bit oriented instead of channel-bit oriented
        ratio. Default is ``False``.
    rate : float
        Target rate; passed to :func:`computeFrozenIndices`.

    Attributes
    ----------

    factorGraph : PolarFactorGraph
        Factor graph of the polar code, containing auxiliary variables, as depicted in Fig. 1 of
        :cite:`TaranalliSiegel14ALPPolar`. Created on-the-fly on first access. See
        :class:`PolarFactorGraph` for details.
    """
    def __init__(self, n, frozen=None, name=None, **kwargs):
        if frozen is None:
            try:
                mu = kwargs['mu']
                SNR = kwargs['SNR']
                rate = kwargs['rate']
                SNR_is_SNRb = kwargs.get('SNR_is_SNRb', False)
            except KeyError:
                raise ValueError('Either frozen bits or all of (SNR, mu, rate) must be specified')
            chan = BMSChannel.AWGNC(SNR, 1000, rate=(rate if SNR_is_SNRb else 1))
            frozen = computeFrozenIndices(chan, n, mu, rate=rate)
            if name is None:
                name = 'PolarCode(n={}, SNR{}={}, mu={}, rate={})'.format(
                    n, 'b' if SNR_is_SNRb else '', SNR, mu, rate)
        frozen = sorted(frozen)
        if name is None:
            name = 'PolarCode(n={}, frozen={})'.format(n, repr(frozen))
        BinaryLinearBlockCode.__init__(self, name=name)
        self.n = n
        self.blocklength = 2 ** n
        self.infolength = self.blocklength - len(frozen)
        self.frozen = frozen

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
            self._parityCheckMatrix = G.T[self.frozen].copy()  # make C-contiguous
        return self._parityCheckMatrix

    @staticmethod
    def reedMullerCode(m, r):
        F = np.array([[1, 0], [1, 1]]) # polar kernel matrix
        Fkron = np.ones((1, 1))
        for i in range(m):
            Fkron = np.kron(Fkron, F)
        # reorder rows by bit-reversal
        G = np.empty(Fkron.shape, dtype=np.int)
        for i in range(2**m):
            G[i] = Fkron[int(np.binary_repr(i, m)[::-1], 2)]
        Gt = G.T.copy()
        frozen = np.flatnonzero(G.sum(1) < 2**(m-r)).tolist()
        return PolarCode(m, frozen=frozen, name='ReedMullerPolar({},{})'.format(m, r))

    def factorGraph(self):
        factorGraph = PolarFactorGraph(self.n)
        for index in self.frozen:
            factorGraph.u[index].frozen = True
        return factorGraph

    def params(self):
        ret = OrderedDict(n=self.n)
        ret['frozen'] = self.frozen
        ret['name'] = self.name
        return ret


def computeFrozenIndices(BMSC, n, mu, threshold=None, rate=None):
    """Compute frozen bit indices by the method presented in :cite:`TalVardy13ConstructPolar`.
    There are two ways to determine the set of frozen bits, either by giving a
    threshold on the bit-channel's error probability or by specifying a target rate.

    Parameters
    ----------
    BMSC : :class:`.BMSChannel`
        Initial binary memoryless symmetric channel to start with.
    n : int
        Determines the block length as :math:`N=2^n`.
    mu : int
        Granularity of channel degrading. A higher value means larger running time but closer
        approximation. Must be an even number.
    threshold : float
        If given, all bit-channels for which the approximate error probability :math:`P` satisfies
        :math:`P >` *threshold* are frozen (mutually exclusive with *rate* below).
    rate : float
        If given, the channels with highest error probability are frozen until the target rate is
        achiveved (the code's rate will be the smallest achievable rate that is at least *rate*).
    """
    def bitChannelDegrading(i):
        """Bit-Channel degrading function to compute degraded version of *i*-th bit channel.
        Note that this is actually Algorithm D of :cite:`TalVardy13ConstructPolar` which
        additionally employs the Bhattacharyya parameter for improved approximation.
        """
        assert mu % 2 == 0
        Z = BMSC.bhattacharyya()
        Q = BMSC.degradingMerge(mu)
        i_binary = np.binary_repr(i, n)
        for j in range(n):
            if i_binary[j] == '0':
                Q = Q.arikanTransform1(mu)
                Z = min(Q.bhattacharyya(), 2*Z-Z*Z)
            else:
                assert i_binary[j] == '1'
                Q = Q.arikanTransform2(mu)
                Z = Z*Z
        PeQ = Q.errorProbability()
        if Z < PeQ:
            return Z
        return PeQ
    N = 2**n
    P = np.zeros(N)
    for i in range(N):
        logging.info('computing channel {} of {}'.format(i, N))
        P[i] = bitChannelDegrading(i)
        #P[i] = channel.errorProbability()
    if threshold:
        ind = [i for i in range(N) if P[i] > threshold]
    else:
        sortedP = np.argsort(P)
        targetLength = int((1-rate)*N)
        ind = sortedP[-targetLength:].tolist()
        logging.info('Error probability upper bound: {}'.format(sum(P[sortedP[:-targetLength]])))
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

    Nodes in the polar factor graph have additional attributes *column* and *row*. The *varNodes*
    list is ordered such that the first :math:`2^n` nodes correspond to the variable nodes.
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
        self.isSparsified = False

    def sparsify(self):
        """Makes the graph smaller by eliminating frozen variables and removing degree-2 checks
        and degree-1 auxiliary variables. The result is still a sparse graph (each check node has
        degree at most 3), but usually much smaller than the original one.

        See :cite:`TaranalliSiegel14ALPPolar` for details on the construction.
        """
        if self.isSparsified:
            return
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
        self.varNodes   = [v for v in self.varNodes   if v.degree > 0 or v in self.x]
        self.checkNodes = [c for c in self.checkNodes if c.degree > 0]
        for i, v in enumerate(self.varNodes):
            v.index = i
        for i, c in enumerate(self.checkNodes):
            c.index = i
        self.isSparsified = True

    def parityCheckMatrix(self):
        H = FactorGraph.parityCheckMatrix(self)
        Hunfrozen = H[:, [i for i in range(H.shape[1]) if not self.varNodes[i].frozen]]
        return Hunfrozen.copy()

