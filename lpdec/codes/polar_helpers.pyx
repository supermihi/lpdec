# -*- coding: utf-8 -*-
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, initializedcheck=False
#  Copyright 2014-2015 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation

"""This module contains helpers for the (approximate) construction of polar codes due to Tal and
Vardy. Because the construction is time-consuming, it is a compiled Cython module."""

from __future__ import division
from libc.math cimport log2, sqrt
import numpy as np
cimport numpy as np
from numpy.math cimport INFINITY
cimport cython
import weakref


cdef double C(double x, double y) nogil:
    """C function needed to calculate deltaI in degrading-merge.
    """
    if x == 0 and y == 0:
        return 0
    elif x == 0:
        return y
    elif y == 0:
        return x
    else:
        return -(x+y)*log2((x+y)/2)+x*log2(x)+y*log2(y)


cdef class DataElement:
    """Data structure for the degrading-merge procedure, as described in
    :cite:`TalVardy13ConstructPolar`."""
    cdef public double a, b, aprime, bprime, deltaI
    cdef public DataElement left, right
    cdef public int h

    def __cinit__(self, a, b, ap, bp):
        self.a = a
        self.b = b
        self.h = -1 # heap index
        self.aprime = ap
        self.bprime = bp
        self.calcDeltaI()
        self.left = self.right = None

    cdef void calcDeltaI(self):
        self.deltaI = C(self.a, self.b) + C(self.aprime, self.bprime) \
                        - C(self.a+self.aprime, self.b+self.bprime)

    def __str__(self):
        return 'DataElement({})'.format(self.deltaI)

    __repr__ = __str__


cdef void heapify(DataElement[:] data, int heapSize, int index):
    """Heapifies a C array of DataElements for given heapSize and index."""
    cdef:
        int left = 2*index+1
        int right= 2*index+2
        int smallest = index
    if left < heapSize and data[left].deltaI < data[index].deltaI:
        smallest = left
    if right < heapSize and data[right].deltaI < data[smallest].deltaI:
        smallest = right
    if smallest != index:
        data[index], data[smallest] = data[smallest], data[index]
        data[index].h = index
        data[smallest].h = smallest
        heapify(data, heapSize, smallest)

cdef void update(DataElement[:] data, int heapSize, int index):
    """Restore heap property if element with index *index* has changed its value."""
    cdef int parent
    # first case: value decreased -> swap with parents, if necessary
    while index > 0 and data[index].deltaI < data[(index-1)//2].deltaI:
        parent = (index-1)//2
        # tmp = data[index]
        # data[index] = data[parent]
        # data[parent] = tmp
        data[parent], data[index] = data[index], data[parent]
        data[parent].h = parent
        data[index].h = index
    # second case: value increased -> call heapify
    heapify(data, heapSize, index)


cdef class BMSChannel:
    """
    A binary memoryless symmetric channel. Characterized by the vector :attr:`Wgiven0`
    containing the probabilities, for all output symbels y, that y is observed when 0 is sent.

    The elements are ordered in such a way that conjugate elements are adjacent, i.e.,
    for an index i, the elements y_i and y_{i^1} are adjacent (^ is bit-wise XOR)."""


    cdef double[:] Wgiven0
    cdef int length
    cdef BMSChannel degraded, ari1, ari2

    def __init__(self, length):
        assert length % 2 == 0
        self.Wgiven0 = np.empty(length, dtype=np.double)
        self.length = length
        self.degraded = self.ari1 = self.ari2 = None

    def __getitem__(self, tpl):
        """Returns W(y|x), where tpl=(y,x)."""
        cdef int y, x
        y, x = tpl
        return self.Wgiven0[y ^ x]

    def __setitem__(self, tpl, double value):
        """Sets W(y|x)=value, where tpl=(y,x)."""
        cdef int x, y
        y, x = tpl
        self.Wgiven0[y ^ x] = value

    def LR(self, int y):
        """Return the likelihood ratio of output element y"""
        if self.Wgiven0[y^1] == 0:
            return INFINITY
        return self.Wgiven0[y] / self.Wgiven0[y^1]

    @staticmethod
    def BSC(eps):
        """Create a binary symmetric channel.

        :param float eps: crossover probability.
        """
        chan = BMSChannel(2)
        chan[0, 0] = 1 - eps
        chan[1, 0] = eps
        return chan

    @staticmethod
    def BEC(eps):
        """Create a binary erasure channel. Note that we create
        two erasure symbols in order to fulfill the assumption of no self-conjugates.

        :param float eps: erasure probability.
        """
        chan = BMSChannel(4)
        # 2 and 3 are the erasure symbols
        chan[2, 0] = chan[3, 0] = eps/2
        chan[0, 0] = 1 - eps
        chan[1, 0] = 0
        return chan

    @cython.wraparound(True)
    @staticmethod
    def AWGNC(SNR, nu, rate=1):
        """Computes a degraded version of the AWGN channel, discretized to :math:`2\cdot \\nu`
        values, for given *SNR* in dB. The SNR value is interpreted wrt. channel symbols; if you
        want to supply SNR_b value (SNR per information bit), you.ll also have to specify *rate*.
        """
        import sympy
        from scipy.optimize import newton
        from scipy.stats import norm
        y = sympy.symbols('y')
        SNR = 10 ** (SNR / 10) # SNR is specified in dB
        lamb = sympy.exp(4*rate*y*SNR)  # likelihood ratio as function of y
        C = 1 - lamb/(1+lamb)*sympy.log(1+1/lamb, 2) - 1/(lamb+1)*sympy.log(lamb+1, 2)
        lC = sympy.lambdify(y, C, 'numpy')
        Cprime = sympy.simplify(sympy.diff(C, y))
        lCprime = sympy.lambdify(y, Cprime, 'numpy')
        Cprimeprime = sympy.simplify(sympy.diff(Cprime, y))
        lCprimeprime = sympy.lambdify(y, Cprimeprime, 'numpy')
        Ai = np.zeros(nu+1) # breakpoints defined by Eq. (33)
        Ai[0] = 0
        for i in range(1, nu):
            # we find the next breakpoint using Newton's method
            lCi = sympy.lambdify(y, C-i/nu, 'numpy')
            Ai[i] = newton(lCi, x0=Ai[i-1] + .5/nu, fprime=lCprime, fprime2=lCprimeprime)
        Ai[-1] = INFINITY
        chan = BMSChannel(2*nu)
        rv0 = norm(loc=1, scale=np.sqrt(1 / (2 * SNR))) # f(y|0)
        rv1 = norm(loc=-1, scale=np.sqrt(1 / (2 * SNR))) # f(y|1)
        for i in range(nu):
            chan.Wgiven0[2*i] = rv0.cdf(Ai[i+1]) - rv0.cdf(Ai[i])
            chan.Wgiven0[2*i+1] = rv1.cdf(Ai[i+1]) - rv1.cdf(Ai[i])
        return chan

    def arikanTransform1(self, int mu=0):
        """Output the channel W[*]W, i.e. Arikan's first channel transformation fed with two copies
        of *self*.

        If *mu* is provided and not 0, the resulting channel will be degrading-merged with parameter
        *mu* before being returend.
        """
        cdef int y1, y2
        if not self.ari1:
            output = BMSChannel(self.length**2)
            for y1 in range(self.length):
                for y2 in range(self.length):
                    # for this transform holds: W(y1, y2|0) = w(y1, \bar y_2|1) (easily checked),
                    # s.t. \bar{y1, y2} = y1, \bar y1. Therefore, we order the elements of W[*]W as:
                    # (0,0) (0,1) (0,2) ... (0,l-1) (1,0) (1,1) .... such that conjugates are adjacent.
                    output[y1*self.length+y2, 0] = 0.5*sum(self[y1, x2]*self[y2, x2] for x2 in (0, 1))
            if mu != 0:
                output = output.degradingMerge(mu)
            self.ari1 = output
        return self.ari1

    def arikanTransform2(self, int mu=0):
        """Output the channel :math:`W\circledast W`, i.e. Arikan's second channel transformation.

        If *mu* is provided and not 0, the resulting channel will be degrading-merged with parameter
        *mu* before being returend.
        """
        cdef int u1, y1, y2
        if not self.ari2:
            output = BMSChannel(2*self.length**2)
            i = 0
            for u1 in (0, 1):
                for y1 in range(0, self.length, 2):
                    for y2 in range(0, self.length, 2):
                        # for this transform holds: W(y1, y2, u1 | 0) = W(\bar y1, \bar y2, u1 | 1),
                        # such that we can define \bar{y1, y2}, u1 = \bar y1, \bar y2, u1.
                        # For fixed u1, we traverse the y1, y2 (elements of (length \times length)) in
                        # the following scheme:
                        #   | 0 1 2 3 4
                        # --+--------
                        # 0 | 1 4 5 8
                        # 1 | 3 2 7 6
                        # 2 | 9 12 ..
                        # 3 |11 10 ..
                        # 4 | .. . . ..
                        # One can easily check that then in the output adjacent entries are conjugates.
                        output[i,   0] = .5*self[y1, u1]*self[y2,0]
                        output[i+1, 0] = .5*self[y1+1,u1]*self[y2+1,0]
                        output[i+2, 0] = .5*self[y1+1,u1]*self[y2,0]
                        output[i+3, 0] = .5*self[y1, u1]*self[y2+1,0]
                        i += 4
            if mu != 0:
                output = output.degradingMerge(mu)
            self.ari2 = output
        return self.ari2

    def errorProbability(self):
        """Return the probability of error of this channel, calculated from (13) in Tal and
        Vardy's paper on polar code construction."""
        return .5*sum(min(self[i, 0], self[i, 1]) for i in range(self.length))

    def sortAndChoose(self):
        """Performs reordering of output symbols such that
        * :math:`LR(y_i) \geq 1` for all even :math:`i`,
        * :math:`LR(y_i) \leq LR(y_{i+2})` for all even :math:`i`.
        """
        # 1. swap elements such that LR(y) ≥ 1 for representatives
        y = np.asarray(self.Wgiven0)
        yc = y.copy()
        for i in range(0, self.length, 2):
            if y[i] < y[i+1]:
                y[i], y[i+1] = y[i+1], y[i]
        LRs = np.array([self.LR(i) for i in range(0, self.length, 2)])
        sortInd = np.argsort(LRs)
        y[::2] = y[2*sortInd]
        y[1::2] = y[2*sortInd+1]

    def degradingMerge(self, int mu):
        """The degrading-merge function, as described in the paper."""
        if self.degraded:
            return self.degraded
        cdef DataElement dLeft, dRight, d
        cdef double aplus, bplus
        cdef int i, heapSize, L, j, nu
        cdef BMSChannel chan
        #cdef list data
        L = self.length // 2
        cdef DataElement[:] data = np.empty(L, dtype=object)
        if self.length <= mu:
            return self
        self.sortAndChoose()
        nu = mu // 2
        for j in range(L-1):
            d = DataElement(*self.Wgiven0[2*j:2*j+4])
            if j > 0:
                data[j-1].right = d
                d.left = data[j-1]
            data[j] = d
        heapSize = L-1
        for i in range(L//2-1, -1, -1):
            heapify(data, heapSize, i)
        for i in range(heapSize):
            data[i].h = i
        while heapSize > nu - 1:
            d = data[0] # pop min element in the heap
            # restore heap property
            data[0] = data[heapSize-1]
            data[0].h = 0
            heapSize -= 1
            heapify(data, heapSize, 0)
            aplus = d.a + d.aprime
            bplus = d.b + d.bprime
            dLeft = d.left
            dRight = d.right
            if d.left is not None:
                d.left.right = d.right
            if d.right is not None:
                d.right.left = d.left
            if dLeft is not None:
                dLeft.aprime = aplus
                dLeft.bprime = bplus
                dLeft.calcDeltaI()
                update(data, heapSize, dLeft.h)
            if dRight is not None:
                dRight.a = aplus
                dRight.b = bplus
                dRight.calcDeltaI()
                update(data, heapSize, dRight.h)
        chan = BMSChannel(mu)
        i = 0
        foundRightmost = False
        for j in range(heapSize):
            d = data[j]
            chan.Wgiven0[i] = d.a
            chan.Wgiven0[i+1] = d.b
            i += 2
            if d.right is None:
                chan.Wgiven0[i] = d.aprime
                chan.Wgiven0[i+1] = d.bprime
                i += 2
                foundRightmost = True
        self.degraded = chan
        return chan

    def bhattacharyya(self):
        """Compute and return the Bhattacharyya parameter of this channel."""
        return sum(sqrt(self[i, 0]*self[i, 1]) for i in range(self.length))

    def __str__(self):
        return self.__class__.__name__+'(' + ','.join('W({}|0)={}'.format(y,self[y,0])
                                                      for y in range(self.length)) + ')'
