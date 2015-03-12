# -*- coding: utf-8 -*-
# Copyright 2014-2015 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation
from __future__ import division, unicode_literals
from collections import OrderedDict
import numpy as np
cimport numpy as np
from numpy.math cimport INFINITY
from lpdec.persistence cimport JSONDecodable
from lpdec import utils

cdef class Decoder(JSONDecodable):
    """
    The base class for all decoders, defining a minimal set of methods.

    The first constructor argument of all subclasses of Decoder should always be the
    code to decode. The :func:`params` method should only return the remaining
    arguments.

    .. attribute:: code

        The :class:`lpdec.codes.LinearBlockCode` instance for which the decoder is
        configured.
    .. attribute:: foundCodeword

        This flag holds whether a valid codeword has been found or not. Should be set by
        implementing subclasses within :func:`solve`.
    .. attribute:: mlCertificate

        This flag holds whether the decoder has the ML certificate about the decoded codeword.
        Should be set in implementing subclasses within :func:`solve`.
    .. attribute:: llrs

        Vector of log-likelihood ratios as :class:`np.ndarray`.
    .. attribute:: name

        Name of the code. If used in a database, the name must uniquely map to this code.
    """

    def __init__(self, code, name):
        self.code = code
        if name is None:
            raise ValueError('A decoder must have a name')
        self.llrs = np.zeros(code.blocklength * (code.q - 1), dtype=np.double)
        self.solution = np.zeros(code.blocklength, dtype=np.double)
        self.name = name
        self.hint = None
        self.mlCertificate = self.foundCodeword = False
        self.setStats(OrderedDict())


    cpdef setLLRs(self, double[::1] llrs, np.int_t[::1] sent=None):
        """Set the LLR vector for decoding. Optionally, the codeword that was actually sent might be
        given as well, which might speed up simulations if the decoder can exploit this knowledge.
        Of course that is only relevant for performance curve generation, since in real scenarios
        the sent word is not available to the decoder.
        """
        self.llrs = llrs
        self.sent = sent

    cpdef solve(self, double lb=-INFINITY, double ub=INFINITY):
        """Run the solver on :attr:`llrs`. A codeword may be given as hint.

        This is the main method to run the decoding algorithm; the LLR vector must be set
        in advance via the :attr:`llrs` attribute.

        :param double lb: (optional) lower bound on the optimal objective value

        :param double ub: (optional) upper bound on the optimal objective value. Allows the
        decoder to
        terminate if the optimal value is proven to be greater than the bound.
        """
        raise NotImplementedError()

    def decode(self,
               double[::1] llrs,
               np.int_t[::1] sent=None,
               double lb=-INFINITY,
               double ub=INFINITY):
        """Decode the given LLR vector and return the solution.

        This convenience function sets the LLR vector, calls solve(), and return self.solution.
        """
        self.setLLRs(llrs, sent)
        self.solve(lb, ub)
        return self.solution

    def setStats(self, object stats):
        self._stats = stats

    def stats(self):
        return self._stats

    cpdef fix(self, int index, int value):
        """Fix codeword variable with index `index` to `value` :math:`\in \{0,1\}`.
        """
        raise NotImplementedError()

    cpdef release(self, int index):
        """Release the constraint on a previously fixed variable."""
        raise NotImplementedError()

    def __str__(self):
        return self.name

    def __repr__(self):
        paramString = ", ".join("{0}={1}".format(k, repr(v)) for k, v in self.params().items())
        return "{c}(code = {cr}, {p})".format(c = self.__class__.__name__,
                                              cr = repr(self.code),
                                              p = paramString)



cdef class ProjectionDecoder(Decoder):
    """A ProjectionDecoder is used for decoding of an :math:`(n,k)` code :math:`C` by actually using
    a longer code :math:`\tilde C` such that :math:`C` is the projection of `\tilde C` onto the
    first :math:`n` coordinates. The ProjectionDecoder inserts zeros into the llrs and cuts the
    solution such that it acts like a decoder for the projected code to the outside world.

    Args:
      - code: the projected code :math:`C`
      - longCode: the long code :math:`\tilde C`
      - decoderCls: class of the wrapped decoder for the long code
      - **decoderParams: additional parameters to the wrapped decoder
    """

    cdef double[::1] longLLR
    cdef np.int_t[::1] longSent
    cdef Decoder decoder
    cdef object decoderCls, longCode
    cdef int n

    def __init__(self, code, longCode, decoderCls, **decoderParams):
        self.longCode = longCode
        if utils.isStr(decoderCls):
            import lpdec.imports
            decoderCls = lpdec.imports.__dict__[decoderCls]
        self.decoder = decoderCls(code=longCode, **decoderParams)
        self.decoderCls = decoderCls
        name = 'Proj_{}({})'.format(code.blocklength, self.decoder.name)
        Decoder.__init__(self, code, name=name)
        self.n = code.blocklength
        self.longLLR = np.zeros(longCode.blocklength, dtype=np.double)
        self.longSent = np.zeros(longCode.blocklength, dtype=np.int)

    cpdef release(self, int index):
        return self.decoder.release(index)

    cpdef fix(self, int index, int value):
        return self.decoder.fix(index, value)

    def __getattr__(self, item):
        return getattr(self.decoder, item)

    def __setattr__(self, item, value):
        setattr(self.decoder, item, value)

    def setStats(self, stats):
        self.decoder.setStats(stats)

    def stats(self):
        return self.decoder.stats()

    cpdef setLLRs(self, double[::1] llrs, np.int_t[::1] sent=None):
        self.longLLR[:self.n] = llrs
        if sent is not None:
            self.longSent[:self.n] = sent
            self.decoder.setLLRs(self.longLLR, sent=self.longSent)
        else:
            self.decoder.setLLRs(self.longLLR)

    cpdef solve(self, double lb=-INFINITY, double ub=INFINITY):
        self.decoder.solve(lb, ub)
        self.solution = self.decoder.solution[:self.n]
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