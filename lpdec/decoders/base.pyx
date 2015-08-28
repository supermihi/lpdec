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


    Parameters
    ----------
    code : LinearBlockCode
        The code on which this decoder operates.
    name : str
        Name of the decoder. If used in a database, the name must be unique.

    Attributes
    ----------
    code : LinearBlockCode
        The code instance for which the decoder is configured.
    foundCodeword : bool
        This flag holds whether a valid codeword has been found or not. Should be set by
        implementing subclasses within :func:`solve`.
    mlCertificate : bool
        This flag holds whether the decoder has the ML certificate about the decoder output: if
        this is true after calling :func:`solve`, the solution is guaranteed to be the ML codeword.
        Subclasses should set this within :func:`solve`.
    status : {OPTIMAL, INFEASIBLE, UPPER_BOUND_HIT, LIMIT_HIT}
        More fine-grained status report of the decoding process. Of interest mainly in the context
        of branch-and-bound decoding.
    llrs : double[::1]
        Log-likelihood ratios of channel output used for decoding.
    name : str
        Name of the decoder. If used in a database, the name must be unique.
    """

    OPTIMAL = 0
    INFEASIBLE = 1
    UPPER_BOUND_HIT = 2
    LIMIT_HIT = 3

    def __init__(self, code, name):
        self.code = code
        if name is None:
            raise ValueError('A decoder must have a name')
        self.llrs = np.zeros(code.blocklength * (code.q - 1), dtype=np.double)
        self.solution = np.zeros(code.blocklength, dtype=np.double)
        self.name = name
        self.hint = None
        self.mlCertificate = self.foundCodeword = False
        self.status = Decoder.OPTIMAL
        self.setStats(OrderedDict())


    cpdef setLLRs(self, double[::1] llrs, np.int_t[::1] sent=None):
        """
        setLLRs(self, llrs, sent=None)

        Set the LLR vector for decoding.

        Optionally, the codeword that was actually sent might be given as well, which can speed up
        simulations if the decoder can exploit this knowledge. Of course, this is only relevant for
        decoding performance simulations, as in reality th sent word is not available.

        Parameters
        ----------
        llrs : double[::1]
            Vector of channel output log-likelihood ratios.
        sent : np.int_t[::1], optional
            Optionally, the sent codeword.
        """
        self.llrs = llrs
        self.sent = sent

    cpdef solve(self, double lb=-INFINITY, double ub=INFINITY):
        """
        solve(self, lb=-INFINITY, ub=INFINITY)

        Run the solver on :attr:`llrs`. A codeword may be given as hint.

        This is the main method to run the decoding algorithm; the LLR vector must be set
        in advance via the :attr:`llrs` attribute or through :func:`decode`.

        Parameters
        ----------
        lb : double, optional
            Lower bound on the optimal objective value

        ub : double, optional
            Upper bound on the optimal objective value. Allows the decoder to terminate when the
            optimal value is proven to be greater than `ub`.
        """
        raise NotImplementedError()

    def decode(self,
               double[::1] llrs,
               np.int_t[::1] sent=None,
               double lb=-INFINITY,
               double ub=INFINITY):
        """
        decode(self, llrs, sent=None, lb=-INFINITY, ub=INFINITY)

        Decode the given LLR vector and return the solution.

        This convenience function sets the LLR vector, calls :func:`solve`, and returns :attr:`solution`.

        Parameters
        ----------
        llrs : double[::1]
            LLR vector to decode.
        sent : np.int_t[::1], optional
            The sent codeword, if it is to be revealed.
        lb : double, optional
            known lower bound on objective value.
        ub : known upper bound on objective value.

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

    cpdef fixed(self, int i):
        """Returns True if and only if the given index is fixed."""
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
    a longer code :math:`\tilde C` such that :math:`C` is the projection of `\\tilde C` onto the
    first :math:`n` coordinates. The ProjectionDecoder inserts zeros into the llrs and cuts the
    solution such that it acts like a decoder for the projected code to the outside world.

    Parameters
    ----------
    code : LinearBlockCode
        The projected code :math:`C`
    longCode : LinearBlockCode
        The long (original) code :math:`\tilde C`.
    decoderCls : Decoder
        Class of the wrapped decoder for the long code.
    decoderParams
        Additional parameters to the wrapped decoder (do not include the code).
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