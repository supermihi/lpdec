# -*- coding: utf-8 -*-
# cython: embedsignature=True
# Copyright 2014 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation
from __future__ import division, unicode_literals
from collections import OrderedDict
import numpy as np
cimport numpy as np
from lpdec.persistence cimport JSONDecodable
from lpdec.codes import BinaryLinearBlockCode


cdef class Decoder(JSONDecodable):
    """
    The base class for all decoders, defining a minimal set of methods.

    The first constructor argument of all subclasses of Decoder should always be the
    code to decode. The :func:`params` method should only return the remaining
    arguments.

    .. attribute:: code

        The :class:`lpdec.codes.BinaryLinearBlockCode` instance for which the decoder is
        configured.
    .. attribute:: foundCodeword

        This flag holds whether a valid codeword has been found or not. Should be set by
        implementing subclasses within :func:`solve`.
    .. attribute:: mlCertificate

        This flag holds whether the decoder has the ML certificate about the decoded codeword.
        Should be set in implementing subclasses within :func:`solve`.
    .. attribute:: llrs

        Vector of log-likelihood ratios as :class:`np.ndarray`.
    """

    def __init__(self, code, name):
        self.code = code
        if name is None:
            raise ValueError('A decoder must have a name')
        self.llrs = np.zeros(code.blocklength, dtype=np.double)
        self.solution = np.zeros(code.blocklength, dtype=np.double)
        self.name = name
        self.mlCertificate = self.foundCodeword = False
        self.setStats(OrderedDict())


    cpdef setLLRs(self, np.double_t[:] llrs, np.int_t[:] sent=None):
        """Set the LLR vector for decoding. Optionally, the codeword that was actually sent might be
        given as well, which might speed up simulations if the decoder can exploit this knowledge.
        Of course that is only relevant for performance curve generation, since in real scenarios
        the sent word is not available to the decoder.
        """
        self.llrs = np.asarray(llrs, dtype=np.double)
        self.sent = sent

    cpdef solve(self, double lb=-np.inf, double ub=np.inf):
        """Run the solver on :attr:`llrs`. A codeword may be given as hint.

        This is the main method to run the decoding algorithm; the LLR vector must be set
        in advance via the :attr:`llrs` attribute.

        ``lb`` (optional) - lower bound on the optimal objective value

        `` ub`` (optional) - upper bound on the optimal objective value. Allows the decoder to
        terminate if the optimal value is proven to be greater than the bound.
        """
        raise NotImplementedError()

    def decode(self,
               np.double_t[:] llrs,
               np.int_t[:] sent=None,
               double lb=-np.inf,
               double ub=np.inf):
        """Decode the given LLR vector and return the solution.

        This convenience function sets the LLR vector, calls solve(), and return self.solution.
        """
        self.setLLRs(llrs, sent)
        self.solve(lb, ub)
        return self.solution

    cpdef setStats(self, object stats):
        self._stats = stats

    cpdef object stats(self):
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


