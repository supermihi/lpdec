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
from lpdec.codes cimport BinaryLinearBlockCode


cdef class Decoder(JSONDecodable):
    """The base class for all decoders, defining a minimal set of methods.

    The first constructor argument of all subclasses of Decoder should always be the
    code to decode. The :func:`params` method should only return the remaining arguments."""

    def __init__(self, code, name):
        self.code = code
        if name is None:
            raise ValueError('A decoder must have a name')
        self.llrs = np.zeros(code.blocklength, dtype=np.double)
        self.solution = np.zeros(code.blocklength, dtype=np.double)
        self.name = name
        self.mlCertificate = self.foundCodeword = False
        self.setStats(OrderedDict())


    cpdef setLLRs(self, np.double_t[:] llrs):
        self.llrs = np.asarray(llrs, dtype=np.double)

    cpdef solve(self, np.int_t[:] sent=None, double lb=-np.inf, double ub=np.inf):
        """Run the solver on :attr:`llrs`. A codeword may be given as hint.

        This is the main method to run the decoding algorithm; the LLR vector must be set
        in advance via the :attr:`llrs` attribute.

        ``sent`` (optional) - the codeword that was actually sent. Might speed up simulations
        because the error/non-error decision can be made faster, but is of course not available
        in relasistic situations.

        ``lb`` (optional) - lower bound on the optimal objective value

        `` ub`` (optional) - upper bound on the optimal objective value. Allows the decoder to
        terminate if the optimal value is proven to be greater than the bound.
        """
        raise NotImplementedError()

    def decode(self,
               np.double_t[:] llrs,
               np.int_t[:] hint=None,
               double lb=-np.inf,
               double ub=np.inf):
        """Decode the given LLR vector and return the solution.

        This convenience function sets the LLR vector, calls solve(), and return self.solution.
        """
        self.setLLRs(llrs)
        self.solve(hint, lb)
        return self.solution

    cpdef setStats(self, object stats):
        self._stats = stats

    cpdef object stats(self):
        return self._stats

    def fix(self, index, value):
        """Fix codeword variable with index `index` to `value` :math:`\in \{0,1\}`.
        """
        raise NotImplementedError()

    def release(self, index):
        """Release the constraint on a previously fixed variable."""
        raise NotImplementedError()

    def __str__(self):
        return self.name

    def __repr__(self):
        paramString = ", ".join("{0}={1}".format(k, repr(v)) for k, v in self.params().items())
        return "{c}(code = {cr}, {p})".format(c = self.__class__.__name__,
                                              cr = repr(self.code),
                                              p = paramString)


