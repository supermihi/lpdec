# -*- coding: utf-8 -*-
# Copyright 2014 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation

from collections import OrderedDict

from lpdec cimport JSONDecodable
from lpdec.codes cimport BinaryLinearBlockCode


cdef class Decoder(JSONDecodable):
    """The base class for all decoders, defining a minimal set of methods.

    The first constructor argument of all subclasses of Decoder should always be the
    code to decode. The :func:`params` method should only return the remaining arguments."""

    def __cinit__(self, BinaryLinearBlockCode code, *args, **kwargs):
        """Initialize the numpy array members with zero-length arrays."""
        self.llrs = np.zeros(code.blocklength, dtype=np.double)
        self.solution = np.zeros(code.blocklength, dtype=np.double)


    def __init__(self, code, name):
        self.code = code
        if name is None:
            raise ValueError('A decoder must have a name')
        self.name = name
        self.mlCertificate = self.foundCodeword = False
        self.setStats(OrderedDict())

    cpdef setLLRs(self, np.double_t[:] llrs):
        self.llrs = np.asarray(llrs, dtype=np.double)

    cpdef solve(self, np.int_t[:] hint=None, double lb=-np.inf, double ub=np.inf):
        """Run the solver on self.llrVector. A codeword may be given as hint.

        This is the main method to run the decoding algorithm; the LLR vector must be set
        in advance via the *llrVector* attribute.
        Some decoders may run faster if the sent codeword is known (useful for simulations),
        which can optionally be supplied using the *hint* parameter.
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

    def __str__(self):
        return self.name

    def __repr__(self):
        paramString = ", ".join("{0}={1}".format(k, repr(v)) for k, v in self.params().items())
        return "{c}(code = {cr}, {p})".format(c = self.__class__.__name__,
                                              cr = repr(self.code),
                                              p = paramString)

