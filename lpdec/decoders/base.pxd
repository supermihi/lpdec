# -*- coding: utf-8 -*-
# Copyright 2014-2015 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation

from lpdec.persistence cimport JSONDecodable
cimport numpy as np

cdef class Decoder(JSONDecodable):

    cdef public object name
    cdef public np.ndarray llrs
    cdef public np.int_t[::1] sent
    cdef public np.ndarray solution
    cdef public double objectiveValue
    cdef public object code
    cdef public bint mlCertificate, foundCodeword
    cdef public object _stats
    cpdef setLLRs(self, np.ndarray[ndim=1, dtype=double] llrs, np.int_t[::1] sent=?)
    cpdef solve(self, double lb=?, double ub=?)
    cpdef fix(self, int index, int value)
    cpdef release(self, int index)
