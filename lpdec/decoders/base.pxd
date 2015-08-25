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
    cdef public double[::1] llrs
    cdef public np.int_t[::1] sent
    cdef public np.int_t[::1] hint
    cdef public double[::1] solution
    cdef public double objectiveValue
    cdef public object code
    cdef public bint mlCertificate, foundCodeword
    cdef public object _stats
    cdef public int status
    cpdef setLLRs(self, double[::1] llrs, np.int_t[::1] sent=?)
    cpdef solve(self, double lb=?, double ub=?)
    cpdef fix(self, int index, int value)
    cpdef release(self, int index)
    cpdef fixed(self, int index)
