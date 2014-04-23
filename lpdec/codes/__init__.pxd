# -*- coding: utf-8 -*-
# Copyright 2014 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation

from lpdec cimport JSONDecodable
cimport numpy as np

cdef class BinaryLinearBlockCode(JSONDecodable):

    cdef:
        public int blocklength, infolength
        public object filename
        public np.ndarray _generatorMatrix, _parityCheckMatrix

    cpdef np.ndarray[ndim=1, dtype=np.int_t] encode(self, np.ndarray[ndim=1, dtype=np.int_t] infoword)
