# -*- coding: utf-8 -*-
# cython: embedsignature=True
# Copyright 2014 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation
cimport numpy as np

cpdef gaussianElimination(np.int_t[:,:] matrix, np.int_t[:] columns=?, bint diagonalize=?)