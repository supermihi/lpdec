# -*- coding: utf-8 -*-
# Copyright 2014-2015 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation
cimport numpy as np
cpdef gaussianElimination(np.int_t[:,:] matrix, Py_ssize_t[:] columns=?, bint diagonalize=?)