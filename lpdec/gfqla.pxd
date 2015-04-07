# -*- coding: utf-8 -*-
# Copyright 2014-2015 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation
cimport numpy as np

cpdef gaussianElimination(np.int_t[:,::1] matrix, np.intp_t[:] columns=?, bint diagonalize=?,
                          np.intp_t[::1] successfulCols=?, int q=?)

cpdef inKernel(np.int_t[:, :] matrix, np.int_t[:] vector, int q=?)