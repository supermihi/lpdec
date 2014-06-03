# -*- coding: utf-8 -*-
# Copyright 2014 Michael Helmling
# cython: embedsignature=True
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation

cdef class Node:
    cdef public int    branchIndex
    cdef public int    branchValue
    cdef public double lb
    cdef public Node   parent
    cdef public int    depth
    cdef double lbChild0, lbChild1

    cpdef updateBound(self, double lbChild, int childValue)