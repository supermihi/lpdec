#! /usr/bin/python
# -*- coding: utf-8 -*-
# Copyright 2015 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation

# this script generates all LTE turbo codes and stores their matrices as bzip2'ed text files
from lpdec.codes.turbolike import LTETurboCode
from lpdec.codes.interleaver import LTEInterleaver
from lpdec.matrices import formatMatrix

for length in sorted(LTEInterleaver.availableBlocklengths()):
    print('computing {}'.format(length))
    code = LTETurboCode(length)
    formatMatrix(code.parityCheckMatrix, filename='LTE_TC_N{}_K{}_y.bz2'.format(code.blocklength, code.infolength))