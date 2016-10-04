# -*- coding: utf-8 -*-
# Copyright 2016 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation

import unittest
from lpdec.codes import BinaryLinearBlockCode
from lpdec.codes.nonbinary import makeNonBinary
from . import testData


class TestMakeNonBinary(unittest.TestCase):

    def test_example(self):
        code = BinaryLinearBlockCode(parityCheckMatrix=testData('Alist_N155_M93.txt'))
        nb = makeNonBinary(code, 5)
        from lpdec import matrices
        print(matrices.numpyToString(nb.parityCheckMatrix))
        self.assertEquals(nb.q, 5)

