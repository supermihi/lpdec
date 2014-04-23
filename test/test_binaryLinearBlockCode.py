# -*- coding: utf-8 -*-
# Copyright 2014 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation

from unittest import TestCase
import numpy as np

from . import testData
from lpdec.codes import BinaryLinearBlockCode


class TestBinaryLinearBlockCode(TestCase):

    def setUp(self):
        self.smallCode = BinaryLinearBlockCode(parityCheckMatrix=testData('Alist_N8_M4.txt'))
        self.tannerCode = BinaryLinearBlockCode(parityCheckMatrix=testData('Alist_N155_M93.txt'))

    def test_parameters(self):
        self.assertEqual(self.smallCode.rate, .5)
        self.assertEqual(self.smallCode.infolength, 4)
        self.assertEqual(self.smallCode.blocklength, 8)
        self.assertEqual(self.smallCode.name, 'Alist_N8_M4.txt')

        # this parity-check matrix has two redundant rows
        self.assertAlmostEqual(self.tannerCode.rate, 0.4129032258064516)
        self.assertEqual(self.tannerCode.blocklength, 155)
        self.assertEqual(self.tannerCode.infolength, 64)
        self.assertEqual(self.tannerCode.name, 'Alist_N155_M93.txt')

    def test_encode(self):
        zero = np.zeros(self.smallCode.blocklength, dtype=np.int)
        self.assertTrue(np.all(self.smallCode.encode(zero) == 0))

        zero = np.zeros(self.tannerCode.blocklength, dtype=np.int)
        self.assertTrue(np.all(self.tannerCode.encode(zero) == 0))