# -*- coding: utf-8 -*-
# Copyright 2014 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation
from __future__ import unicode_literals, division

from unittest import TestCase
import numpy as np

from . import testData
from lpdec.codes import BinaryLinearBlockCode


class TestBinaryLinearBlockCode(TestCase):

    def setUp(self):
        self.smallCode = BinaryLinearBlockCode(parityCheckMatrix=testData('Alist_N8_M4.txt'))
        self.tannerCode = BinaryLinearBlockCode(parityCheckMatrix=testData('Alist_N155_M93.txt'))
        self.codes = (self.smallCode, self.tannerCode)

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
        for code in self.codes:
            zero = np.zeros(code.infolength, dtype=np.int)
            self.assertTrue(np.all(code.encode(zero) == 0))
            for trial in range(20):
                info = np.random.random_integers(0, 1, code.infolength)
                codeword = code.encode(info)
                self.assertTrue(codeword in code)

    def test_generatorMatrix(self):
        for code in self.codes:
            generator = code.generatorMatrix
            self.assertEqual(generator.shape, (code.infolength, code.blocklength))
            self.assertEqual(np.linalg.norm(code.parityCheckMatrix.dot(generator.T) % 2), 0)

    def test_Persistence(self):
        for code in self.codes:
            deserialized = BinaryLinearBlockCode.fromJSON(code.toJSON())
            self.assertEqual(code, deserialized)
            deserialized.parityCheckMatrix[0, 0] = 1 - deserialized.parityCheckMatrix[0, 0]
            self.assertNotEqual(code, deserialized)