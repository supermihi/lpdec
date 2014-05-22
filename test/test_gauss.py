# -*- coding: utf-8 -*-
# Copyright 2014 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation

import unittest

import numpy as np

from lpdec.mod2la import gaussianElimination
from lpdec import matrices
from . import testData


class SmallGaussTest(unittest.TestCase):
    def setUp(self):
        self.matrix = np.array([[1, 0, 0, 1, 0, 1, 1],
                                [0, 1, 0, 1, 1, 1, 0],
                                [0, 0, 1, 0, 1, 1, 1]], dtype=np.int)
        self.diagAuto = self.matrix
        self.diag456 = np.array([[0, 1, 1, 1, 0, 0, 1],
                                 [1, 1, 0, 0, 1, 0, 1],
                                 [1, 1, 1, 0, 0, 1, 0]])
        self.tridiag456 = np.array([[1, 0, 0, 1, 0, 1, 1],
                                    [1, 1, 0, 0, 1, 0, 1],
                                    [1, 1, 1, 0, 0, 1, 0]])
        self.tridiag345 = np.array([[0, 0, 1, 0, 1, 1, 1],
                                    [0, 1, 0, 1, 1, 1, 0],
                                    [1, 1, 0, 0, 1, 0, 1]])

    def test_diag123(self):
        gaussianElimination(self.matrix)
        self.assert_((self.matrix == self.diagAuto).all())

    def test_diag456(self):
        gaussianElimination(self.matrix, np.array([3, 4, 5]))
        self.assert_((self.matrix == self.diag456).all())

    def test_tridiag456(self):
        gaussianElimination(self.matrix, np.array([3, 4, 5]), diagonalize=False)
        self.assert_((self.matrix == self.tridiag456).all())

    def test_tridiag345(self):
        # involves pivoting
        gaussianElimination(self.matrix, np.array([2, 3, 4]), diagonalize=False)
        self.assert_((self.matrix == self.tridiag345).all())


class LargeGaussTest(unittest.TestCase):

    def setUp(self):
        self.matrix = matrices.getBinaryMatrix(testData('Alist_N155_M93.txt'))

    def test_gaussTanner(self):
        columns = np.arange(self.matrix.shape[1])
        ans = gaussianElimination(self.matrix, columns)
        # tanner code has 2 redundant rows
        self.assertEqual(ans.size, self.matrix.shape[0] - 2)
        # check unit columns
        for i, col in enumerate(ans):
            self.assertEqual(self.matrix[:, col].sum(), 1)
            self.assertEqual(self.matrix[i, col], 1)
