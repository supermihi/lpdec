# -*- coding: utf-8 -*-
# Copyright 2014 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation

from unittest import TestCase
import numpy as np
from lpdec.mod2la import orthogonalComplement


class TestOrthogonalComplement(TestCase):

    def test_orthogonalComplement(self):
        """Tests :func:`orthogonalComplement` by """
        for trial in range(100):
            cols = np.random.randint(5, 200)
            rows = np.random.randint(2, cols-1)
            matrix = np.random.random_integers(0, 1, size=(rows, cols))
            complement = orthogonalComplement(matrix)
            forward = matrix.dot(complement.T) % 2
            backward = complement.dot(matrix.T) % 2
            self.assertTrue(np.all(forward == 0))
            self.assertTrue(np.all(backward == 0))
