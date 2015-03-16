# -*- coding: utf-8 -*-
# Copyright 2014-2015 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation
from __future__ import division

import unittest
from lpdec import channels
import numpy as np


class TestBSC(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.inputSeed = np.random.randint(0, int(1e8))
        cls.channelSeed = np.random.randint(0, int(1e8))

    def setUp(self):
        np.random.seed(self.inputSeed)
        self.p = .5*np.random.random()
        self.channel = channels.BSC(self.p, self.channelSeed)

    def testRandomBSC(self):
        TRIALS = 10000
        totals = flips = 0
        for i in range(TRIALS):
            # generate random BPSK vector of random size between 10 and 100
            input = np.random.randint(0, 2, np.random.randint(10, 100))
            totals += input.size
            output = self.channel(input)
            self.assertEqual(input.shape, output.shape)
            flips += np.sum(1-2*input != np.sign(output))
        self.assertAlmostEqual(self.p, flips / totals, 2)

    def testAllzeroBSC(self):
        TRIALS = 10000
        flips = 0
        totals = 0
        for i in range(TRIALS):
            input = np.zeros(np.random.randint(10, 100))  # generate random-length zero-codeword
            totals += input.size
            output = self.channel(input)
            self.assertEqual(input.shape, output.shape)
            flips += np.sum(output < 0)
        self.assertAlmostEqual(self.p, flips / totals, 2)

    def testSeed(self):
        channel2 = channels.BSC(self.p, self.channelSeed)
        input = np.ones(100)
        for _ in range(100):
            self.assertTrue(np.all(self.channel(input) == channel2(input)))
        self.channel.resetSeed()
        channel2.resetSeed()
        for _ in range(100):
            self.assertTrue(np.all(self.channel(input) == channel2(input)))
