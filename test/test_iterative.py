# -*- coding: utf-8 -*-
# Copyright 2014-2015 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation

import unittest
import itertools
import numpy as np
from lpdec.codes import BinaryLinearBlockCode
from lpdec.channels import *
from lpdec.decoders.iterative import IterativeDecoder
from . import testData


class TestIterativeDecoder(unittest.TestCase):
    
    def setUp(self):
        self.code = BinaryLinearBlockCode(parityCheckMatrix=testData('BCH_127_85_6_strip.alist'))

    def test_decoding(self):
        for snr, minSum, rr in itertools.product([1, 3], [True, False], [0.1, 0.5, 1]):
            channel = AWGNC(snr, self.code.rate, seed=100)
            decoders = [IterativeDecoder(self.code, minSum=minSum, reencodeOrder=i,
                                         reencodeRange=rr)
                        for i in [-1, 0, 1, 2]]
            sig = channel.signalGenerator(self.code)
            errors = {decoder: 0 for decoder in decoders}
            for i in range(50):
                llr = next(sig)
                for decoder in decoders:
                    solution = decoder.decode(llr)
                    if not np.allclose(solution, sig.codeword):
                        errors[decoder] += 1
                    if decoder.foundCodeword:
                        self.assertNotEquals(decoder.objectiveValue, np.inf)
            for i in range(len(decoders) - 1):
                self.assertGreaterEqual(errors[decoders[i]], errors[decoders[i+1]])