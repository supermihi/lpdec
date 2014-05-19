# -*- coding: utf-8 -*-
# Copyright 2014 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation

import unittest

import numpy as np

from lpdec.codes import BinaryLinearBlockCode
from lpdec.channels import *
from lpdec.decoders.iterative import IterativeDecoder
from . import testData


class TestIterativeDecoder(unittest.TestCase):
    
    def setUp(self):
        self.code = BinaryLinearBlockCode(parityCheckMatrix=testData('Alist_N155_M93.txt'))


    def test_decoding(self):
        for snr in [1, 2, 3]:
            channel = AWGNC(snr, self.code.rate, seed=100)
            decoders = [IterativeDecoder(self.code, reencodeOrder=i, reencodeRange=1)
                        for i in [-1, 0, 1, 2]]
            sig = channel.signalGenerator(self.code)
            errors = {decoder: 0 for decoder in decoders}
            for i in range(100):
                llr = next(sig)
                for decoder in decoders:
                    solution = decoder.decode(llr)
                    if not np.allclose(solution, sig.encoderOutput):
                        errors[decoder] += 1
            for i in range(len(decoders) - 1):
                print(errors[decoders[i]], errors[decoders[i+1]], snr)
                self.assertGreaterEqual(errors[decoders[i]], errors[decoders[i+1]])