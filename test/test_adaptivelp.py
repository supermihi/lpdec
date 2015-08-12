# -*- coding: utf-8 -*-
# Copyright 2014-2015 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation

import unittest
import numpy as np
from lpdec.codes import BinaryLinearBlockCode
from lpdec.codes.classic import HammingCode
from lpdec.channels import *
from lpdec.decoders.adaptivelp_glpk import AdaptiveLPDecoder
from lpdec.decoders.adaptivelp_gurobi import AdaptiveLPDecoderGurobi
from . import testData


class TestAdaptiveLPDecoder(unittest.TestCase):

    def test_decoding(self):
        code1 = BinaryLinearBlockCode(parityCheckMatrix=testData('Alist_N23_M11.txt'))
        code2 = HammingCode(3)
        for code in code1, code2:
            channel = AWGNC(0, code.rate, seed=1337)
            decoders = [AdaptiveLPDecoder(code, maxRPCrounds=i) for i in [0, 3, -1]]
            sig = channel.signalGenerator(code, wordSeed=1337)
            errors = {decoder: 0 for decoder in decoders}
            for i in range(100):
                llr = next(sig)
                for decoder in decoders:
                    solution = decoder.decode(llr)
                    if not np.allclose(solution, sig.codeword):
                        errors[decoder] += 1
            for i in range(len(decoders) - 1):
                self.assertGreaterEqual(errors[decoders[i]], errors[decoders[i+1]])


    def test_different_classes(self):
        code1 = BinaryLinearBlockCode(parityCheckMatrix=testData('Alist_N23_M11.txt'))
        code2 = HammingCode(3)
        for code in code1, code2:
            channel = AWGNC(0, code.rate, seed=1337)
            decoders = [cls(code, maxRPCrounds=0) for cls in (AdaptiveLPDecoder, AdaptiveLPDecoderGurobi)]
            sig = channel.signalGenerator(code, wordSeed=1337)
            for i in range(1000):
                llr = next(sig)
                for decoder in decoders:
                    decoder.decode(llr)
                for decoder in decoders[1:]:
                    self.assertTrue(np.allclose(decoder.solution, decoders[0].solution))
