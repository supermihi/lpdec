#!/usr/bin/python2.7
# -*- coding: utf-8 -*-
# Copyright 2014 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation

import unittest
import os.path
from collections import defaultdict, OrderedDict
from lpdec.codes import BinaryLinearBlockCode
from lpdec.channels import *
from lpdec.decoders.ip import CplexIPDecoder

import numpy as np
from . import testData


class TestCplexIPDecoder(unittest.TestCase):
    """Run various test with the (23, 12) Golay code."""
    
    def setUp(self):
        self.code = BinaryLinearBlockCode(parityCheckMatrix=testData('Alist_N23_M11.txt'))
    
    def test_minDistance(self):
        """Test if the minimum distance computation works."""
        try:
            import cplex
        except ImportError:
            self.skipTest('CPLEX is not installed')
        self.decoder = CplexIPDecoder(self.code)
        distance, codeword = self.decoder.minimumDistance()
        self.assertEqual(distance, 7)
        self.assertEqual(codeword.sum(), 7)
        self.assertIsInstance(codeword, np.ndarray)

    def test_decoding(self):
        try:
            import cplex
        except ImportError:
            self.skipTest('CPLEX is not installed')
        seed = 3498543
        for snr in [0, 2, 4]:
            channelRC = AWGNC(snr, self.code.rate, seed=seed)
            channelZC = AWGNC(snr, self.code.rate, seed=seed)
            decoder = CplexIPDecoder(self.code)
            sigRC = channelRC.signalGenerator(self.code, wordSeed=seed)
            sigZC = channelZC.signalGenerator(self.code, wordSeed=-1)
            for i in range(10):
                llrRC = next(sigRC)
                llrZC = next(sigZC)
                for useHint in True, False:
                    if useHint:
                        hintRC = sigRC.encoderOutput
                        hintZC = sigZC.encoderOutput
                    else:
                        hintRC = hintZC = None
                    outputRC = decoder.decode(llrRC, hint=hintRC)
                    objRC = decoder.objectiveValue
                    strikedRC = decoder.callback.occured
                    if useHint:
                        self.assertNotEqual(strikedRC, decoder.mlCertificate)
                    outputZC = decoder.decode(llrZC, hint=hintZC)
                    objZC = decoder.objectiveValue
                    strikedZC = decoder.callback.occured
                    if useHint:
                        self.assertNotEqual(strikedZC, decoder.mlCertificate)
                    errorRC = not np.allclose(outputRC, sigRC.encoderOutput)
                    errorZC = not np.allclose(outputZC, sigZC.encoderOutput)
                    self.assertEqual(errorRC, errorZC)
                    if not useHint or (not strikedRC and not strikedZC):
                        self.assertTrue(np.allclose(objRC, objZC + sigRC.correctObjectiveValue()))
