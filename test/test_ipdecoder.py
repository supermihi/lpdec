# -*- coding: utf-8 -*-
# Copyright 2014-2015 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation

import unittest
import numpy as np

from lpdec.codes import BinaryLinearBlockCode
from lpdec.channels import *
from lpdec.codes.classic import HammingCode
from lpdec.decoders.ip import CplexIPDecoder, GurobiIPDecoder
from lpdec.decoders.adaptivelp_glpk import AdaptiveLPDecoder
from lpdec.decoders.branchcut import BranchAndCutDecoder
from lpdec.persistence import JSONDecodable
from . import testData
from test import requireCPLEX


class TestMLDecoders:

    """Compare all available ML decoders and check that they yield the same results."""

    def createDecoders(self, code):
        decoders = []
        try:
            import cplex
            decoders.append(CplexIPDecoder(code))
        except:
            pass

        import gurobimh
        from lpdec.decoders.adaptivelp_gurobi import AdaptiveLPDecoderGurobi
        decoders.append(GurobiIPDecoder(code))

        decoders.append(BranchAndCutDecoder(code, name='BC1', selectionMethod='mixed50/2.0',
                        childOrder='llr',
                        lpClass=AdaptiveLPDecoderGurobi,
                        lpParams=dict(removeInactive=100, keepCuts=True, maxRPCrounds=20, minCutoff=.5),
                        iterParams=dict(iterations=100, reencodeOrder=2, reencodeIfCodeword=False),
                        branchClass='MostFractional'))

        decoders.append(BranchAndCutDecoder(code, lpClass=AdaptiveLPDecoder,
                                            name='BC2',
                                            selectionMethod='bbs',
                                            branchClass='ReliabilityBranching'))

        return decoders

    def codes(self):
        yield BinaryLinearBlockCode(parityCheckMatrix=testData('Alist_N23_M11.txt')), 7
        yield BinaryLinearBlockCode(parityCheckMatrix=testData('Alist_N155_M93.txt')), 20
        yield HammingCode(4), 3

    def computeDmin(self, decoder, asserted):
        baseMsg = 'Error calculating minimum distance of code {} using {}: '.format(
                decoder.code.name, decoder.name)
        dmin = decoder.minimumDistance()
        assert dmin == asserted, baseMsg + '{} (calc) != {} (corr)'.format(dmin, asserted)
        assert decoder.solution in decoder.code
        assert np.sum(decoder.solution) == asserted

    def test_minDistance(self):
        for code, dmin in self.codes():
            if dmin > 10:
                continue # skip dmin for too large codes
            for decoder in self.createDecoders(code):
                print('dminning {} with {}'.format(code, decoder))
                yield self.computeDmin, decoder, dmin

    def compareDecoders(self, decoders, snr):
        seed = 340987
        code = decoders[0].code
        channel = AWGNC(snr, code.rate, seed=seed)
        sig = channel.signalGenerator(code, wordSeed=seed)
        for i in range(25):
            llr = next(sig)
            refOutput = None
            refObjVal = None
            refError = None
            for useHint in False, True:
                hint = sig.codeword if useHint else None
                for decoder in decoders:
                    print('decoding {} {} {}'.format(i, useHint, decoder))
                    solution = decoder.decode(llr, sent=hint)
                    error = not np.allclose(solution, sig.codeword)
                    if refOutput is None:
                        refOutput = solution.copy()
                        refObjVal = decoder.objectiveValue
                        refError = error
                    else:
                        assert refError == error
                        if not useHint:
                            assert np.allclose(refObjVal, decoder.objectiveValue)
                            assert np.allclose(refOutput, solution)


    def test_decoding(self):
        for code, _ in self.codes():
            decoders = self.createDecoders(code)
            snrs = [1, 3, 5]
            if code.blocklength == 155:
                # tanner code takes too long for best-bound and/or snr=1
                decoders = decoders[:-1]
                snrs = snrs[1:]
            for snr in snrs:
                yield self.compareDecoders, decoders, snr


class TestCplexIPDecoder(unittest.TestCase):
    """Run various test with the (23, 12) Golay code."""

    def setUp(self):
        self.code = BinaryLinearBlockCode(parityCheckMatrix=testData('Alist_N23_M11.txt'))

    @requireCPLEX
    def test_minDistance(self):
        """Test if the minimum distance computation works."""
        self.decoder = CplexIPDecoder(self.code)
        distance = self.decoder.minimumDistance()
        codeword = self.decoder.solution
        self.assertEqual(distance, 7)
        self.assertEqual(np.sum(codeword), 7)

    @requireCPLEX
    def test_decoding(self):
        seed = 3498543
        for snr in [0, 2, 4]:
            channelRC = AWGNC(snr, self.code.rate, seed=seed)
            channelZC = AWGNC(snr, self.code.rate, seed=seed)
            decoder = CplexIPDecoder(self.code, cplexParams={'threads': 1})
            sigRC = channelRC.signalGenerator(self.code, wordSeed=seed)
            sigZC = channelZC.signalGenerator(self.code, wordSeed=-1)
            for i in range(10):
                llrRC = next(sigRC)
                llrZC = next(sigZC)
                for useHint in True, False:
                    if useHint:
                        hintRC = sigRC.codeword
                        hintZC = sigZC.codeword
                    else:
                        hintRC = hintZC = None
                    outputRC = decoder.decode(llrRC, sent=hintRC)
                    objRC = decoder.objectiveValue
                    strikedRC = decoder.callback.occured
                    if useHint:
                        self.assertNotEqual(strikedRC, decoder.mlCertificate)
                    outputZC = decoder.decode(llrZC, sent=hintZC)
                    objZC = decoder.objectiveValue
                    strikedZC = decoder.callback.occured
                    if useHint:
                        self.assertNotEqual(strikedZC, decoder.mlCertificate)
                    errorRC = not np.allclose(outputRC, sigRC.codeword)
                    errorZC = not np.allclose(outputZC, sigZC.codeword)
                    self.assertEqual(errorRC, errorZC)
                    if not useHint or (not strikedRC and not strikedZC):
                        self.assertTrue(np.allclose(objRC, objZC + sigRC.correctObjectiveValue()))


class TestCplexIPPersistence(unittest.TestCase):

    def setUp(self):
        self.code = HammingCode(4)

    @requireCPLEX
    def testDefault(self):
        decoders = [CplexIPDecoder(self.code),
                    CplexIPDecoder(self.code, name='OtherDecoder'),
                    CplexIPDecoder(self.code, cplexParams=dict(threads=1))]
        for decoder in decoders:
            self.assertEqual(len(decoders[0].cplex.parameters.get_changed()), 0)
            parms = decoder.toJSON()

            reloaded = JSONDecodable.fromJSON(parms, code=self.code)
            self.assertEqual(decoder, reloaded)

            def reprParms(cpx):
                return [(repr(x), y) for (x, y) in cpx.parameters.get_changed()]
            self.assertEqual(reprParms(decoder.cplex),
                             reprParms(reloaded.cplex))