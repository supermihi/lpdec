#! /usr/bin/env python2
# -*- coding: utf-8 -*-
# Copyright 2014 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation

from lpdec.imports import *

code = HammingCode(4)
decoder1 = CplexIPDecoder(code, name='Decoder 1')
decoder2 = BranchAndCutDecoder(code, name='B&C Decoder',
                               selectionMethod='mixed-/30/100/5/2',
                               childOrder='llr',
                               lpParams=dict(removeInactive=100,
                                             insertActive=0,
                                             keepCuts=True,
                                             maxRPCrounds=100,
                                             minCutoff=.2),
                               iterParams=dict(iterations=100,
                                               reencodeOrder=2,
                                               reencodeIfCodeword=False))

for snr in frange(1, 2, step=.5):
    channel = AWGNC(snr, code.rate)
    simulator = Simulator(code, channel, [decoder1, decoder2], 'example')
    simulator.maxSamples = 10000
    simulator.maxErrors = 100
    simulator.wordSeed = 1337

    simulator.run()
