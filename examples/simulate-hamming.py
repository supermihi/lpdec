#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2014-2015 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation

# convenience module: contains most of the imports that you will need
from lpdec.imports import *

code = HammingCode(4)
# alternative: code = BinaryLinearBlockCode(parityCheckMatrix='~/codes/SomeCode.alist')

# simple IP decoder
decoder1 = CplexIPDecoder(code, name='CPLEX IP Decoder')
# IP decoder with more options
decoder2 = CplexIPDecoder(code, cplexParams={'threads': 1}, name='Cplex SingleThreaded')
# advanced configuration of B&C decoder:
decoder3 = BranchAndCutDecoder(code, name='B&C Decoder',
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

# frange is like xrange but allows fractional step size
for snr in frange(1, 2, step=.5):
    # create the channel. If you provide a seed, the pseudorandom noise will be the same in
    # subsequent calls to this script
    channel = AWGNC(snr=snr, coderate=code.rate, seed=3487)
    # create a simulator for this code/channel/decoders combination
    # 'example' is an arbitrary identifier given to the computation
    simulator = Simulator(code, channel, [decoder1, decoder2, decoder3], 'example')
    # maximum number of samples and errors
    simulator.maxSamples = 100000
    simulator.maxErrors = 100
    # as again: random seed for generating random codewords
    simulator.wordSeed = 1337
    # starts simulation
    simulator.run()
