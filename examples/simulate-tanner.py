#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2015 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation

# convenience module: contains most of the imports that you will need
from lpdec.imports import *

code = BinaryLinearBlockCode(parityCheckMatrix='../test/data/Alist_N155_M93.txt')

decoders = []
# simple IP decoder using Gurobi
# decoders.append(GurobiIPDecoder(code))

# branch-and-cut decoder using simple branching (MostFractional)
decoders.append(BranchAndCutDecoder(code, name='BC1', selectionMethod='mixed50/2.0',
                        childOrder='llr',
                        lpClass=AdaptiveLPDecoderGurobi,
                        lpParams=dict(removeInactive=100, keepCuts=True, maxRPCrounds=20, minCutoff=.5),
                        iterParams=dict(iterations=100, reencodeOrder=2, reencodeIfCodeword=False),
                        branchClass='MostFractional'))

# branch-and-cut decoder using reliability branchung
decoders.append(BranchAndCutDecoder(code, lpClass=AdaptiveLPDecoder, name='BC2',
                                    selectionMethod='bbs',
                                    branchClass='ReliabilityBranching'))

simulation.ALLOW_DIRTY_VERSION = True
# frange is like range but allows fractional step size
for snr in frange(1, 2.1, step=.5):
    # create the channel. If you provide a seed, the pseudorandom noise will be the same in
    # subsequent calls to this script
    channel = AWGNC(snr=snr, coderate=code.rate, seed=3487)
    # create a simulator for this code/channel/decoders combination
    # 'example' is an arbitrary identifier given to the computation
    simulator = Simulator(code, channel, decoders[-1:], 'example simulation')
    # maximum number of samples and errors
    simulator.maxSamples = 100000
    simulator.maxErrors = 100
    # as again: random seed for generating random codewords
    simulator.wordSeed = 1337
    # starts simulation
    simulator.run()
