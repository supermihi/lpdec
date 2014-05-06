#! /usr/bin/env python2
# -*- coding: utf-8 -*-
# Copyright 2014 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation

from lpdec.decoders.ip import CplexIPDecoder
from lpdec.codes.classic import HammingCode
from lpdec.channels import *
from lpdec.simulation import *

code = HammingCode(4)
decoder1 = CplexIPDecoder(code, name='Decoder 1')
decoder2 = CplexIPDecoder(code, name='Decoder 2')

for snr in [1, 1.5, 2]:
    channel = AWGNC(snr, code.rate)
    simulator = Simulator(code, channel, [decoder1, decoder2], 'example')
    simulator.maxSamples = 10000
    simulator.maxErrors = 100
    simulator.wordSeed = 1337

    simulator.run()
