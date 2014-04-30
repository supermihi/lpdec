# -*- coding: utf-8 -*-
# Copyright 2014 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation
from __future__ import division, print_function
from collections import OrderedDict
import datetime
import numpy as np
import lpdec
from lpdec import database as db, channels
from lpdec.database import simulation as dbsim


class DataPoint:
    """Data class storing information about a single point of measurement, i.e. a certain
    combination of code, decoder, channel, and identifier.
    """
    def __init__(self, code, channel, decoder, identifier):
        self.code = code
        self.channel = channel
        self.decoder = decoder
        self.samples = self.errors = self.cputime = 0
        self.date_start = datetime.datetime.utcnow()
        self.date_end = None
        self.stats = {}
        self.version = lpdec.__version__
        self._dbCputime = self._dbSamples = 0

    def frameErrorRate(self):
        if self.samples == 0:
            return 0
        return self.errors / self.samples

    @property
    def snr(self):
        return self.channel.snr


class Simulator(object):
    """A Simulator computes frame error rates for a code / channel combination with different
    decoders by monte-carlo simulations.
    """
    def __init__(self, code, channel, decoders, identifier):
        self.code = code
        self.channel = channel
        self.decoders = decoders
        self.identifier = identifier
        self.maxSamples = 100000
        self.maxErrors = 100
        self.dataPoints = OrderedDict() # maps decoders to DataPoint instances

        self.useHint = False
        self.checkWord = True
        self.randomCodewords = True
        self.wordSeed = None
        self.dbStoreFrameInterval = self.maxSamples
        self.dbStoreTimeInterval = 60*5 # 5 minutes
        self.outputInterval = datetime.timedelta(seconds=30)
        #  check if the code exists in the database but has different parameters. This avoids
        #  a later error which would imply a waste of time.
        db.checkCode(code, insert=False)
        dbsim.init()

    def decodingCorrect(self, decoder, signalGenerator):
        """Helper to check for decoding error.

        Depending on :attr:`checkWord`, this either checks if the sent codeword from
        ``signalGenerator`` matches the output of the decoding algorithm, or compares the decoder's
        objective value with the scalar product of LLR vector and codeword.
        """
        if self.checkWord:
            return np.allclose(decoder.solution, signalGenerator.encoderOutput, 1e-7)
        else:
            objectiveDiff = abs(decoder.objectiveValue - signalGenerator.correctObjectiveValue())
            return objectiveDiff < 1e-8

    def simulate(self):
        #  check for problems with the decoders before time is spent on computations
        for decoder in self.decoders:
            db.checkDecoder(decoder, insert=False)

        for decoder in self.decoders:
            point = dbsim.dataPoint(self.code, self.channel, decoder, self.identifier)
            if point.samples >= self.maxSamples or point.errors >= self.maxErrors:
                continue # point exists but is already done
            if point.version != lpdec.__version__:
                raise RuntimeError('VERSION MISMATCH {} != {}'.format(point.version,
                                                                      lpdec.__version__))
            self.dataPoints[decoder] = point
            # initialize statistics from potential previous run
            decoder.setStats(point.statistics)
        if len(self.dataPoints) == 0:
            print('nothing to do')
            return
        signalGenerator = self.channel.signalGenerator(self.code, randomCodewords=self
                                                       .randomCodewords, wordSeed=self.wordSeed)
        startSample = min(point.samples for point in self.dataPoints.values()) + 1
        if startSample > 1:
            #  ensure random seed matches
            print('Skipping first {} instances which are already computed'.format(startSample-1))
            signalGenerator.skip(startSample-1)
        lastOutput = datetime.datetime.min
        for i in xrange(startSample, self.maxSamples+1):
            channelOutput = next(signalGenerator)
            if i == startSample or datetime.datetime.utcnow() - lastOutput > self.outputInterval:
                self.printStatus()
                lastOutput = datetime.datetime.utcnow()
            print("{0:5d}: ".format(i), end='')
            anyDecoderBelowMaxErrors = False
            for decoder, point in self.dataPoints.items():
                if point.errors >= self.maxErrors or point.samples >= self.maxSamples:
                    print('{:25s}'.format('finished'), end='')
                    continue
                anyDecoderBelowMaxErrors = True
                if point.samples > i:
                    print('{:25s}'.format('skipping'), end='')
                    continue
                with utils.stopwatch() as timer:
                    if self.useHint:
                        decoder.decode(channelOutput, hint=signalGenerator.encoderOutput)
                    else:
                        decoder.decode(channelOutput)
                point.cputime += timer.duration
                point.samples += 1
                if not self.decodingCorrect(decoder, signalGenerator):
                    point.errors += 1
                    if decoder.mlCertificate:
                        print('\033[31;1m', end='') # red & bold
                    else:
                        print('\033[31m', end='') # red
                else:
                    if decoder.mlCertificate:
                        print('\033[0;1m', end='') # bold
                    else:
                        print('\033[0m', end='') # color: normal
                store = False
                if point.samples == self.maxSamples:
                    store = True
                if point.errors == self.maxErrors:
                    store = True
                if point.samples - point._dbSamples >= self.dbStoreFrameInterval:
                    store = True
                if point.cputime - point._dbCputime > self.dbStoreTimeInterval:
                    store = True
                if store:
                    # add /update point to DB
                    point.date_end = datetime.datetime.utcnow()
                    point.stats = decoder.stats()
                    dbsim.addDataPoint(point)
                    print('*', end='')
                #  this hack avoids "-0" in the output
                val = 0 if abs(decoder.objectiveValue) < 1e-8 else decoder.objectiveValue
                print(("{0:<2" + ("4" if store else "5") + ".10f}").format(val), end='')
                print('\033[0m', end='')
            if not anyDecoderBelowMaxErrors:
                print('All decoders reached maxErrors.')
                break
            print()
        print('Finished channel {0}'.format(self.channel))

    def printStatus(self):
        print("code: {}, channel: {} [{}]".format(self.code, self.channel, self.identifier))
        print('       ', end='')
        #  each decoder gets a width of 25 characters in the display
        for decoder in self.dataPoints:
            print('{0:<24.24s} '.format(decoder), end='')
        print()
        print('       ', end='')
        for point in self.dataPoints.values():
            print('errors: {0:<17d}'.format(point.errors), end='')
        print()
        print('       ', end='')
        for point in self.dataPoints.values():
            print('time: {0:<19f}'.format(point.cputime), end='')
        print('\n')
