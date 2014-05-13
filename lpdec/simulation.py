# -*- coding: utf-8 -*-
# Copyright 2014 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation
from __future__ import division, print_function
from collections import OrderedDict
import datetime
import math
import numpy as np
import lpdec
from lpdec import database as db, utils
from lpdec.database import simulation as dbsim


class DataPoint:
    """Data class storing information about a single point of measurement, i.e. a certain
    combination of code, decoder, channel, and identifier.
    """
    def __init__(self, code, channel, wordSeed, decoder, identifier):
        self.code = code
        self.channel = channel
        self.wordSeed = wordSeed
        self.decoder = decoder
        self.identifier = identifier
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

    def unstoredCPUTime(self):
        return self.cputime - self._dbCputime

    def unstoredSamples(self):
        return self.samples - self._dbSamples

    def store(self):
        self.date_end = datetime.datetime.utcnow()
        self.stats = self.decoder.stats()
        dbsim.addDataPoint(self)
        self._dbSamples = self.samples
        self._dbCputime = self.cputime


TERM_BOLD_RED = '\033[31;1m'
TERM_RED = '\033[31m'
TERM_BOLD = '\033[0;1m'
TERM_NORMAL = '\033[0m'


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
        self.dataPoints = None
        # options
        self.revealSent = False
        self.checkWord = True
        self.wordSeed = None
        self.dbStoreSampleInterval = self.maxSamples
        self.dbStoreTimeInterval = 60*5  # 5 minutes
        self.outputInterval = datetime.timedelta(seconds=30)
        #  check if the code exists in the database but has different parameters. This avoids
        #  a later error which would imply a waste of time.
        if not dbsim.initialized:
            dbsim.init()
        db.checkCode(code, insert=False)

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

    def run(self):
        """Output of the information line:
        ** <code> / <channel> / <identifier>
                  <name of decoder1>|<name of decoder2>
                  <#err> errors     |<#err> errors
                  <#cputime> sec    |<#cputime> sec
        Output of a single line:
        <sample>: <objValue>        |<objValue>

        Error is formatted as integer 4-digits. Cputime as "general" with precision 4, hence the
        cputime column has width max(len(decoder.name), 9 + len(" sec") = 13)
        """
        self.dataPoints = OrderedDict()  # maps decoders to DataPoint instances
        #  check for problems with the decoders before time is spent on computations
        for decoder in self.decoders:
            db.checkDecoder(decoder, insert=False)
        outputFormat = {}
        for decoder in self.decoders:
            point = dbsim.dataPoint(self.code, self.channel, self.wordSeed, decoder,
                                    self.identifier)
            if point.samples >= self.maxSamples or point.errors >= self.maxErrors:
                continue  # point is already done
            if point.version != lpdec.__version__:
                raise RuntimeError('VERSION MISMATCH {} != {}'.format(point.version,
                                                                      lpdec.__version__))
            self.dataPoints[decoder] = point
            decoder.setStats(point.stats)
            outputFormat[decoder] = '{:<' + str(max(len(decoder.name), 13)) + 's} '
        if len(self.dataPoints) == 0:
            return
        signaller = self.channel.signalGenerator(self.code, wordSeed=self.wordSeed)
        startSample = min(point.samples for point in self.dataPoints.values()) + 1
        if startSample > 1:
            #  ensure random seed matches
            print('skipping {} frames ...'.format(startSample-1))
            signaller.skip(startSample - 1)
        lastOutput = datetime.datetime.min
        for i in xrange(startSample, self.maxSamples+1):
            channelOutput = next(signaller)
            sampleOffset = max(5, int(math.ceil(math.log10(i)))) + len(': ')
            if i == startSample or datetime.datetime.utcnow() - lastOutput > self.outputInterval:
                # print status output
                print('*** {} / {} / {} ***'.format(self.code.name, self.channel, self.identifier))
                for row in 'name', 'errors', 'seconds':
                    print(' ' * sampleOffset, end='')
                    for decoder, point in self.dataPoints.items():
                        if row == 'name':
                            string = decoder.name
                        elif row == 'errors':
                            string = '{} errors'.format(point.errors)
                        elif row == 'seconds':
                            string = '{:.4g} sec'.format(point.cputime)
                        print(outputFormat[decoder].format(string), end='')
                    print('')
                lastOutput = datetime.datetime.utcnow()
            print(('{:' + str(sampleOffset-2) + 'd}: ').format(i), end='')
            unfinishedDecoders = len(self.dataPoints)
            for decoder, point in self.dataPoints.items():
                if point.errors >= self.maxErrors or point.samples >= self.maxSamples:
                    print(outputFormat[decoder].format('finished'), end='')
                    unfinishedDecoders -= 1
                    continue
                if point.samples > i:
                    print(outputFormat[decoder].format('skip'), end='')
                    continue
                with utils.stopwatch() as timer:
                    if self.revealSent:
                        decoder.decode(channelOutput, sent=signaller.encoderOutput)
                    else:
                        decoder.decode(channelOutput)
                point.cputime += timer.duration
                point.samples += 1
                if not self.decodingCorrect(decoder, signaller):
                    point.errors += 1
                    print(TERM_BOLD_RED if decoder.mlCertificate else TERM_RED, end='')
                else:
                    print(TERM_BOLD if decoder.mlCertificate else TERM_NORMAL, end='')
                store = False
                if point.samples == self.maxSamples or point.errors == self.maxErrors:
                    store = True
                    unfinishedDecoders -= 1
                if point.unstoredSamples() >= self.dbStoreSampleInterval:
                    store = True
                if point.unstoredCPUTime() > self.dbStoreTimeInterval:
                    store = True
                if store:
                    point.store()
                #  avoid "-0" in the output
                val = 0 if abs(decoder.objectiveValue) < 1e-8 else decoder.objectiveValue
                outputString = '{:<.7f}'.format(val) + ('*' if store else '')
                print(outputFormat[decoder].format(outputString) + TERM_NORMAL, end='')
            print('')
            if unfinishedDecoders == 0:
                break
