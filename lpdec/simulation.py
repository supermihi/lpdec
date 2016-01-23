# -*- coding: utf-8 -*-
# Copyright 2014-2015 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation

from __future__ import division, print_function

from collections import OrderedDict
import datetime
import math
import threading
try:
    from Queue import Queue
except ImportError:
    from queue import Queue

import numpy as np
import lpdec

from lpdec import database as db, utils
from lpdec.utils import *

# global vars for debugging purposes
DEBUG_SAMPLE = None
ALLOW_DIRTY_VERSION = False
ALLOW_VERSION_MISMATCH = False


class DataPoint:
    """Data class storing information about a single point of frame-error rate measurement.

    Attributes
    ----------
    code
        The code used.
    channel
        The channel object.
    wordSeed
        The random seed used for generating random codewords.
    decoder
        The decoder instance.
    identifier
        Simulation identifier string.
    """
    def __init__(self, code, channel, wordSeed, decoder, identifier):
        self.code = code
        self.channel = channel
        self.wordSeed = wordSeed
        self.decoder = decoder
        self.identifier = identifier
        self.samples = self.errors = self.cputime = 0
        self.date_start = utcnow()
        self.date_end = None
        self.stats = {}
        self.program = 'lpdec'
        self.version = lpdec.exactVersion()
        self.machine = utils.machineString()
        # CPU time and number of samples in database. This is used to calculate when DB should be
        #  updated after a specific time / number of samples have been computed.
        self._dbCputime = self._dbSamples = 0

    @property
    def frameErrorRate(self):
        if self.samples == 0:
            return 0
        return self.errors / self.samples

    @property
    def avgTime(self):
        if self.samples == 0:
            return 0
        return self.cputime / self.samples

    @property
    def snr(self):
        return self.channel.snr

    def unstoredCPUTime(self):
        return self.cputime - self._dbCputime

    def unstoredSamples(self):
        return self.samples - self._dbSamples

    def store(self):
        """Store the current data of this point in the database."""
        from lpdec.database import simulation as dbsim
        self.date_end = utcnow()
        self.stats = self.decoder.stats()
        dbsim.addDataPoint(self)
        self._dbSamples = self.samples
        self._dbCputime = self.cputime

    def checkResume(self):
        """Check if computation for this code can be resumed. For this to be true, the program
        name and versions have to match."""
        # version check
        if self.program != 'lpdec':
            raise RuntimeError('Program name mismatch: "{}" != "lpdec"'.format(self.program))
        if not ALLOW_VERSION_MISMATCH and self.version != lpdec.exactVersion():
            raise RuntimeError('Version mismatch: "{}" != "{}"'.format(self.version,
                                                                       lpdec.exactVersion()))

        if not ALLOW_DIRTY_VERSION and lpdec.exactVersion().endswith('dirty'):
            raise RuntimeError('Dirty program version {}'.format(lpdec.exactVersion()))


class Simulation(list):
    """Data class to encapsulate the information about one "Simulation", i.e., frame-error rates
    for a specific tuple of (code, decoder, channel type, word seed, identifier) run for different
    SNR values.

    The class subclasses :class:`list`, the entries are :class:`.DataPoint` instances ordered by
    SNR value.
    """
    def __init__(self, points=None):
        list.__init__(self)
        if points:
            self.extend(sorted(points, key=lambda point: point.channel.snr))

    def minSNR(self):
        """Minimum SNR among all points in this run."""
        return self[0].channel.snr

    def maxSNR(self):
        """Maximum SNR among all points in this run."""
        return self[-1].channel.snr

    @property
    def code(self):
        return self[0].code

    @property
    def decoder(self):
        return self[0].decoder

    @property
    def identifier(self):
        return self[0].identifier

    @property
    def channelClass(self):
        return type(self[0].channel)

    @property
    def wordSeed(self):
        return self[0].wordSeed

    @property
    def date_start(self):
        """Return the earliest computation start of the run."""
        return min(point.date_start for point in self)

    @property
    def date_end(self):
        """Return the latest computation end of the run."""
        return max(point.date_end for point in self)

    @property
    def version(self):
        return self[0].version

    @property
    def program(self):
        return self[0].program

    def add(self, newPoint):
        """Add `newPoint` to the simulation run. The point is inserted in such a way that sorted
        SNR values are maintained."""
        for i, point in enumerate(self):
            if point.snr >= newPoint.snr:
                assert newPoint.snr != point.snr
                self.insert(i, newPoint)
                break
        else:
            self.append(newPoint)


class Simulator(object):
    """A Simulator computes frame error rates for a code / channel combination with different
    decoders by monte-carlo simulations.

    Simulation is started by calling the :meth:`.run` method.

    Parameters
    ----------
    code : BinaryLinearBlockCode
        The code for which to run simulations.
    channel : :class:`.Channel`
        The channel to use for simulations.
    decoders : list of :class:`.Decoder`
        List of decoder instances for which to run the simulation.
    identifier: str
        An identifier string describing this simulation.



    Attributes
    ----------
    maxSamples : int
        Maximum number of samples to simulate. (default: 100000)
    maxErrors : int
        Maximum number of decoding errors. After that number is reached, simulation is stopped
        (independently of the :attr:`maxSamples` value). Default: 100.
    revealSent : bool
        Whether to reveal the sent codeword to the decoder. Might be used to speed up decoding for
        simulation purposes. Default: False.
    dbStoreSampleInterval : int
        Number of decoded frames after which the current results are stored to the database.
        Defaults to :attr:`maxSamples`.
    dbStoreTimeInterval : float
        Interval (in seconds) for storing results to database. Default: 300 (5 minutes).
    outputInterval : float
        Interval (in seconds) for statistics output (number of errors, decoding time, ...) for each
        decoder. Default: 30.
    verbose : bool
        Whether to output objective value for each decoded frame. Default: True.
    conurrent : False
        Allow concurrent decoding in case of more than one decoder. Might be buggy, use with care.
        Default: False.
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
        self.wordSeed = None
        self.dbStoreSampleInterval = self.maxSamples
        self.dbStoreTimeInterval = 60 * 5  # 5 minutes
        self.outputInterval = 30
        self.verbose = True
        self.concurrent = False
        self.allDecodersShouldBeEqual = False
        #  check if the code exists in the database but has different parameters. This avoids
        #  a later error which would imply a waste of time.
        from lpdec.database import simulation as dbsim
        if not db.initialized:
            db.init()
        if not dbsim.initialized:
            dbsim.init()
        db.checkCode(code, insert=False)

    def run(self):
        """Starts the simulation with configured parameters."""
        def prv(*args, **kwargs):  # print function conditioned on self.verbose
            if self.verbose:
                print(*args, **kwargs)
        timer = Timer()
        from lpdec.database import simulation as dbsim
        self.dataPoints = OrderedDict()  # maps decoders to DataPoint instances
        #  check for problems with the decoders before time is spent on computations
        global DEBUG_SAMPLE
        if DEBUG_SAMPLE is None:
            for decoder in self.decoders:
                db.checkDecoder(decoder, insert=False)
        outputFormat = {}
        for decoder in self.decoders:
            if DEBUG_SAMPLE is None:
                point = dbsim.dataPoint(self.code, self.channel, self.wordSeed, decoder,
                                        self.identifier)
                if point.samples >= self.maxSamples or point.errors >= self.maxErrors:
                    continue
                point.checkResume()
            else:
                point = DataPoint(self.code, self.channel, self.wordSeed, decoder, self.identifier)
            self.dataPoints[decoder] = point
            decoder.setStats(point.stats)
            outputFormat[decoder] = '{:<' + str(max(len(decoder.name), 13)) + 's} '
        if len(self.dataPoints) == 0:
            return
        signaller = self.channel.signalGenerator(self.code, wordSeed=self.wordSeed)
        startSample = min(point.samples for point in self.dataPoints.values()) + 1
        if DEBUG_SAMPLE is not None:
            if isinstance(DEBUG_SAMPLE, int):
                DEBUG_SAMPLE = (DEBUG_SAMPLE, )
            DEBUG_SAMPLE = sorted(DEBUG_SAMPLE)
            print('**** DEBUG MODE: running samples {} only ****'.format(DEBUG_SAMPLE))
            startSample = self.maxSamples = DEBUG_SAMPLE[0]
            self.maxSamples = DEBUG_SAMPLE[-1]
        if startSample > 1:
            #  ensure random seed matches
            print('skipping {} frames ...'.format(startSample-1))
            signaller.skip(startSample - 1)
        lastOutput = datetime.datetime.min

        def printStatus():
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
            print('*** {} / {} / {} ***'.format(self.code.name, self.channel, self.identifier))
            for row in 'name', 'errors', 'seconds':
                print(' ' * sampleOffset, end='')
                for decoder, point in self.dataPoints.items():
                    if row == 'name':
                        string = decoder.name
                    elif row == 'errors':
                        string = '{} errors'.format(point.errors)
                    else:  # row == 'seconds':
                        string = '{:.4g} sec'.format(point.cputime)
                    print(outputFormat[decoder].format(string), end='')
                print('')
        if len(self.decoders) == 1:
            self.concurrent = False
        if self.concurrent:
            threads = {decoder: DecodingThread(decoder, self.revealSent)
                       for decoder in self.decoders}

        for i in range(startSample, self.maxSamples+1):
            channelOutput = next(signaller)
            if DEBUG_SAMPLE and i not in DEBUG_SAMPLE:
                continue
            sampleOffset = max(5, int(math.ceil(math.log10(i)))) + len(': ')
            if i == startSample or (utcnow() - lastOutput).total_seconds() > self.outputInterval:
                printStatus()
                lastOutput = utcnow()
            prv(('{:' + str(sampleOffset-2) + 'd}: ').format(i), end='')
            unfinishedDecoders = len(self.dataPoints)
            # go through decoders. For those not yet finished, start decoding or, in concurrent
            # mode, place item in the queue
            for decoder, point in self.dataPoints.items():
                if point.errors >= self.maxErrors or point.samples >= self.maxSamples or point.samples >= i:
                    continue
                if self.concurrent:
                    threads[decoder].jobQueue.put((channelOutput, signaller.codeword))
                else:
                    with timer:
                        if self.revealSent:
                            decoder.decode(channelOutput, sent=signaller.codeword)
                        else:
                            decoder.decode(channelOutput)
                    point.cputime += timer.duration
            if self.allDecodersShouldBeEqual:
                solution = None
                firstDecoder = None
                for decoder in self.dataPoints:
                    if solution is None:
                        solution = decoder.solution
                        firstDecoder = decoder
                    else:
                        if not np.allclose(decoder.solution, solution):
                            print('solution of {}:\n{}'.format(firstDecoder, solution))
                            print('solution of {}:\n{}'.format(decoder, decoder.solution))
                            print('objective values: {} vs {}'.format(firstDecoder.objectiveValue,
                                                                      decoder.objectiveValue))
                            raise AssertionError()
            # go through decoders again. Print output for each decoder. In concurrent mode,
            # join all the job queues.
            for decoder, point in self.dataPoints.items():
                if point.errors >= self.maxErrors or point.samples >= self.maxSamples:
                    prv(outputFormat[decoder].format('finished'), end='')
                    unfinishedDecoders -= 1
                    continue
                if point.samples >= i:
                    prv(outputFormat[decoder].format('skip {}'.format(point.samples)), end='')
                    continue
                if self.concurrent:
                    thread = threads[decoder]
                    thread.jobQueue.join()
                    point.cputime += thread.time
                error = not np.allclose(decoder.solution, signaller.codeword, 1e-7)
                obj = decoder.objectiveValue
                ml = decoder.mlCertificate
                correctObj = signaller.correctObjectiveValue()
                if ml and obj > correctObj + 1e-4 and correctObj / obj - 1 > 1e-5:
                    raise AssertionError('Decoder {} reports ML certificate but has larger '
                                         'objective value ({}) than sent codeword ({})!'
                                         .format(decoder.name, obj, correctObj))
                point.samples += 1
                if error:
                    point.errors += 1
                    prv(TERM_BOLD_RED if ml else TERM_RED, end='')
                else:
                    prv(TERM_BOLD if ml else TERM_NORMAL, end='')
                store = False
                if point.samples == self.maxSamples or point.errors == self.maxErrors:
                    store = True
                    unfinishedDecoders -= 1
                if point.unstoredSamples() >= self.dbStoreSampleInterval:
                    store = True
                if point.unstoredCPUTime() > self.dbStoreTimeInterval:
                    store = True
                if DEBUG_SAMPLE:
                    store = False  # avoid storing results in debug mode
                if store:
                    point.store()
                #  avoid "-0" in the output
                val = 0 if abs(obj) < 1e-8 else obj
                outputString = '{:<.7f}'.format(val) + ('*' if store else '')
                prv(outputFormat[decoder].format(outputString) + TERM_NORMAL, end='')
            prv(' {}{:6f}{}'.format(TERM_CYAN, signaller.correctObjectiveValue(), TERM_NORMAL))
            if unfinishedDecoders == 0:
                break
        printStatus()


class DecodingThread(threading.Thread):
    """Thread for concurrent simulation.

    A :class:`DecodingThread` is responsible for one specific decoder. As soon as an item is
    placed on the :attr:`jobQueue`, decoding starts. After finishing, the attributes
    :attr:`time`, :attr:`error`, :attr:`objVal` and :attr:`mlCertificate` contain information
    about the solution.

    :param decoder: The :class:`.Decoder` used for this process.
    :param revealSent: If decoding should reveal the sent codeword.

    .. attribute:: jobQueue

      On this queue, pairs (llr, sentCodeword) are put. The process will start decoding
      immediately, and signal :func:`JoinableQueue.task_done` when finished."""

    def __init__(self, decoder, revealSent):
        threading.Thread.__init__(self)
        self.decoder = decoder
        self.jobQueue = Queue()
        self.daemon = True
        self.revealSent = revealSent
        self.time = 0.0
        self.start()

    def run(self):
        while True:
            llr, sent = self.jobQueue.get()
            with Timer() as timer:
                if self.revealSent:
                    self.decoder.decode(llr, sent=sent)
                else:
                    self.decoder.decode(llr)
            self.time = timer.duration
            self.jobQueue.task_done()
