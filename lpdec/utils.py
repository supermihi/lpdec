# -*- coding: utf-8 -*-
# Copyright 2014-2015 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation
from __future__ import print_function, unicode_literals
from contextlib import contextmanager
import datetime
import os, sys, platform
from dateutil import tz


def clock():
    tmp = os.times()
    return tmp[0] + tmp[2]

if sys.version_info >= (3, 3):
    import time
    def clock():
        return time.clock_gettime(time.CLOCK_THREAD_CPUTIME_ID)


class Timer:
    """Class for time measurement that can be used as a context manager (see example below).

    Attributes
    ----------
    startTime : float
        Time (CPU clock time) the timer was started.
    endTime : float
        Time (CPU clock time) the timer was ended.
    duration : float
        Duration (in seconds) the timer was running.

    Examples
    --------
        >>> with Timer() as timer:
        ...     x = sum(range(100000))
        >>> cpuTime = timer.duration
    """
    def __init__(self):
        self.startTime = self.endTime = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.stop()

    @property
    def duration(self):
        return None if self.endTime is None else self.endTime - self.startTime

    def start(self):
        self.startTime = clock()

    def stop(self):
        self.endTime = clock()
        return self.duration


def utcnow():
    """Returns a timezone-aware datetime object representing the current date and time in UTC."""
    return datetime.datetime.now(tz.tzutc())


def frange(start, end, step=1):
    """Generalization of the built-in :func:`range` function that supports fractional step
    sizes.
    """
    current = start
    while current < end:
        yield current
        current += step


def splitRanges(rangesString):
    """Split a string of type '1-3 9 12' into a list of ints [1,2,3,9,12]."""
    nums = []
    singleRangeStrings = rangesString.split()
    for part in singleRangeStrings:
        try:
            lower, upper = map(int, part.split('-'))
            nums.extend(range(lower, upper + 1))
        except ValueError:
            nums.append(int(part))
    return nums


def machineString():
    """A string identifying the current machine, composed of the host name and platform
    information.
    """
    return '{} ({})'.format(platform.node(), platform.platform())


def isStr(arg):
    """Python version-agnostic test whether `arg` is of a string type (:class:`str` in python 3,
    ``basestring`` in python 2).
    """
    if sys.version_info.major == 3:
        return isinstance(arg, str)
    else:
        return isinstance(arg, basestring)


# terminal color codes
TERM_BOLD_RED = '\033[31;1m'
TERM_RED = '\033[31m'
TERM_BOLD = '\033[0;1m'
TERM_NORMAL = '\033[0m'
TERM_CYAN = '\033[36m'