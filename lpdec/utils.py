# -*- coding: utf-8 -*-
# Copyright 2014 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation
from __future__ import print_function, unicode_literals
from contextlib import contextmanager
import datetime
import os
from dateutil import tz


class Timer(object):
    """Data class that stores timing information. Has the attributes :attr:`start`, :attr:`end`
    and :attr:`duration`.
    """
    def __init__(self, start):
        self.start = start
        self.end = None

    @property
    def duration(self):
        return None if self.end is None else self.end - self.start


@contextmanager
def stopwatch():
    """Context manager to measure user time of python code plus called subprocesses.

    Use like this:
    >>> with stopwatch() as timer:
    >>>     x = 1 + 2
    >>> cpuTime = timer.duration
    """
    tmp = os.times()
    timer = Timer(tmp[0] + tmp[2])
    yield timer
    tmp = os.times()
    timer.end = tmp[0] + tmp[2]


def utcnow():
    return datetime.datetime.now(tz.tzutc())


# terminal color codes
TERM_BOLD_RED = '\033[31;1m'
TERM_RED = '\033[31m'
TERM_BOLD = '\033[0;1m'
TERM_NORMAL = '\033[0m'