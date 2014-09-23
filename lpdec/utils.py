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
import platform
from dateutil import tz


def clock():
    tmp = os.times()
    return tmp[0] + tmp[2]


class Timer(object):
    """Class for time measurement. Has the attributes :attr:`startTime`, :attr:`endTime` and
    :attr:`duration`. Can be used as context manager like this::

        >>> with Timer() as timer:
        >>>     x = 1 + 2
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
    """Generalization of the built-in :func:`xrange` function that supports fractional step
    sizes.
    """
    current = start
    while current < end:
        yield current
        current += step


def machineString():
    """A string identifying the current machine, composed of the host name and platform
    information.
    :rtype: unicode
    """
    return '{0} ({1})'.format(platform.node(), platform.platform())


# terminal color codes
TERM_BOLD_RED = '\033[31;1m'
TERM_RED = '\033[31m'
TERM_BOLD = '\033[0;1m'
TERM_NORMAL = '\033[0m'