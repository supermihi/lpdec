# -*- coding: utf-8 -*-
# Copyright 2014-2015 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation

from os.path import dirname, join
from functools import wraps

here = dirname(__file__)


def testData(*pathParts):
    return join(here, 'data', *pathParts)


def requireCPLEX(func):
    """Can be used as a decorator for tests requiring CPLEX.

    Will skip the test if CPLEX is not installed.
    """
    @wraps(func)
    def test_func(self, *args, **kwargs):
        try:
            import cplex
        except ImportError:
            self.skipTest('CPLEX is not installed')
        return func(self, *args, **kwargs)
    return test_func