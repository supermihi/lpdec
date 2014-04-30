# -*- coding: utf-8 -*-
# Copyright 2014 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation

"""This module contains helpers to simplify and unify usage of the CPLEX python interface."""

from __future__ import print_function
from collections import OrderedDict
import sys
import cplex


def getInstance(**params):
    """Create and return a :class:`cplex.Cplex` instance with disabled debugging output. Keyword
    args are passed to :func:`setCplexParams`.
    """
    cpx = cplex.Cplex()
    cpx.set_results_stream(None)
    cpx.set_warning_stream(None)
    cpx.set_error_stream(None)
    setCplexParams(cpx, **params)
    return cpx


def enableDebugOutput(cpx):
    cpx.set_results_stream(sys.stderr)
    cpx.set_warning_stream(sys.stderr)
    cpx.set_error_stream(sys.stderr)


def setCplexParams(cpx, **kwargs):
    """Helper to set CPLEX parameters.

    CPLEX parameters can be set by using their name in the python interface, excluding the
    leading ``cplex.parameters.``, as key (e.g. ``workmem``, ``mip.strategy``).
    """
    if 'version' in kwargs:
        assert cpx.get_version() == kwargs.pop('version')
    for arg, val in kwargs.items():
        parts = arg.split('.')
        param = cpx.parameters
        for part in parts:
            param = getattr(param, part)
        param.set(val)


def getCplexParams(cpx):
    """Return all non-default CPLEX parameters as (ordered) dictionary. Additionally contains the
    CPLEX version under the ``version`` key.
    """
    params = OrderedDict()
    params['version'] = cpx.get_version()
    for param, value in cpx.parameters.get_changed():
        params[repr(param)] = value
    return params


class ShortcutCallback(cplex.callbacks.MIPInfoCallback):
    """A MIP callback that aborts computation codeword with an objective value below that of
    the sent codeword is found. In that event, it is sure that the ML decoder would fail,
    even though the exact ML solution might still be a different (better-valued) codeword. If
    only the ML performance is of interest, however, this difference does not matter.

    To use the callback, install it using ::func:`cplex.Cplex.register_callback`,
    set :attr:`codewordVars` to the names of the codeword variables (for example, `['x1',
    'x2', 'x3']`. and reset :attr:`occured` to False before any call to :func:`Decoder.solve`.

    After solving, :attr:`occured` will tell if the short callback has come to effect. In that
    case, the incumbent solution and its objective value are acessible via :attr:`solution` and
    :attr:`objectiveValue`, respectively.

    The callback can also be used with random (in contrast to all-zero) codewords. In that case,
    set :attr:`realObjective` to the objective value of the correct codeword before solving.
    """

    def __init__(self, *args, **kwargs):
        cplex.callbacks.MIPInfoCallback.__init__(self, *args, **kwargs)
        self.codewordVars = self.objectiveValue = self.solution = None
        self.occured = False

    def __call__(self):
        if self.has_incumbent() and \
                self.get_incumbent_objective_value() < self.realObjective - 1e-6:
            self.occured = True
            self.objectiveValue = self.get_incumbent_objective_value()
            self.solution = self.get_incumbent_values(self.codewordVars)
            self.abort()


def checkKeyboardInterrupt(cpx):
    """Checks the solution status of the given :class:`cplex.Cplex` instance for keyboard
    interrupts, and raises a :class:`KeyboardInterrupt` exception in that case.
    """
    if cpx.solution.get_status() in (cplex._internal._constants.CPX_STAT_ABORT_USER,
                                     cplex._internal._constants.CPXMIP_ABORT_FEAS,
                                     cplex._internal._constants.CPXMIP_ABORT_INFEAS):
        raise KeyboardInterrupt()
