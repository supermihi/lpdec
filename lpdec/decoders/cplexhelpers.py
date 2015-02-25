# -*- coding: utf-8 -*-
# Copyright 2014-2015 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation

"""This module contains helpers to simplify and unify usage of the CPLEX python interface."""

from __future__ import print_function
from collections import OrderedDict
import sys
import numpy as np
from lpdec.decoders import Decoder


class CplexDecoder(Decoder):
    """Generic base class for CPLEX based integer programming decoders.

    .. attribute:: x

       Vector of names of the codeword variables
    """

    def __init__(self, code, name, cplexParams=None):
        Decoder.__init__(self, code, name)
        if cplexParams is None:
            cplexParams = {}
        self.cplex = self.createCplex(**cplexParams)
        self.cplex.objective.set_sense(self.cplex.objective.sense.minimize)
        self.x = ['x' + str(num) for num in range(code.blocklength)]
        self.cplex.variables.add(types=['B'] * code.blocklength, names=self.x)
        self.callback = self.cplex.register_callback(ShortcutCallback)
        self.callback.decoder = self

    @staticmethod
    def createCplex( **params):
        """Create and return a :class:`cplex.Cplex` instance with disabled debugging output. Keyword
        args are used to set parameters.

        CPLEX parameters can be set by using their name in the python interface, excluding the
        leading ``cplex.parameters.``, as key (e.g. ``workmem``, ``mip.strategy``).
        """
        import cplex
        cpx = cplex.Cplex()
        stream = None
        if params.get('debug', False):
            stream = sys.stderr
        if 'debug' in params:
            del params['debug']
        cpx.set_results_stream(stream)
        cpx.set_warning_stream(stream)
        cpx.set_error_stream(stream)
        if 'version' in params:
            assert cpx.get_version() == params.pop('version')

        for arg, val in params.items():
            parts = arg.split('.')
            param = cpx.parameters
            for part in parts:
                param = getattr(param, part)
            param.set(val)
        return cpx

    @staticmethod
    def cplexParams(cpx):
        """Return all non-default CPLEX parameters as (ordered) dictionary. Additionally contains
        the CPLEX version under the ``version`` key.
        """
        params = OrderedDict()
        params['version'] = cpx.get_version()
        for param, value in cpx.parameters.get_changed():
            params[repr(param).split('.', 1)[1]] = value
        return params

    def setStats(self, stats):
        if 'CPLEX nodes' not in stats:
            stats['CPLEX nodes'] = 0
        Decoder.setStats(self, stats)

    def solve(self, lb=-np.inf, ub=np.inf):
        self.cplex.objective.set_linear(zip(self.x, self.llrs))
        if self.sent is not None:
            sent = np.asarray(self.sent)
            # add sent codeword as CPLEX MIP start solution
            zValues = np.dot(self.code.parityCheckMatrix, sent / 2).tolist()
            self.cplex.MIP_starts.add([self.x + self.z, sent.tolist() + zValues],
                                      self.cplex.MIP_starts.effort_level.auto)
            self.callback.activate(np.dot(self.sent, self.llrs))
        self.cpxSolve()
        if self.sent is not None:
            if self.callback.occured:
                self.objectiveValue = self.callback.objectiveValue
                self.solution = self.callback.solution
                self.mlCertificate = False
                self.foundCodeword = True
            self.callback.deactivate()
            self.cplex.MIP_starts.delete()
        if self.sent is None or not self.callback.occured:
            if not self.callback.occured:
                checkKeyboardInterrupt(self.cplex)
            self.mlCertificate = self.foundCodeword = True
            self.objectiveValue = self.cplex.solution.get_objective_value()
            self.solution = np.rint(self.cplex.solution.get_values(self.x))
        self._stats['CPLEX nodes'] += self.cplex.solution.progress.get_num_nodes_processed()

    def cpxSolve(self):
        self.cplex.solve()

try:
    import cplex
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
            import cplex
            cplex.callbacks.MIPInfoCallback.__init__(self, *args, **kwargs)
            self.active = False
            self.occured = False
            self.realObjective = 0
            self.decoder = None

        def activate(self, objective):
            self.realObjective = objective
            self.active = True
            self.occured = False

        def __call__(self):
            if self.active and self.has_incumbent():
                incObj = self.get_incumbent_objective_value()
                if incObj < self.realObjective - 1e-6:
                    self.occured = True
                    self.objectiveValue = incObj
                    self.solution = np.rint(self.get_incumbent_values(self.decoder.x))
                    self.abort()

        def deactivate(self):
            self.active = False
except ImportError:
    pass  # CPLEX not available


# noinspection PyProtectedMember
def checkKeyboardInterrupt(cpx):
    """Checks the solution status of the given :class:`cplex.Cplex` instance for keyboard
    interrupts, and raises a :class:`KeyboardInterrupt` exception in that case.
    """
    import cplex
    if cpx.solution.get_status() in (cplex._internal._constants.CPX_STAT_ABORT_USER,
                                     cplex._internal._constants.CPXMIP_ABORT_FEAS,
                                     cplex._internal._constants.CPXMIP_ABORT_INFEAS):
        raise KeyboardInterrupt()
