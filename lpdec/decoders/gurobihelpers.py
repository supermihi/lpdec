# -*- coding: utf-8 -*-
# Copyright 2015 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation

"""This module contains helpers to simplify and unify usage of the gurobi python interface."""

from __future__ import print_function
from collections import OrderedDict
import sys
import numpy as np
from lpdec.decoders import Decoder


def createModel(name, version, **params):
    from gurobimh import Model, gurobi
    model = Model(name)
    model.setParam('OutputFlag', 0)
    if version:
        installedVersion = '.'.join(str(v) for v in gurobi.version())
        if version != installedVersion:
            raise RuntimeError('Installed Gurobi version {} does not match requested {}'
                               .format(installedVersion, version))
    for param, value in params.items():
        model.setParam(param, value)
    return model


class GurobiDecoder(Decoder):
    """Generic base class for Gurobi based LP/IP decoders.
    """

    def __init__(self, code, name, gurobiParams=None, gurobiVersion=None, integer=True):
        Decoder.__init__(self, code, name)
        if gurobiParams is None:
            gurobiParams = {}
        self.model = createModel(name, gurobiVersion, **gurobiParams)
        self.grbParams = gurobiParams.copy()
        from gurobimh import GRB
        vt = GRB.BINARY if integer else GRB.CONTINUOUS
        self.x = OrderedDict()
        for i in range(code.blocklength):
            for k in range(1, code.q):
                self.x[i, k] = self.model.addVar(0, 1, vtype=vt, name='x{},{}'.format(i, k))
        self.xlist = list(self.x.values())
        self.model.update()

    def fix(self, index, value):
        if value == 0:
            for k in range(1, self.code.q):
                self.x[index, k].ub = 0
                self.x[index, k].lb = 0
        else:
            for k in range(1, self.code.q):
                self.x[index, k].lb = value == k
                self.x[index, k].ub = value == k

    def release(self, index):
        for k in range(1, self.code.q):
            self.x[index, k].lb = 0
            self.x[index, k].ub = 1

    def setLLRs(self, llrs, sent=None):
        from gurobimh import LinExpr
        self.model.setObjective(LinExpr(llrs, self.xlist))
        Decoder.setLLRs(self, llrs, sent)

    def readSolution(self):
        self.objectiveValue = self.model.ObjVal
        codeword = True
        if self.code.q == 2:
            for i, x in enumerate(self.xlist):
                self.solution[i] = x.X
                if self.solution[i] > 1e-5 and self.solution[i] < 1-1e-5:
                    codeword = False
            return codeword
        for i in range(self.code.blocklength):
            self.solution[i] = 0
            for k in range(1, self.code.q):
                if self.x[i, k].X > 1e-5:
                    if self.solution[i] != 0:
                        self.solution[:] = .5  # error
                        return False
                    else:
                        self.solution[i] = k
        return True

    def params(self):
        ret = OrderedDict()
        if len(self.grbParams):
            ret['gurobiParams'] = self.grbParams
        import gurobimh
        ret['gurobiVersion'] = '.'.join(str(v) for v in gurobimh.gurobi.version())
        ret['name'] = self.name
        return ret
