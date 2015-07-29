# -*- coding: utf-8 -*-
# Copyright 2011-2015 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation

"""This module contains classes for trellis-based code definitions."""

from __future__ import unicode_literals, absolute_import

import math
from collections import OrderedDict
from lpdec.persistence import JSONDecodable


class ConvolutionalEncoder(JSONDecodable):
    """A class for representing convolutional encoders.
    
    Convolutional encoders are defined by a state transition function: For
    a given current state and input bit, the table outputs the subsequent
    state and the output bit emitted during this transition. Note that, at
    the moment, only one output per transition is supported. The table can
    be read from a text file containing lines of the format
    
    a b c d
    
    where a is the state before transition, b is the input bit, c the next
    state, and d the output bit."""
     
    def __init__(self, filename=None, transitionTable=None, name=None):
        """Create a new Encoder by either givin a file name or the table.
        
        If the table is not given by a text file via *filename*, then
        *transitionTable* must either be a dictionary mapping (a,b) tuples
        to (c,d) tuples (cf. class documentation) or a list of tuples
        (a,b,c,d).
        
        If you plan to use this encoder for a code that is to be stored in a database,
        you should also specify a unique name via *name*."""
        # todo: allow creation by defining polynomials
        self.forwardMap = {}
        self.backwardMap = {}
        if transitionTable is not None:
            if isinstance(transitionTable, dict):
                self.forwardMap = transitionTable
            else:
                self.forwardMap = {(t[0], t[1]):(t[2], t[3])
                                    for t in transitionTable }
        else:
            with open(filename, "rt") as tableFile:
                for line in tableFile:
                    oldState, inbit, newState, outbit = map(int, line.strip().split())
                    self.forwardMap[oldState, inbit] = newState, outbit
        self.backwardMap = {(y[0], x[1]):(x[0], y[1])
                             for x, y in self.forwardMap.items()}
        # number of states: maximum index + 1
        self.states = max(list(zip(*self.forwardMap))[0]) + 1
        self.tailbits = int(math.log(self.states, 2))
        self.name = 'unnamed' if name is None else name
        
    def stateTransition(self, state, inputBit):
        """Returns next state and parity output for given state/input as a tuple."""
        return self.forwardMap[state, inputBit]
    
    def stateTransitionBack(self, state, inputBit):
        """The "pseudo-inverse" function of stateTransition().
        
        More specifically, returns a tuple (origState, outputBit) such that the
        encoder transits from *origState* to *state* when fed an *inputBit* and
        emits *outputBit* thereby."""
        return self.backwardMap[state, inputBit]
    
    def params(self):
        return OrderedDict([
                ('transitionTable', list(a + b for a, b in self.forwardMap.items())),
                ('name', self.name) ])
    
    def __str__(self):
        return self.name
    
    def __eq__(self, other):
        return self.forwardMap == other.forwardMap
    
    def __ne__(self, other):
        return self.forwardMap != other.forwardMap


class RepeatAccumulateEncoder(ConvolutionalEncoder):
    """The finite state machine corresponding to an RA code."""
    
    def __init__(self):
        table = {(0, 0): (0, 0), (0, 1): (1, 1), (1, 1): (0, 0), (1, 0): (1, 1)}
        ConvolutionalEncoder.__init__(self, transitionTable=table, name ='RA Encoder')
    
    def params(self):
        return dict()


class LTEEncoder(ConvolutionalEncoder):
    """The encoder given by (1+D+D^3)/(1+D^2 + D^3), as defined in the LTE standard."""
    def __init__(self):
        ConvolutionalEncoder.__init__(
                self,
                transitionTable={(0, 1): (4, 1), (0, 0): (0, 0), (7, 0): (3, 0),
                                 (7, 1): (7, 1), (3, 0): (1, 1), (6, 1): (3, 1),
                                 (3, 1): (5, 0), (6, 0): (7, 0), (2, 1): (1, 0),
                                 (2, 0): (5, 1), (5, 0): (6, 1), (5, 1): (2, 0),
                                 (1, 0): (4, 0), (4, 1): (6, 0), (1, 1): (0, 1),
                                 (4, 0): (2, 1)},
                name='LTE encoder')
    
    def params(self):
        return dict()


class TDInnerEncoder(ConvolutionalEncoder):
    """The convolutional encoder given by 1/(1+D^2), used as inner encoder in 3-D Turbo Codes."""
    def __init__(self):
        ConvolutionalEncoder.__init__(
                self,
                transitionTable={(0,0): (0,0), (0,1): (2,1), (1,0): (2,1),
                                 (1,1): (0,0), (2,0): (1,0), (2,1): (3,1),
                                 (3,0): (3,1), (3,1): (1,0)},
                name='3D inner encoder')
    
    def params(self):
        return dict()