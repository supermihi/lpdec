# -*- coding: utf-8 -*-
# Copyright 2011-2015 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation

# this is the empty __init__ file for the decoders package.

"""This module contains tools to create interleavers for turbo-like code.

Both QPP (quadratic permutation polynomial) interleavers and random interleavers are supported.
"""

from __future__ import unicode_literals, division

from collections import defaultdict, OrderedDict
import functools, itertools
import math, random

from lpdec.persistence import JSONDecodable


class Interleaver(JSONDecodable):
    """An interleaver of a given size s is a permutation on the set {0, ..., s-1}.
    
    The permutation function is available by just calling the interleaver object. Additionally,
    its inverse is supplied by the inv() function.
    """
     
    def __init__(self, **kwargs):
        """Initialize the interleaver by either filename, QPP parameters, ore explicit permutation.
        
        There are three methods to define an Interleaver:
        - read from a two-column text file of the form
            0    3
            1    2
            2    4
            3    0
            4    1
          given by the keyword argument *filename* (Interleaver(filename="int.txt"))
        - specify *size* and the two coefficients *f1* and *f2* of a quadratic permutation
          polynomial (QPP), given as f(x) = f1*x + f2*x² (Interleaver(size=20, f1=2, f2=13).
        - explicitly specify the permutation in the form of a list, defining the permuted sequence
          (0, …, size-1).
        """
        
        JSONDecodable.__init__(self)
        if "permutation" in kwargs:
            self.permutation = kwargs["permutation"]
            table = dict()
            invTable = dict()
            for i, piI in enumerate(self.permutation):
                table[i] = piI
                invTable[piI] = i
        elif "filename" in kwargs:
            self.filename = kwargs["filename"]
            table = dict()
            invTable = dict()
            with open(self.filename, "rt") as intFile:
                for line in intFile:
                    arg, val =  [ int(decimal) for decimal in line.strip().split() ]
                    table[arg] = val
                    invTable[val] = arg
        else:
            if not ("size" in kwargs and "f1" in kwargs and "f2" in kwargs):
                raise ValueError("Need either permutation, filename, or all of {size, f1, f2}")
            self.f1, self.f2 = kwargs["f1"], kwargs["f2"]
            self.size = kwargs["size"]
            table = dict(( (x, (self.f1*x + self.f2*x*x) % self.size) for x in range(self.size) ))
            if len(set(table.values())) != self.size:
                raise ValueError("Given size/f1/f2 parameters {}/{}/{} don't define a permutation"
                                 .format(self.size, self.f1, self.f2))
            invTable = dict((x, y) for y, x in table.items())
        self.size = len(table)
        self.table = [ table[key] for key in sorted(table) ]
        self.invTable = [ invTable[key] for key in sorted(invTable) ]
        
    @staticmethod
    def random(length):
        """Creates and returns a random interleaver of the given length."""
         
        indexes = list(range(length))
        shuffle(indexes)
        return Interleaver(permutation=indexes)
        
    def __call__(self, i):
        """The permutation function."""
        return self.table[i]
    
    def inv(self, i):
        """The inverse permutation function."""
        return self.invTable[i]
    
    def params(self):
        if hasattr(self, "f1"):
            return OrderedDict([ ("size", self.size), ("f1", self.f1), ("f2", self.f2) ])
        else:
            return dict(permutation=[self(i) for i in range(self.size)])
    
    def inverted(self):
        """Returns an interleaver whose permutation is the inverse of this one."""
        return Interleaver(permutation = [self.inv(i) for i in range(self.size)])
    
    def __str__(self):
        if hasattr(self, "filename"):
            return "{}-bit interleaver({})".format(self.size, self.filename)
        if hasattr(self, "f1"):
            return "QPP {}-bit interleaver[f1={},f2={}]".format(self.size, self.f1, self.f2)
        return "Interleaver({})".format([self(i) for i in range(self.size)])
    
    def __eq__(self, other):
        return not (self != other)
    
    def __ne__(self, other):
        return self.table != other.table
    
    
def shuffle(sequence):
    """Randomly permute the elements of x in-place.
    
    This method implements the Knuth shuffle procedure and thus provides better distribution
    than the random.shuffle.
    """
    for i in reversed(range(1, len(sequence))):
        j = random.randrange(i+1)
        sequence[i], sequence[j] = sequence[j], sequence[i]


def factorize(n):
    """Factorize the given number *n*, returning a dict that maps factors to their multiplicity.
    """
    # pylint: disable=C0103
    # math function: using *n* is okay
    if n == 0:
        raise ValueError("Can't factorize 0!")
    if n == 1:
        return dict()  # special case
    res = defaultdict(lambda: 0)
    while n % 2 == 0:
        res[2] += 1
        n //= 2
    # Try odd numbers up to sqrt(n)    
    i = 3
    while i <= n:
        while n % i == 0:
            res[i] += 1
            n //= i
        i += 2
    #if n != 1:
    #    res.append(n)
    return res


def randomf1(size, factors):
    """Creates a random number f1 in [1, size) such that for all x in factors: x \nmid f1"""
    while True:
        f1 = random.randrange(1, size+1, 2) #  random odd number
        #  ensure gcd(f1, size) == 1
        if not any(f1 % factor == 0 for factor in factors):
            return f1


def randomf2(size, kernel):
    """Creates a random number f2 in [1,size) such that kernel|f2"""
    if size == kernel:
        raise ValueError("Can only compute f2 if size != kernel")
    return random.randrange(kernel, size, kernel)


def randomf1f2(size, onlyQI=False):
    """Generate random coefficients f1,f2 for a QPP interleaver of given size"""
    factors = frozenset(factorize(size))
    f1 = randomf1(size, factors)
    if onlyQI:
        f2 = random.choice(allf2(size, onlyQI=True))
    else:
        kernel = functools.reduce(lambda a, b : a * b, factors, 1)
        f2 = randomf2(size, kernel)
    return f1, f2


def allf1(size):
    """Generates all valid coefficients f1 for a QPP interleaver of given size"""
    factors = frozenset(factorize(size).keys())
    for i in range(1, size):
        if not any(map(lambda n: i%n==0, factors)):
            yield i


def allf2(size, onlyQI=False):
    """Generates all valid f2 coefficients for a QPP interleaver of given size.
    
    If onlyQI is true, only generate coefficients that lead to a QPP with a QPP inverse.
    """
    factors = factorize(size)
    if onlyQI:
        kernel = 1
        if factors[2] > 1:
            kernel *= 2**max(math.ceil( (factors[2]-2) / 2), 1)
        if factors[3] > 0:
            kernel *= 3**max(math.ceil( (factors[3]-1) / 2), 1)
        for factor in set(factors.keys()) - {2, 3}:
            if factors[factor] > 0:
                kernel *= factor**math.ceil(factors[factor]/2)
        kernel = int(round(kernel))
    else:
        kernel = functools.reduce(lambda a, b : a * b, factors.keys(), 1)
    return ( range(kernel, size, kernel) )


def randomQPP(size, onlyQI=False):
    """Generate a random QPP of given size and return a corresponding Interleaver instance.

    The optional *onlyQI* parameter, if set to True, specifies that only QPPs with a quadratic
    inverse should be selected."""
    f1, f2 = randomf1f2(size, onlyQI)
    return Interleaver(size=size, f1=f1, f2=f2)


def allQPPInterleavers(size, unique=True, onlyQI=False):
    """Generates *all* QPP interleavers of a given size.
    
    If unique=True, interleavers with different f1,f2-coefficients that generate the same function
    are contained only once. If onlyQI=True, only QPPs with a quadratic inverse are considered.
    """
    # pylint: disable=C0103
    interleavers = dict()
    for f1, f2 in itertools.product(allf1(size), allf2(size, onlyQI)):
        table = [ (x, (f1*x + f2*x*x) % size) for x in range(size) ]
        found = False
        if unique:
            for existingQPP in interleavers.values():
                if table == existingQPP:
                    found = True
                    break
        if not found:
            interleavers[(f1, f2)] = table
    return interleavers


class LTEInterleaver(Interleaver):
    """The class of interleavers specified for 3GPP LTE Turbo Codes."""
    
    def __init__(self, k):
        """Create the interleaver of size *k*, which must exist in the standard."""
        try:
            f1, f2 = self.lteTable[k]
        except KeyError:
            raise KeyError('{} is not a valid 3GPP LTE turbo code input length'.format(k))
        Interleaver.__init__(self, size=k, f1=f1, f2=f2)
    
    def params(self):
        return dict(k=self.size)
    
    @classmethod
    def availableBlocklengths(cls):
        """Returns all available block lengths defined in the 3GPP LTE standard."""
        return cls.lteTable.keys()

    def __str__(self):
        return 'LTE {}-bit interleaver'.format(self.size)
    
    lteTable = dict(((40, (3, 10)), (48, (7, 12)), (56, (19, 42)), (64, (7, 16)), (72, (7, 18)),
                     (80, (11, 20)), (88, (5, 22)), (96, (11, 24)), (104, (7, 26)),
                     (112, (41, 84)), (120, (103, 90)), (128, (15, 32)), (136, (9, 34)),
                     (144, (17, 108)), (152, (9, 38)), (160, (21, 120)), (168, (101, 84)),
                     (176, (21, 44)), (184, (57, 46)), (192, (23, 48)), (200, (13, 50)),
                     (208, (27, 52)), (216, (11, 36)), (224, (27, 56)), (232, (85, 58)),
                     (240, (29, 60)), (248, (33, 62)), (256, (15, 32)), (264, (17, 198)),
                     (272, (33, 68)), (280, (103, 210)), (288, (19, 36)), (296, (19, 74)),
                     (304, (37, 76)), (312, (19, 78)), (320, (21, 120)), (328, (21, 82)),
                     (336, (115, 84)), (344, (193, 86)), (352, (21, 44)), (360, (133, 90)),
                     (368, (81, 46)), (376, (45, 94)), (384, (23, 48)), (392, (243, 98)),
                     (400, (151, 40)), (408, (155, 102)), (416, (25, 52)), (424, (51, 106)),
                     (432, (47, 72)), (440, (91, 110)), (448, (29, 168)), (456, (29, 114)),
                     (464, (247, 58)), (472, (29, 118)), (480, (89, 180)), (488, (91, 122)),
                     (496, (157, 62)), (504, (55, 84)), (512, (31, 64)), (528, (17, 66)),
                     (544, (35, 68)), (560, (227, 420)), (576, (65, 96)), (592, (19, 74)),
                     (608, (37, 76)), (624, (41, 234)), (640, (39, 80)), (656, (185, 82)),
                     (672, (43, 252)), (688, (21, 86)), (704, (155, 44)), (720, (79, 120)),
                     (736, (139, 92)), (752, (23, 94)), (768, (217, 48)), (784, (25, 98)),
                     (800, (17, 80)), (816, (127, 102)), (832, (25, 52)), (848, (239, 106)),
                     (864, (17, 48)), (880, (137, 110)), (896, (215, 112)), (912, (29, 114)),
                     (928, (15, 58)), (944, (147, 118)), (960, (29, 60)), (976, (59, 122)),
                     (992, (65, 124)), (1008, (55, 84)), (1024, (31, 64)), (1056, (17, 66)),
                     (1088, (171, 204)), (1120, (67, 140)), (1152, (35, 72)), (1184, (19, 74)),
                     (1216, (39, 76)), (1248, (19, 78)), (1280, (199, 240)), (1312, (21, 82)),
                     (1344, (211, 252)), (1376, (21, 86)), (1408, (43, 88)), (1440, (149, 60)),
                     (1472, (45, 92)), (1504, (49, 846)), (1536, (71, 48)), (1568, (13, 28)),
                     (1600, (17, 80)), (1632, (25, 102)), (1664, (183, 104)), (1696, (55, 954)),
                     (1728, (127, 96)), (1760, (27, 110)), (1792, (29, 112)), (1824, (29, 114)),
                     (1856, (57, 116)), (1888, (45, 354)), (1920, (31, 120)), (1952, (59, 610)),
                     (1984, (185, 124)), (2016, (113, 420)), (2048, (31, 64)), (2112, (17, 66)),
                     (2176, (171, 136)), (2240, (209, 420)), (2304, (253, 216)),
                     (2368, (367, 444)), (2432, (265, 456)), (2496, (181, 468)), (2560, (39, 80)),
                     (2624, (27, 164)), (2688, (127, 504)), (2752, (143, 172)), (2816, (43, 88)),
                     (2880, (29, 300)), (2944, (45, 92)), (3008, (157, 188)), (3072, (47, 96)),
                     (3136, (13, 28)), (3200, (111, 240)), (3264, (443, 204)), (3328, (51, 104)),
                     (3392, (51, 212)), (3456, (451, 192)), (3520, (257, 220)), (3584, (57, 336)),
                     (3648, (313, 228)), (3712, (271, 232)), (3776, (179, 236)),
                     (3840, (331, 120)), (3904, (363, 244)), (3968, (375, 248)),
                     (4032, (127, 168)), (4096, (31, 64)), (4160, (33, 130)), (4224, (43, 264)),
                     (4288, (33, 134)), (4352, (477, 408)), (4416, (35, 138)), (4480, (233, 280)),
                     (4544, (357, 142)), (4608, (337, 480)), (4672, (37, 146)), (4736, (71, 444)),
                     (4800, (71, 120)), (4864, (37, 152)), (4928, (39, 462)), (4992, (127, 234)),
                     (5056, (39, 158)), (5120, (39, 80)), (5184, (31, 96)), (5248, (113, 902)),
                     (5312, (41, 166)), (5376, (251, 336)), (5440, (43, 170)), (5504, (21, 86)),
                     (5568, (43, 174)), (5632, (45, 176)), (5696, (45, 178)), (5760, (161, 120)),
                     (5824, (89, 182)), (5888, (323, 184)), (5952, (47, 186)), (6016, (23, 94)),
                     (6080, (47, 190)), (6144, (263, 480))))
