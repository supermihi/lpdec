# -*- coding: utf-8 -*-
# Copyright 2011-2015 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation
from __future__ import division, print_function

"""A general framework for the definition of binary turbo-Like codes, including 3D turbo codes.

Turbo-Like codes are represented by a directed acyclic graph, where a (unique) source vertex
corresponds to the "information input" and a unique target vertex is the codeword output.
Intermediate vertices can be component (convolutional) encoders, muxers and demuxers. Arcs are
parallel connections of a certain number of bits, and may have an attached interleaver which
permutes the bits.
"""

from collections import OrderedDict

import numpy as np
from lpdec.codes import interleaver, convolutional, BinaryLinearBlockCode, trellis
from lpdec.codes.trellis import INFO, PARITY


class TurboVertex:
    """Base class for all vertices of turbo-like codes.

    The class property *isStopper* tells whether the vertex is a "stopper", that is, doing more
    complicated things with the incoming bits than just routing them to a specific output position.
    At the moment, the information source, code vertex, and constituend encoders are stoppers,
    while (de)muxing vertices are no stoppers.
    """
    
    isStopper = False
    
    @property
    def inSize(self):
        return sum(arc.size for arc in self.inArcs)
    
    @property
    def outSize(self):
        return sum(arc.size for arc in self.outArcs)    
    
    def __init__(self):
        self.inArcs = []
        self.outArcs = []
        self.name = ''

    def connect(self, target, size, interleaver=None):
        """Connect *size* bits of this vertex' output to the input of *target*.

        Creates an InterleaverArc if an interleaver is specified, or an ordinary TurboArc
        otherwise.
        """
        if interleaver is None:
            TurboArc(size, self, target)
        else:
            InterleaverArc(interleaver, self, target)

    def finalize(self):
        self.connections = {}
        for arc in self.outArcs:
            for i in range(arc.size):
                vertex, pos = arc.endOfPath(i)
                if vertex not in self.connections:
                    self.connections[vertex] = []
                self.connections[vertex].append([i, pos])
        for v in list(self.connections.keys()):
            self.connections[v] = np.array(self.connections[v], dtype=np.int)
    

class InformationSource(TurboVertex):
    """The vertex representing the information source in a turbo-like code. Each
    code must have exactly one information source.

    The information source can be connected to several vertices, which is interpreted
    such that each connection carries a complete copy of the information input.
    """

    def __init__(self, infobits):
        """Create the source vertex with information length *infobits*."""
        TurboVertex.__init__(self)
        self.infobits = infobits

    def connect(self, target, interleaver=None):
        """Convenience function: Automatically determine the size"""
        TurboVertex.connect(self, target, self.infobits, interleaver)

    def finalize(self):
        TurboVertex.finalize(self)
        self.numTrellises = []
        self.trellises = []
        self.segments = []
        for i in range(self.infobits):
            ends = [arc.endOfPath(i) for arc in self.outArcs
                                     if isinstance(arc.endOfPath(i)[0], EncoderVertex )]
            self.numTrellises.append(len(ends))
            self.trellises.append([])
            self.segments.append([])
            for j, (vertex, segment) in enumerate(ends):
                vertex.isInfoConnected = True
                self.trellises[i].append(vertex.trellis)
                self.segments[i].append(segment)

    def __repr__(self):
        return 'InformationSource({})'.format(self.infobits)


class EncoderVertex(TurboVertex):
    """Represents a constituent convolutional encoder. Each encoder has an
    associated Trellis graph. Depending on the termination scheme used, the
    output might be larger than the input.
    """
    
    isStopper = True
    
    def __init__(self, encoder, length, name=None, **kwargs):
        """Create the EncoderVertex, defined by the *encoder* (an instance of
        ConvolutionalEncoder) and *length* which is the trellis length. Note that
        for tailbiting trellises, the length is inputSize+tailSize. Any additional
        keyword arguments will be passed to the Trellis constructor."""
        TurboVertex.__init__(self)
        self.trellis = trellis.Trellis(encoder, length, name=name, **kwargs)
        self.name = name
        self._inWord = np.empty(self.trellis.insize, dtype=np.int)
        self.isInfoConnected = False

    def connect(self, target):
        """Convenience function: Automatically sets *size*."""
        TurboVertex.connect(self, target, self.trellis.outsize)

    def __repr__(self):
        return 'EncoderVertex(name={})'.format(self.name)


class MuxVertex(TurboVertex):
    """A MuxVertex takes several inputs and muxes them together into one output.

    Muxing is performed in either of the following styles:

    - ``'sequential'``: The bits of the inputs are sequentially concatenated
    - ``'alternating'``: The inputs are "interleaved" into the output.

    Example
    -------
    Assume there are three inputs of length four each, so we have the bits

    - from input ``a``: ``a1 a2 a3 a4``
    - from input ``b``: ``b1 b2 b3 b4``
    - from input ``c``: ``c1 c2 c3 c4``

    Sequential muxing outputs ``a1 a2 a3 a4 b1 b2 b3 b4 c1 c2 c3 c4``, while alternating
    muxing yields ``a1 b1 c1 a2 b2 c2 a3 b3 c3 a4 b4 c4``.

    Parameters
    ----------
    style : {'sequential', 'alternating'}, optional
        Selects the muxing scheme, which defaults to 'sequential'.
    """
    
    def __init__(self, style='sequential', name=None):
        """Creates a MuxVertex with the given muxing style ('sequential' or 'alternating')."""
        TurboVertex.__init__(self)
        self.style = style
        self.name = name

    def connect(self, target, interleaver=None):
        """Connect to the target; automatically sets the size of the outgoing arc. Note
        that this method has to be called /after/ all ingoing connections have been defined,
        since the total size is not known before."""
        TurboVertex.connect(self, target, self.inSize, interleaver)

    def outPosition(self, inArc, inBit):
        """Returns a tuple (arc, pos) such that the *inBit*-th bit incoming through *inArc*
        is the *pos*-th bit on *arc*. Note that *arc* will always be the single output arc."""
        inIndex = self.inArcs.index(inArc)
        if self.style == 'alternating':
            return self.outArcs[0], len(self.inArcs) * inBit + inIndex
        else:
            sum = 0
            for i in range(inIndex):
                sum += self.inArcs[i].size
            sum += inBit
            return self.outArcs[0], sum

    def inPosition(self, outArc, outBit):
        """Returns a tuple (arc, pos) such that the *pos*-th bit incoming through *arc* is
        the *outPos*-th bit on *outArc* (which must be the single output arc).
        """
        
        if self.style == 'alternating':
            pos, inIndex = divmod(outBit, len(self.inArcs))
            return self.inArcs[inIndex], pos
        else:
            sumSize = 0
            for arc in self.inArcs:
                if outBit < arc.size + sumSize:
                    return arc, outBit - sumSize
                sumSize += arc.size
            raise ValueError('Something went wrong')

    def __repr__(self):
        return 'MuxVertex(name={})'.format(self.name)


class DeMuxVertex(TurboVertex):
    """Takes a single input and splits its bits into  a number of output arcs.

    DeMuxVertex acts as a counterpart of a MuxVertex. Which bit goes to which output is determined
    by a pattern, which is a tuple of output positions. For instance, the pattern ``(0,1,1)`` means
    that every third bit goes to the first output and the rest go to the second output.

    More formally, if :math:`(p_1, \dots, p_k)` is a pattern, then the i-th input bit goes to output
    :math:`p_{i \bmod k}`. Note how this supports outputs of different length.
    """
    
    def __init__(self, pattern, name=None):
        TurboVertex.__init__(self)
        if pattern is None:
            pattern = np.array((0, 1))
        self.pattern = pattern
        numPatterns = np.max(pattern) + 1
        self.patSizes = np.zeros(numPatterns, dtype=np.int)
        for symbol in pattern:
            self.patSizes[symbol] += 1
        self.name = name

    def connect(self, target, interleaver=None):
        """Connects this demuxer to another vertex, optionally using an interleaver.

        This method has to be called after the input connection has been made, and each call
        corresponds to the next symbol of the pattern. The size is determined automatically
        from the insize and the pattern.

        Parameters
        ----------
        target : TurboVertex
            The target vertex to connect to.
        interleaver: :class:`.Interleaver`, optional
            An interleaver to use. If left out, no interleaving is performed.
        """
        outSize = self.inSize * self.patSizes[len(self.outArcs)] // len(self.pattern)
        TurboVertex.connect(self, target, outSize, interleaver)

    def inPosition(self, outArc, outPos):
        """Compute input position routed to *outPos*-th bit on *outArc*.

        Returns a tuple (arc, pos) such that the *pos*-th input bit on *arc* (the single incoming
        arc) is routed to the *outPos*-th bit on *outArc*.
        """
        i = self.outArcs.index(outArc)
        patternsBefore = outPos // self.patSizes[i]
        inPos = len(self.pattern) * patternsBefore
        remainder = outPos % self.patSizes[i]
        for symbol in self.pattern:
            if symbol == i:
                if remainder == 0:
                    break
                remainder -= 1
            inPos += 1
        return self.inArcs[0], inPos

    def outPosition(self, inArc, inBit):
        """Compute output position and arc to which *inBit* of the single *inArc* is routed.

        Returns a tuple (arc, pos) such that the *inBit*-th input bit on *inArc* (the single
        incoming arc) is routed to the *pos*-th bit of *arc*.
        """
        assert inArc == self.inArcs[0]
        positionInPattern = inBit % len(self.pattern)
        outArcNum = self.pattern[positionInPattern]
        outBit = (inBit // len(self.pattern)) \
                 * self.patSizes[outArcNum] \
                 + np.count_nonzero(self.pattern[:positionInPattern] == outArcNum)
        return self.outArcs[outArcNum], outBit

    def __repr__(self):
        return 'DeMuxVertex(name={})'.format(self.name)


class CodeVertex(TurboVertex):
    """A CodeVertex represents the codeword of a turbo-like code.

    It has a single incoming arc, the bits of which comprise the codeword."""

    isStopper = True

    def finalize(self):
        TurboVertex.finalize(self)
        blocklength = self.inSize
        self.trellises = []
        self.labels = []
        self.segments = []
        for i in range(blocklength):
            vertex, pos = self.inArcs[0].startOfPath(i)
            if isinstance(vertex, InformationSource):
                self.trellises.append(None)
                self.segments.append(pos)
                self.labels.append(None)
            else:
                trellis = vertex.trellis
                self.trellises.append(trellis)
                seg, lab = trellis.segmentAndLabel(pos, 'out')
                self.segments.append(seg)
                self.labels.append(lab)

    def __repr__(self):
        return 'CodeVertex'


class TurboArc:
    """A TurboArc connects two TurboVertex instances.

    It has the attributes:
      - size, the number of bits transfered through the arc
      - tail, the vertex where it starts,
      - head, the vertex where it ends.
    """

    def __init__(self, size, tail, head):
        """Creates the arc and directly appends it to head's and tail's *outArcs* and *inArcs*.
        """
        self.size = size
        self.tail = tail
        self.head = head
        self.tail.outArcs.append(self)
        self.head.inArcs.append(self)

    def endPosition(self, start):
        """Returns the position to which *start*-th input bit is wired.

        This is the identity mapping for non-interleaver arcs."""
        # pylint: disable=R0201
        return start

    def startPosition(self, end):
        """Returns the position which is wired to the *end*-th bit in the output.

        This is the identity mapping for non-interleaver arcs."""
        # pylint: disable=R0201
        return end

    def endOfPath(self, i):
        """Compute where the *i*-th bit on this arc encounters the next stopper vertex.

        Returns a tuple (vertex, pos) such that the *i*-th bit of this arc is wired to the *pos*-th
        input bit of *vertex*, which is a stopper vertex (encoder or codeword). In other words,
        this method identifiers the terminal position of the *i*-th bit on its path through muxers
        and demuxers.
        """
        if i < 0 or i >= self.size:
            raise ValueError('value {} out of range for arc {}'.format(i, self))
        if self.head.isStopper:
            return self.head, self.endPosition(i)
        else:
            arc, bit = self.head.outPosition(self, i)
            return arc.endOfPath(bit)

    def startOfPath(self, i):
        """Compute where the *i*-th bit of this arc originates at an encoder or InformationSource.

        This is complementary to endOfPath: Returns a tuple (vertex, pos) such that the bit at
        position *i* at the /end/ of this arc originates from *vertex* (encoder or
        InformationSource) at output position *pos*.
        """
        if i < 0 or i >= self.size:
            raise ValueError('value {} out of range for arc {}'.format(i, self))
        if self.tail.isStopper or isinstance(self.tail, InformationSource):
            return self.tail, self.startPosition(i)
        else:
            arc, bit = self.tail.inPosition(self, i)
            return arc.startOfPath(bit)


class InterleaverArc(TurboArc):
    """A TurboArc with an interleaver that permutes the bits between *tail* and *head*.
    """

    def __init__(self, interleaver, tail, head):
        """Create the arc; *interleaver* must be an instance of Interleaver.
        """
        TurboArc.__init__(self, interleaver.size, tail, head)
        self.interleaver = interleaver

    def endPosition(self, start):
        return self.interleaver(start)

    def startPosition(self, end):
        return self.interleaver.inv(end)


class TurboLikeCode(BinaryLinearBlockCode):
    """Defines a turbo-like code consisting of vertices and arcs as defined above.
    """

    def __init__(self, vertices, name):
        """Create the code with TurboVertices *vertices* and a given *name*.

        *vertices* must be a list of TurboVertex instances. The attributes *blocklength*,
        *infolength* and *rate* are computed automatically from the code layout.

        After initialization, the information source vertex is stored in the *infoVertex*
        attribute, the code vertex as *codeVertex*, and all encoder vertices are put in the tuple
        *encoders*.
        """

        encoders = []
        self.infoVertex = self.codeVertex = None
        for vertex in vertices:
            if isinstance(vertex, InformationSource):
                assert self.infoVertex is None
                self.infoVertex = vertex
            elif isinstance(vertex, CodeVertex):
                assert self.codeVertex is None
                self.codeVertex = vertex
            elif isinstance(vertex, EncoderVertex):
                encoders.append(vertex)
        self.vertices = vertices[:]
        self.encoders = encoders
        self.blocklength = self.codeVertex.inSize
        self.codeVertex._inWord = np.empty(self.blocklength, dtype=np.int)
        self.infolength = self.infoVertex.infobits
        BinaryLinearBlockCode.__init__(self, name=name)
        self.stoppers = [self.infoVertex] + self.encoders + [self.codeVertex]

        for vertex in self.stoppers:
            vertex.finalize()
        for i in range(self.blocklength):
            sals = self.segmentsForCodeBit(i)
            for seg, lab in sals:
                if lab == trellis.INFO:
                    seg.info_code_bit = i
                    seg.info_code_ratio = 1/len(sals)
                else:
                    seg.parity_code_bit = i
                    seg.parity_code_ratio = 1/len(sals)

    def matchingPath(self, source, path, target):
        """Compute the path in *target*'s trellis matching *path* in *source*'s trellis.

        Both *source* and *target* must be encoder vertices. This method only works if both
        encoders are connected directly to the information source, as in standard turbo codes.
        The inner encoder of 3D turbo codes, for example, would not be supported.
        """
        sourceArc = source.inArcs[0]
        targetArc = target.inArcs[0]
        assert sourceArc.tail == self.infoVertex
        assert targetArc.tail == self.infoVertex

        outPath = np.empty(target.trellis.length, dtype=np.int)
        node = target.trellis.initVertex
        cost = 0
        calcPos = lambda pos : sourceArc.endPosition(targetArc.startPosition(pos))

        for i in range(target.trellis.length):
            if len(node.outArcs()) > 1:
                arc = node.outArcs()[path[calcPos(i)]]
            else:
                arc = node.outArcs()[0]
            outPath[i] = arc.infobit
            cost += arc.cost
            node = arc.head
        return outPath, cost


    def encode(self, infoword):
        self.infoVertex._outWord = infoword
        for vertex in self.stoppers:
            if isinstance(vertex, EncoderVertex):
                vertex._outWord = vertex.trellis.encode(vertex._inWord)
            for targetVertex, info in vertex.connections.items():
                for outP, inP in info:
                    targetVertex._inWord[inP] = vertex._outWord[outP]
        return self.codeVertex._inWord


    def encodePath(self, path, encoder):
        """Try to encode the given *path* in the trellis of EncoderVertex *encoder* to a codeword.

        This will only work if the encoder is directly connected to the information source.
        """
        infoWord = np.empty(self.infolength, dtype=np.int)
        if encoder.inArcs[0].tail != self.infoVertex:
            raise ValueError('invalid encoder for TurboLikeCode.encode()')
        for i in range(self.infolength):
            infoWord[i] = path[encoder.inArcs[0].endPosition(i)]
        return self.encode(infoWord)
    
    def segmentsForCodeBit(self, i):
        position = self.codeVertex.segments[i]
        label = self.codeVertex.labels[i]
        if self.codeVertex.trellises[i] is None:
            numInfos = self.infoVertex.numTrellises[position]
            ret = []
            for j in range(numInfos):
                trellis = self.infoVertex.trellises[position][j]
                segment = trellis[self.infoVertex.segments[position][j]]
                ret.append((segment, INFO))
            return ret
        else:
            trellis = self.codeVertex.trellises[i]
            segment = trellis[position]
            return [(segment, label)]
    
    def trellisSegmentsOfOutBit(self, i):
        """Returns those arcs of the trellis graphs in code that define the *i*-th output bit.

        If *i* is in the systematic part, i.e. a copy of an input bit, the returned list contains
        a list of arcs for each segments corresponding to that bit -- i.e. in the case of a
        standard turbo code, this would return the "one"-arcs of the *i*-th segment of trellis1 and
        the *π(i)*-th segment of trellis2.

        If on the other hand the *i*-th bit corresponds to a parity bit of some trellis (or
        termination bits), the returned list has length one, its only element being the list of
        arcs corresponding to that bit ("parity"-arcs in case of a normal parity bit, "one"-arcs
        for termination bits).
        """
        start, pos = self.codeVertex.inArcs[0].startOfPath(i)
        outList = []
        if start is self.infoVertex:
            for out in start.outArcs:
                end, endPos = out.endOfPath(pos)
                if isinstance(end, EncoderVertex):
                    outList.append(end.trellis.inArcs(endPos))
        else:
            outList.append(start.trellis.outArcs(pos))
        return outList

    def params(self):
        raise NotImplementedError()


class StandardTurboCode(TurboLikeCode):
    """
    A conventional parallely concatenated turbo code.

    The code uses the following encoding scheme::

        u −−−+−−−−−−−−−−−−−−−−‣
             |
             +−−−[Encoder1]−−−‣
             |
        [Interleaver]
             |
             +−−−[Encoder2]−−−‣

    The encoders are terminated. The output is muxed in the following way:
    [input][enc1 termination][enc1 output (incl. tail)][enc2 termination][enc2 output (incl. tail)]
    If k is the size of the information word, the blocklength is thus 3*k+4*M, where M is the
    memory of the encoder.
    """

    def __init__(self, encoder, interleaver, name):
        """Create the code defined by a convolutional encoder, an interleaver and a name.

        The code's size is determined by the size of the interleaver.
        """
        self.infolength = interleaver.size
        self.encoder = encoder
        self.interleaver = interleaver
        if name is None:
            name = "StandardTurbo({}".format(interleaver.size)
        source = InformationSource(interleaver.size)
        codeV = CodeVertex()
        mux = MuxVertex('sequential')
        encoder1 = EncoderVertex(encoder, self.infolength + encoder.tailbits, 'C_a')
        encoder2 = EncoderVertex(encoder, self.infolength + encoder.tailbits, 'C_b')

        source.connect(mux)
        source.connect(encoder1)
        source.connect(encoder2, interleaver)

        encoder1.connect(mux)
        encoder2.connect(mux)

        mux.connect(codeV)
        TurboLikeCode.__init__(self, [source, encoder1, encoder2, mux, codeV], name)
        assert self.blocklength == interleaver.size * 3 + 4 * encoder.tailbits

    def params(self):
        return OrderedDict([("encoder", self.encoder),
                            ("interleaver", self.interleaver),
                            ("name", self.name) ])


class LTETurboCode(StandardTurboCode):
    """Turbo code as defined in the 3GPP LTE standard."""

    def __init__(self, infolength):
        """Construct an LTE code for the given *infolength*, which must be defined in the standard.
        """
        encoder = convolutional.LTEEncoder()
        inter = interleaver.LTEInterleaver(infolength).inverted()
        StandardTurboCode.__init__(self, encoder, inter, "LTE Turbo Code(k={})".format(infolength))

    def params(self):
        return {"infolength": self.infolength}


class ThreeDTurboCode(TurboLikeCode):
    """A 3-dimensional turbo code, i.e. a standard turbo code with an additional "patch" encoder.

    A part of the output bits of the normal encoders are sent through a smaller third encoder to
    improve performance.
    """

    def __init__(self, K, outerInterleaver, innerInterleaver, name):
        """Create a 3-dimensional turbocode with given outer and inner interleavers.
        """
        # pylint: disable=C0103
        if name is None:
            raise ValueError("3D turbo codes need a name")
        self.outerInterleaver = outerInterleaver
        self.innerInterleaver = innerInterleaver
        source = InformationSource(K)
        encoder = convolutional.LTEEncoder()
        encoder1 = EncoderVertex(encoder, K, 'C_a', infoTail='in', parityTail='out')
        encoder2 = EncoderVertex(encoder, K, 'C_b', infoTail='in', parityTail='out')
        encoder3 = EncoderVertex(convolutional.TDInnerEncoder(),
                              K // 2, 'C_c',
                              infoTail='in', parityTail='out')
        innerMux = MuxVertex('alternating', 'mux_1')
        patchDemuxer = DeMuxVertex(np.array((1, 1, 0, 0, 0, 0, 0, 0)), 'p')
        finalMux = MuxVertex('sequential', 'mux_code')
        codeVertex = CodeVertex()
        vertices = [source, encoder1, encoder2, innerMux, patchDemuxer, encoder3, finalMux, codeVertex]
        # create Arcs
        source.connect(finalMux)
        source.connect(encoder1)
        source.connect(encoder2, interleaver=outerInterleaver)

        TurboArc(K, encoder1, innerMux)
        TurboArc(K, encoder2, innerMux)

        innerMux.connect(patchDemuxer)
        patchDemuxer.connect(finalMux)
        patchDemuxer.connect(encoder3, interleaver=innerInterleaver)
        TurboArc(K // 2, encoder3, finalMux)
        finalMux.connect(codeVertex)
        TurboLikeCode.__init__(self, vertices, name)

    def params(self):
        return OrderedDict([ ("K", self.infolength),
                             ("outerInterleaver", self.outerInterleaver),
                             ("innerInterleaver", self.innerInterleaver),
                             ("name", self.name)])


class RepeatAccumulateCode(TurboLikeCode):
    def __init__(self, repeat, interleaver, name):
        """Creates the RA(*repeat*) code defined by the given interleaver.

        *repeat* states how often to repeat each input bit. The input size of the code is then
        defined to be interleaver.size/repeat.
        """
        assert interleaver.size % repeat == 0
        infolength = interleaver.size // repeat
        blocklength = interleaver.size + 1 #one termination bit
        source = InformationSource(infolength)
        mux = MuxVertex('alternating', 'mux')
        for _ in range(repeat):
            source.connect(mux)
        encoder = convolutional.RepeatAccumulateEncoder()
        encoderVertex = EncoderVertex(encoder, blocklength - 1, 'RA-encoder')
        mux.connect(encoderVertex, interleaver)
        code = CodeVertex()
        TurboArc(blocklength, encoderVertex, code)
