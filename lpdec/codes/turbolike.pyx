# cython: boundscheck=False
# cython: nonecheck=False
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

from collections import OrderedDict, defaultdict

import numpy as np
cimport numpy as np
import cython
cimport cython
from lpdec.codes import interleaver, convolutional, BinaryLinearBlockCode
from lpdec.codes cimport trellis
from lpdec.codes.trellis cimport _INFO, _PARITY, OUTPUT, Trellis, Segment
from libc.stdlib cimport malloc, free


cdef class TurboVertex:
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
        self.name = ""
        self._inWord = None
        self._outWord = None

    def addInArc(self, arc):
        self.inArcs.append(arc)
        
    def addOutArc(self, arc):
        self.outArcs.append(arc)

    def connect(self, target, size, interleaver=None):
        """Connect *size* bits of this vertex' output to the input of *target*.

        Creates an InterleaverArc if an interleaver is specified, or an ordinary TurboArc
        otherwise.
        """
        if interleaver is None:
            TurboArc(size, self, target)
        else:
            InterleaverArc(interleaver, self, target)

    cdef void finalize(self):
        self._outSize = self.outSize
        self._inSize = self.inSize
        targets = {}
        for arc in self.outArcs:
            for i in range(arc.size):
                vertex, pos = arc.endOfPath(i)
                if vertex not in targets:
                    targets[vertex] = []
                targets[vertex].append( (i, pos) )
        self._connects = <ConnectInfo *>malloc(len(targets) * sizeof(ConnectInfo))
        self._numConnects = len(targets)
        cdef:
            ConnectInfo info
        for i, (vertex, wires) in enumerate(targets.items()):
            info = ConnectInfo(<void*>vertex, <int*>malloc(len(wires)*sizeof(int)),
                               <int*>malloc(len(wires)*sizeof(int)),
                               len(wires))
            self._connects[i] = info
            for j, (outPos, inPos) in enumerate(wires):
                info.outPositions[j] = outPos
                info.inPositions[j] = inPos
    

cdef class InformationSource(TurboVertex):
    """The vertex representing the information source in a turbo-like code. Each
    code must have exactly one information source.

    The information source can be connected to several vertices, which is interpreted
    such that each connection carries a complete copy of the information input.
    """

    def __init__(self, int infobits):
        """Create the source vertex with information length *infobits*."""
        TurboVertex.__init__(self)
        self.infobits = infobits

    def connect(self, target, interleaver=None):
        """Convenience function: Automatically determine the size"""
        TurboVertex.connect(self, target, self.infobits, interleaver)

    cdef void finalize(self):
        TurboVertex.finalize(self)
        self._numTrellises = <int*>malloc(sizeof(int) * self.infobits)
        self._trellises = <void***>malloc(sizeof(void**) * self.infobits)
        self._segments = <int**>malloc(sizeof(int**) * self.infobits)
        for i in range(self.infobits):
            ends = [ arc.endOfPath(i)
                        for arc in self.outArcs
                        if isinstance(arc.endOfPath(i)[0], EncoderVertex ) ]
            self._numTrellises[i] = len(ends)
            self._trellises[i] = <void**>malloc(sizeof(void*) * len(ends))
            self._segments[i] = <int*>malloc(sizeof(int) * len(ends))
            for j, (vertex, segment) in enumerate(ends):
                vertex.isInfoConnected = True
                self._trellises[i][j] = <void*>vertex.trellis
                self._segments[i][j] = segment

    def __repr__(self):
        return 'InformationSource({0})'.format(self.infobits)


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
        self._inWord = np.empty(self.trellis.insize, dtype=np.int)
        self.name = name
        self.isInfoConnected = False

    def connect(self, target):
        """Convenience function: Automatically sets *size*."""
        TurboVertex.connect(self, target, self.trellis.outsize)

    def __repr__(self):
        return 'EncoderVertex(name={0})'.format(self.name)


class MuxVertex(TurboVertex):
    """A MuxVertex takes several inputs and muxes them together into one output.
    There are two supported muxing schemes:
      - sequential: The bits of the inputs are sequentially concatenated
      - alternating: The inputs are "interleaved" into the output.
    Example: Let there be three inputs of length four each, so we have the bits
    from input a: a1 a2 a3 a4
    from input b: b1 b2 b3 b4
    from input c: c1 c2 c3 c4
    Sequential muxing outputs a1 a2 a3 a4 b1 b2 b3 b4 c1 c2 c3 c4, while alternating
    muxing yields a1 b1 c1 a2 b2 c2 a3 b3 c3 a4 b4 c4.
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
        if self.style == "alternating":
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
        
        if self.style == "alternating":
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
        return 'MuxVertex(name = {0})'.format(self.name)


class DeMuxVertex(TurboVertex):
    """Takes a single input and splits its bits into  a number of output arcs.

    DeMuxVertex acts as a counterpart of a MuxVertex. Which bit goes to which output is determined
    by a pattern, which is a tuple of output positions. For instance, the pattern (0,1,1) means
    that every third bit goes to the first output and the rest go to the second output. More
    formally, if p_1, ..., p_k is a pattern, then the i-th input bit goes to output p_{i mod k}.
    Note how this supports outputs of different length.
    """
    
    def __init__(self, np.ndarray[ndim=1, dtype=np.int_t] pattern, name=None):
        """Creates the DeMuxVertex; see the class doc on how to specify the pattern."""
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
        """Connects the demuxer to the *target* vertex, optionally using an interleaver.

        This method has to be called after the input connection has been made, and each call
        corresponds to the next symbol of the pattern. The size is determined automatically
        from the insize and the pattern.
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
        return 'DeMuxVertex(name = {0})'.format(self.name)


cdef class CodeVertex(TurboVertex):
    """A CodeVertex represents the codeword of a turbo-like code.

    It has a single incoming arc, the bits of which comprise the codeword."""

    isStopper = True

    cdef void finalize(self):
        TurboVertex.finalize(self)
        cdef:
            Trellis trellis
        blocklength = self.inSize
        self._trellises = <void**>malloc(sizeof(void*) * blocklength)
        self._labels = <int*>malloc(sizeof(int) * blocklength)
        self._segments = <int*>malloc(sizeof(int) * blocklength)
        for i in range(blocklength):
            vertex, pos = self.inArcs[0].startOfPath(i)
            if isinstance(vertex, InformationSource):
                self._trellises[i] = NULL
                self._segments[i] = pos
            else:
                trellis = <Trellis>vertex.trellis
                self._trellises[i] = <void*>vertex.trellis
                self._segments[i] = trellis.segmentFromPosAndDirection(pos, OUTPUT)
                self._labels[i] = trellis.labelFromPosAndDirection(pos, OUTPUT)

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
        self.tail.addOutArc(self)
        self.head.addInArc(self)

    def endPosition(self, int start):
        """Returns the position to which *start*-th input bit is wired.

        This is the identity mapping for non-interleaver arcs."""
        # pylint: disable=R0201
        return start

    def startPosition(self, int end):
        """Returns the position which is wired to the *end*-th bit in the output.

        This is the identity mapping for non-interleaver arcs."""
        # pylint: disable=R0201
        return end

    def endOfPath(self, int i):
        """Compute where the *i*-th bit on this arc encounters the next stopper vertex.

        Returns a tuple (vertex, pos) such that the *i*-th bit of this arc is wired to the *pos*-th
        input bit of *vertex*, which is a stopper vertex (encoder or codeword). In other words,
        this method identifiers the terminal position of the *i*-th bit on its path through muxers
        and demuxers.
        """
        if i < 0 or i >= self.size:
            raise ValueError('value {0} out of range for arc {1}'.format(i, self))
        if self.head.isStopper:
            return self.head, self.endPosition(i)
        else:
            arc, bit = self.head.outPosition(self, i)
            return arc.endOfPath(bit)

    def startOfPath(self, int i):
        """Compute where the *i*-th bit of this arc originates at an encoder or InformationSource.

        This is complementary to endOfPath: Returns a tuple (vertex, pos) such that the bit at
        position *i* at the /end/ of this arc originates from *vertex* (encoder or
        InformationSource) at output position *pos*.
        """
        if i < 0 or i >= self.size:
            raise ValueError('value {0} out of range for arc {1}'.format(i, self))
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

    def endPosition(self, int start):
        return self.interleaver(start)

    def startPosition(self, int end):
        return self.interleaver.inv(end)


cdef class TurboLikePrivate:
     pass


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
        cdef:
            TurboLikePrivate p = TurboLikePrivate()
            int i
            TurboVertex _vertex
        self.p = p

        p.codeVertex = p.infoVertex = None
        for vertex in vertices:
            if isinstance(vertex, InformationSource):
                assert p.infoVertex is None
                p.infoVertex = vertex
            elif isinstance(vertex, CodeVertex):
                assert p.codeVertex is None
                p.codeVertex = vertex
            elif isinstance(vertex, EncoderVertex):
                encoders.append(vertex)
        self.vertices = vertices
        self.encoders = encoders
        self.blocklength = p.codeVertex.inSize
        self.infolength = p.infoVertex.infobits
        p.codeVertex._inWord = np.empty(self.blocklength, dtype=np.int)
        BinaryLinearBlockCode.__init__(self, name=name)
        stoppers = [p.infoVertex] + self.encoders + [p.codeVertex]
        p._numStoppers = len(stoppers)
        p._stoppers = <void**>malloc(sizeof(void*) * p._numStoppers)
        for i, _vertex in enumerate(stoppers):
            p._stoppers[i] = <void*>_vertex
            _vertex.finalize()
        for i in range(self.blocklength):
            sals = self.segmentsForCodeBit(i)
            for seg, lab in sals:
                if lab == _INFO:
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

    def encode(self, np.int_t[::1] infoword):
        cdef:
            TurboVertex vertex, targetVertex
            TurboLikePrivate p = self.p
            ConnectInfo info
            int i, j, k
            np.int_t[:] outWord, inWord
        self.p.infoVertex._outWord = infoword
        for i in range(p._numStoppers):
            vertex = <TurboVertex>(p._stoppers[i])
            if isinstance(vertex, EncoderVertex):
                vertex._outWord = vertex.trellis.encode(vertex._inWord)
            outWord = vertex._outWord
            for j in range(vertex._numConnects):
                info = vertex._connects[j]
                targetVertex = <TurboVertex>info.vertex
                inWord = targetVertex._inWord
                for k in range(info.numWires):
                    inWord[info.inPositions[k]] = outWord[info.outPositions[k]]
        return vertex._inWord.copy()
    
    def encodePath(self, np.int_t[::1] path, encoder):
        """Try to encode the given *path* in the trellis of EncoderVertex *encoder* to a codeword.

        This will only work if the encoder is directly connected to the information source.
        """
        cdef:
            np.ndarray[ndim=1, dtype=np.int_t] infoWord = np.empty(self.infolength, dtype=np.int)
            int i
        if encoder.inArcs[0].tail != self.p.infoVertex:
            raise ValueError('invalid encoder for TurboLikeCode.encode()')
        for i in range(self.infolength):
            infoWord[i] = path[encoder.inArcs[0].endPosition(i)]
        return self.encode(infoWord)

    def fixCodeBit(self, int i, int val):
        cdef:
            Trellis trellis
            position = self.p.codeVertex._segments[i]
            label = self.p.codeVertex._labels[i]
            Segment segment
            CodeVertex codeVertex = self.p.codeVertex
            int j, numInfos
        if codeVertex._trellises[i] is NULL:
            numInfos = self.infoVertex._numTrellises[position]
            for j in range(numInfos):
                trellis = <Trellis>self.p.infoVertex._trellises[position][j]
                segment = <Segment>trellis.segments[self.infoVertex._segments[position][j]]
                segment.fix_info = val
        else:
            trellis = <Trellis>self.p.codeVertex._trellises[i]
            segment = <Segment>trellis.segments[position]
            if label == _INFO:
                segment.fix_info = val
            else:
                segment.fix_parity = val
    
    def segmentsForCodeBit(self, i):
        cdef:
            Trellis trellis
            CodeVertex codeVertex = self.p.codeVertex
            position = codeVertex._segments[i]
            label = codeVertex._labels[i]
            InformationSource infoVertex = self.p.infoVertex
            Segment segment

            int j, numInfos
        if codeVertex._trellises[i] is NULL:
            numInfos = infoVertex._numTrellises[position]
            ret = []
            for j in range(numInfos):
                trellis = <Trellis>infoVertex._trellises[position][j]
                segment = <Segment>trellis.segments[infoVertex._segments[position][j]]
                ret.append( (segment, _INFO) )
            return ret
        else:
            trellis = <Trellis>codeVertex._trellises[i]
            segment = <Segment>trellis.segments[position]
            return ( (segment, label), )
    
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
        start, pos = self.p.codeVertex.inArcs[0].startOfPath(i)
        outList = []
        if start is self.p.infoVertex:
            for out in start.outArcs:
                end, endPos = out.endOfPath(pos)
                if isinstance(end, EncoderVertex):
                    outList.append(end.trellis.inArcs(endPos))
        else:
            outList.append(start.trellis.outArcs(pos))
        return outList

    def prepareConstraintsData(self):
        """Prepares data structures for algorithms that employ the equality-type constraints.
        """
        labelStr = { _INFO: 'info', _PARITY: 'parity'}
        constraints = self.equalityPairs()
        for i, ((trellis1, segment1, label1), (trellis2, segment2, label2)) \
                in enumerate(constraints):
            setattr(trellis1[segment1], "g_{0}_index".format(labelStr[label1]), i)
            setattr(trellis1[segment1], "g_{0}".format(labelStr[label1]), 1)
            
            setattr(trellis2[segment2], "g_{0}_index".format(labelStr[label2]), i)
            setattr(trellis2[segment2], "g_{0}".format(labelStr[label2]), -1)   
        return constraints

    def equalityPairs(self):
        """Returns pairs of trellis segments which carry the same codeword bits.

        The result is a list of pairs ( (t1, s1, b1), (t2, s2, b2) ), meaning that the flow on the
        *b1*-arcs of segment *s1* in trellis *t1* must equal the flow on the *b2*-arcs on segment
        *s2* in trellis *t2*, where
            - *b1* and *b2* are one of(trellis.INFO, trellis.PARITY),
            - *s1* and *s2* are segment indexes, and
            - *t1* and *t2* are trellis.Trellis instances."""
        pairs = []
        #  first we take care of the info source vertex and encoders connected to it
        equalInfoArcs = []
        for arc in self.p.infoVertex.outArcs:
            stopper = arc.endOfPath(0)[0]
            if isinstance(stopper, EncoderVertex):
                equalInfoArcs.append(arc)
        referenceArc = equalInfoArcs[0]
        for i in range(referenceArc.size):
            refEncoder, refPosition = referenceArc.endOfPath(i)
            refSegment, refLabel = refEncoder.trellis.segmentAndLabel(refPosition, "in")
            for arc in equalInfoArcs[1:]:
                otherEncoder, otherPosition = arc.endOfPath(i)
                otherSegment, otherLabel = otherEncoder.trellis.segmentAndLabel(otherPosition,
                                                                                "in")
                pairs.append(((refEncoder.trellis, refSegment, refLabel),
                               (otherEncoder.trellis, otherSegment, otherLabel)))
        #  now take care of encoder-to-encoder connections, like in 3-D turbo codes
        for encoder in self.encoders:
            for i in range(encoder.outArcs[0].size):
                refPosition, refLabel = encoder.trellis.segmentAndLabel(i, "out")
                endVertex, endPos = encoder.outArcs[0].endOfPath(i)
                if isinstance(endVertex, EncoderVertex):
                    endSegment, endLabel = endVertex.trellis.segmentAndLabel(endPos, "in")
                    pairs.append(((encoder.trellis, i, trellis._PARITY),
                                   (endVertex.trellis, endSegment, endLabel)))
        return pairs

    def setCost(self, np.double_t[::1] costVector):
        """Sets arc costs on all involved trellis graphs according to the given cost vector.

        The size of *costVector* must equal the blocklength of the code. The method determines
        which arcs are affected by which costs by calling trellisSegmentsOfOutBit for each
        component of the cost vector.
        """
        cdef:
            int i
            double val
        for vertex in self.encoders:
            vertex.trellis.clearArcCosts()
        for i, val in enumerate(costVector):
            listList = self.trellisSegmentsOfOutBit(i)
            for liste in listList:
                for arc in liste:
                    arc.cost += (val / len(listList))

    def constraints(self):
        """Returns the constraint set Ax=b of the corresponding LP.

        Returns a 3-tuple (A,b,i), where i is the index of the first row corresponding
        to equality constraints.
        """
        # TODO: so far only equality constraints corresponding to arcs originating in the source
        # node are considered --> does not work for e.g. 3-D turbo codes.
        # TODO: code reuse: use equalityPairs()   
        nodes = OrderedDict()
        arcs = OrderedDict()
        for vertex in (v for v in self.vertices if isinstance(v, EncoderVertex)):
            trellis = vertex.trellis
            for segment in trellis:
                for _, node in sorted(segment.items()):
                    nodes[node] = len(nodes)
                    for _, arc in sorted(node.outs.items()):
                        arcs[arc] = len(arcs)

        equalInputEncoders = []
        for arc in self.p.infoVertex.outArcs:
            if isinstance(arc.head, EncoderVertex):
                equalInputEncoders.append((arc, arc.head))

        naim = np.zeros((len(nodes) + (len(equalInputEncoders) - 1) * self.infolength, len(arcs)),
                        dtype=np.int)
        rhs = np.zeros((naim.shape[0], 1), dtype=np.int)
        for arc, column in arcs.items():
            naim[nodes[arc.tail], column] = 1
            naim[nodes[arc.head], column] = -1

        for node, i in nodes.items():
            if len(node.outs) == 0:
                rhs[i] = -1
            elif len(node.ins) == 0:
                rhs[i] = 1
        row = len(nodes)
        for i in range(len(equalInputEncoders) - 1):
            (arc1, encoder1), (arc2, encoder2) = equalInputEncoders[i:i + 2]
            for index in range(arc1.size):
                for arc in encoder1.trellis[arc1.endPosition(index)].allArcsOfInfoBit(1):
                    naim[row, arcs[arc]] = 1
                for arc in encoder2.trellis[arc2.endPosition(index)].allArcsOfInfoBit(1):
                    naim[row, arcs[arc]] = -1
                row += 1
        return naim, rhs, len(nodes)

    def params(self):
        raise NotImplementedError()


class StandardTurboCode(TurboLikeCode):
    """
    A conventional parallel concatenated turbo code.

    The code follows following encoding scheme:
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
            name = "StandardTurbo({0}".format(interleaver.size)
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
