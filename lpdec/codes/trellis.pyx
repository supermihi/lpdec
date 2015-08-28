# -*- coding: utf-8 -*-
# Copyright 2012-2015 Michael Helmling
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation

from __future__ import print_function

cimport cython
cimport cpython.ref
cimport numpy as np
import numpy as np
from libc.stdlib cimport calloc, malloc, realloc, free

INFO, PARITY = 0, 1
_INFO=0
_PARITY=1


cdef class Arc:
    """Data class representing an arc in a trellis (arcs are sometimes called branches or edges in
    literature).

    Parameters
    ----------
    tail : :class:`.Node`
        Tail (starting point) of the arc.
    head : :class:`.Node`
        Head (end point) of the arc.
    infobit : int
        Information bit (also called input bit, systematic bit) corresponding to this arc.
    parity : int
        Parity bit (also called output bit) corresponding to this arc.
    pos : int
        Position (time step) of this arc's tail in the trellis.
    state : int
        State of this arc's tail in the trellis.
    """
    def __init__(self, Node tail, Node head, int infobit, int parity, int pos, int state):
        self.tail = tail
        self.head = head
        self.infobit = infobit
        self.parity = parity
        self.pos = pos
        self.state = state
        self.cost = 0
        
    def remove(self):
        """Remove the arc from the trellis, updating head and tail."""
        setattr(self.tail, "outArc_{}".format(self.infobit), None)
        setattr(self.head, "inArc_{}".format(self.infobit), None)

    def __str__(self):
        return "{}/{}-arc({},{}->{},{})".format(self.infobit, self.parity,
                                                self.pos, self.state,
                                                self.head.pos, self.head.state)

    def __repr__(self):
        return "Arc({tl},{hd},{inf},{par},{p},{s})".format(tl=self.tail, hd=self.head,
                                                           inf=self.infobit, par=self.parity,
                                                           p=self.pos, s=self.state)


cdef class Node(object):
    """A trellis node (vertex). Has a (time-domain) position and a state."""
    def __cinit__(Node self, int pos, int state):
        self.pos = pos
        self.state = state     
        
    def outArcs(self):
        """Return a tuple of the (up to two) outgoing arcs of this node.
        
        It is guaranteed that the "zero" out arc appears first, if it exists."""
        return tuple(arc for arc in (self.outArc_0,self.outArc_1) if arc)
    
    def inArcs(self):
        """Return a tuple of the (up to two) ingoing arcs of this node.
        
        It is guaranteed that the "zero" in arc appears first, if it exists."""
        return tuple(arc for arc in (self.inArc_0, self.inArc_1) if arc)


@cython.final
cdef class Segment:
    """One segment (vertical slice) of a trellis.
    
    Consists of up to `numstates` nodes, some of which may be None. Can be used like a dict mapping
    state number to node. For fast access, use the C-level, e.g. ``<Node>(segment.nodes[i])``."""
    def __cinit__(self, Trellis trellis, int pos, int numstates):
        """Initialize the segment at *pos* in  *trellis* with *numstates* maximum states.
        
        Does not create any Node objects, but creates an empty array of size *numstates* to store
        them.
        """
        self.trellis = trellis
        self.pos = pos
        self.states = numstates
        self.fix_info = self.fix_parity = -1
        self.info_code_bit = self.parity_code_bit = -1
        self.nodes = <void **>calloc(numstates, sizeof(void*))
    
    cdef inline void setNode(Segment self, Node node, int state):
        cpython.ref.Py_INCREF(node) #  neede because next line circumveits ref counting
        self.nodes[state] = <void*>node
    
    def setCost(self, double infoCost, double parityCost):
        """Set the cost on the arcs in this segments.
        
        The cost of each :class:`Arc` ``arc`` in the segment will be calculated as ``arc.infobit``
        * `infoCost` + ``arc.parity`` * `parityCost`.

        Parameters
        ----------
        infoCost : double
            Cost associated with setting the information bit corresponding to this segment to ``1``.
        parityCost : double
            Cost associated with setting the parity bit corresponding to this segment to ``1``.
        """
        cdef:
            int i
            Node node
        for i in range(self.states):
            if self.nodes[i]:
                node = <Node>self.nodes[i]
                if node.outArc_0:
                    node.outArc_0.cost = node.outArc_0.parity*parityCost
                if node.outArc_1:
                    node.outArc_1.cost = infoCost + node.outArc_1.parity*parityCost
            
    def connectForward(self, Segment next, object encoder):
        """Creates outgoing arcs from *self* and target nodes in the next segment, if necessary.
        """
        cdef:
            int i
            Node node
            Arc arc
        for state in range(self.states):
            if self.nodes[state]:
                node = <Node>self.nodes[state]
                for inputValue in 0, 1:
                    nextState, parityOutput = encoder.stateTransition(state, inputValue)
                    if nextState not in next:
                        next.setNode(Node(self.pos+1, nextState), nextState)
                    arc = Arc(node, next[nextState], inputValue, parityOutput, self.pos, state)
                    setattr(next[nextState], 'inArc_{}'.format(inputValue), arc)
                    setattr(node, 'outArc_{}'.format(inputValue), arc)
    
    def connectBackward(self, Segment prev, object encoder):
        """Creates incoming arcs and source nodes in the previous segment, if necessary."""
        cdef:
            int i
            Node node
            Arc arc
        for state in range(self.states):
            if self.nodes[state]:
                node = <Node>self.nodes[state]
                for inputValue in 0, 1:
                    prevState, parityOutput = encoder.stateTransitionBack(state, inputValue)
                    if prevState not in prev:
                        prev.setNode(Node(self.pos-1, prevState), prevState)
                    if getattr(node, 'inArc_{}'.format(inputValue)) is None:
                        arc = Arc(prev[prevState], node,
                                  inputValue, parityOutput,
                                  self.pos-1, prevState)
                        setattr(prev[prevState], 'outArc_{}'.format(inputValue), arc)
                        setattr(node, 'inArc_{}'.format(inputValue), arc) 
    
    def allArcsOfInfoBit(self, int bit):
        """Generator that yields all arcs which have the given information bit (0 or 1)."""
        cdef:
            int i
            Node node
            Arc arc
        for i in range(self.states):
            if self.nodes[i]:
                node = <Node>self.nodes[i]
                arc = getattr(node, "outArc_{}".format(bit))
                if arc:
                    yield arc
                    
    def allArcsOfParity(self, int bit):
        """Yield all arcs which have the given parity bit (0 or 1)."""
        cdef:
            int i
            Node node
            Arc arc
        for i in range(self.states):
            if self.nodes[i]:
                node = <Node>self.nodes[i]
                for arc in node.outArcs():
                    if arc.parity == bit:
                        yield arc
 
    def __dealloc__(self):
        cdef Node node
        for i in range(self.states):
            if self.nodes[i]:
                node = <Node>(self.nodes[i])
                cpython.ref.Py_DECREF(node)
        free(self.nodes)
        
    def __getitem__(self, int state):
        if state < 0:
            state = self.states + state
        if state >= self.states:
            raise IndexError("segment index {} out of bounds".format(state))
        if self.nodes[state]:
            return <Node>self.nodes[state]
        else:
            raise IndexError("segment does not have a node in state {} ".format(state))
    
    def __setitem__(self, int key, Node value):
        if key < 0:
            key = self.states + key
        if key >= self.states:
            raise IndexError("segment index {} out of bounds".format(key))
        if self.nodes[key]:
            cpython.ref.Py_DECREF(<Node>self.nodes[key])
        self.setNode(value, key)
        
    def __delitem__(self, int key):
        if key < 0:
            key = self.states + key
        if key >= self.states:
            raise IndexError("segment index {} out of bounds".format(key))
        if self.nodes[key]:
            cpython.ref.Py_DECREF(<Node>self.nodes[key])
        self.nodes[key] = NULL
        
    def __contains__(self, int state):
        try:
            self[state]
            return True
        except IndexError:
            return False
    
    def __len__(self):
        cdef int i = 0, count = 0
        for i in range(self.states):
            if self.nodes[i]:
                count += 1
        return count
    
    def values(self):
        return ( <Node>self.nodes[i] for i in range(self.states) if self.nodes[i])
    
    def keys(self):
        return (i for i in range(self.states) if self.nodes[i])
    
    def items(self):
        return zip(self.keys(), self.values())
    
    def __str__(self):
        ret = "{0:5d}: ".format(self.pos)
        for k in range(self.states):
            if k in self:
                ret += "{0:5d}".format(k)
            else:
                ret += " "*5
        return ret

    def __repr__(self):
        return "Segment({},{})".format(self.pos, self.trellis.name)
            

cdef str tailSpecToStr(tailSpec spec):
    if spec == INPUT:
        return "in"
    elif spec == OUTPUT:
        return "out"
    else:
        return "drop"

cdef tailSpec strToTailSpec(str value):
    if value == "in":
        return INPUT
    elif value == "out":
        return OUTPUT
    elif value == "drop":
        return DROP
    else:
        raise ValueError("Invalid tailbit specifier: {}".format(value))


cdef class Trellis(object):
    """A trellis."""
    def __init__(Trellis self, object encoder, int length, int states=-1,
                 str infoTail="out", str parityTail="out", str name=None):
        object.__init__(self)
        self.segments = <void **>malloc((length+1)* sizeof(void*))
        self.length = length
        self.name = name
        if encoder is None:
            if states == -1:
                raise ValueError("either encoder or positive number of states must be given")
            self.states = states
        else:
            self.states = encoder.states
        
        cdef:
            Segment seg
            int i
        for i in range(length+1):
                seg  = Segment(self, i, self.states)
                cpython.ref.Py_INCREF(seg)
                self.segments[i] = <void*>seg
        if encoder is not None:
            if length < 2*encoder.tailbits:
                raise ValueError("Trellis must be at least twice as long as the number of tail bits")                    
            self.encoder = encoder
            self.tailbits = self.encoder.tailbits
            self.initVertex = Node(0, 0)
            self.setNode(self.initVertex, 0, 0)
            for i in range(length-self.tailbits):
                (<Segment>(self.segments[i])).connectForward(<Segment>(self.segments[i+1]),
                                                             self.encoder)
            self.endVertex = Node(length, 0)
            self.setNode(self.endVertex, length, 0)
            for i in range(length, length-self.tailbits, -1):
                (<Segment>(self.segments[i])).connectBackward(<Segment>(self.segments[i-1]),
                                                              self.encoder)
            self.infoTail, self.parityTail = infoTail, parityTail
            self.insize = length - (0 if self._infoTail == INPUT else self.tailbits)
            self.outsize = length + (self.tailbits if self._infoTail == OUTPUT else 0) \
                                  - (0 if self._parityTail == OUTPUT else self.tailbits)
    
    cdef inline void setNode(Trellis self, Node node, int pos, int state):
        (<Segment>self.segments[pos]).setNode(node, state)

    def inArcs(self, int pos):
        """Returns the arcs corresponding to the *pos*-th input bit.
        
        This function respects the infoTail and parityTail attributes."""
        if pos < self.length - self.tailbits:
            return self[pos].allArcsOfInfoBit(1)
        if self._infoTail == INPUT:
            return self[pos].allArcsOfInfoBit(1)
        else:
            raise ValueError('inArcs({0}) requested but trellis has only {1}'
                             .format(pos, self.length - self.tailbits))
        
    def outArcs(self, int pos):
        """Returs the arcs corresponding to the *pos*-th output bit.
        
        This function respects the infoTail and parityTail attributes."""
        if self._infoTail == OUTPUT:
            if pos < self.tailbits:
                return self[self.length - self.tailbits + pos].allArcsOfInfoBit(1)
            pos -= self.tailbits
        if self._parityTail != OUTPUT and pos >= self.length:
            raise ValueError('outArcs() index out of range')
        return self[pos].allArcsOfParity(1)
    
    cdef int segmentFromPosAndDirection(self, int pos, tailSpec direction):
        if direction == INPUT:
            return pos
        else:
            if self._infoTail == OUTPUT:
                if pos < self.tailbits:
                    return self.length - self.tailbits + pos
                pos -= self.tailbits
            return pos

    cdef int labelFromPosAndDirection(self, int pos, tailSpec direction):
        if direction == INPUT:
            return _INFO
        if self._infoTail == OUTPUT and pos < self.tailbits:
            return _INFO
        return _PARITY
    
    def segmentAndLabel(self, int i, str l):
        """Returns segment and label type corresponding to a specific bit in the input or output.
        
        The method returns a tuple (seg, lab) such that the arcs labeled *lab* (∈ {INFO, PARITY})
        in the *seg*-th trellis segment correspond to the *i*-th input bit (if l="in") or the
        *i*-th output bit (if l="out"), respectively, of the trellis viewed as an encoder."""
        return (self.segmentFromPosAndDirection(i, strToTailSpec(l)),
                self.labelFromPosAndDirection(i, strToTailSpec(l)))

    cpdef np.int_t[::1] encode(self, np.int_t[::1] path):
        """Encode the given *path*, return the corresponding parity output.
        
        The length of *path* must be at least self.insize; for convenience additional bits (e.g.
        tail bits) will be ignored instead of raising an error.
        """
        cdef:
            Node node = self.initVertex
            Arc arc
            np.ndarray[dtype=np.int_t,ndim=1] out = np.empty(self.outsize, dtype=np.int)
            int i, shift = self.tailbits if self._infoTail == OUTPUT else 0
        for i in range(self.length - self.tailbits):
            if path[i] == 0:
                arc = node.outArc_0
            else:
                arc = node.outArc_1
            out[shift + i] = arc.parity
            node = arc.head
        if not (self._infoTail == DROP and self._parityTail == DROP):
            for i in range(self.tailbits):
                arc = node.outArcs()[0]
                if self._infoTail == OUTPUT:
                    out[i] = arc.infobit
                if self._parityTail == OUTPUT:
                    out[self.length - self.tailbits + shift + i ] = arc.parity
                node = arc.head
        return out
    
    def allArcs(self):
        """Generator over all arcs in the trellis in canonical order.
        
        The arcs are yielded left to right, top-state to down-state, infobit 0 to infobit 1.
        """
        for segment in self:
            for node in segment.values():
                for arc in node.outArcs():
                    yield arc

    def clearArcCosts(self):
        cdef Arc arc 
        for arc in self.allArcs():
            arc.cost = 0
    
    property infoTail:
        """The info tail bit interpretation -- "in", "out" or "drop"."""
        def __get__(self):
            return tailSpecToStr(self._infoTail)
        
        def __set__(self, value):
            self._infoTail = strToTailSpec(value)
       
    property parityTail:
        """The parity tail bit interpretation -- "in", "out" or "drop"."""
        def __get__(self):
            return tailSpecToStr(self._parityTail)
        
        def __set__(self, value):
            self._parityTail = strToTailSpec(value)
        
    def __getitem__(self, index):
        if isinstance(index, slice):
            return tuple(<Segment>self.segments[i] for i in range(self.length+1)).__getitem__(index)
        if index < 0:
            index = self.length + 1 + index
        if index > self.length:
            raise IndexError("trellis index '{}' out of range".format(index))
        return <Segment>self.segments[index]
    
    def __dealloc__(self):
        cdef Segment segment
        for i in range(self.length+1):
            if self.segments[i]:
                segment = <Segment>self.segments[i]
                cpython.ref.Py_DECREF(segment)
        free(self.segments)
    
    def append(self, Segment seg):
        """Hacky method to allow appending a segment at the end."""
        self.length += 1
        self.segments = <void**>realloc(self.segments, (self.length+1) * sizeof(void*))
        cpython.ref.Py_INCREF(seg)
        self.segments[self.length]  = <void*>seg
    
    def __iter__(self):
        for i in range(self.length+1):
            yield self[i]
    
    def __len__(self):
        return self.length+1
    
    def __str__(self):
        return "\n\n".join( map (str, [self[i] for i in range(self.length+1)]))
    
    def __repr__(self):
        return "Trellis({}, {}, {}, {})".format(self.encoder, self.length, self.infoTail, self.parityTail)