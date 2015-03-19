from lpdec.decoders.base cimport Decoder
from lpdec.decoders.branchcut.node cimport Node


cdef class BranchingRule:

    cdef:
        object code
        Decoder bcDecoder
        int lamb
        double ub, mu
        Node node
        int[:] candInds

    cdef int[:] candidates(self)
    cdef double calculateScore(self, int index)
    cdef int branchIndex(self, Node node, double ub, double[::1] solution) except -1
    cdef double score(self, double qminus, double qplus)
    cdef int beginScoreComputation(self) except -1
    cdef int endScoreComputation(self) except -1
    cdef int callback(self, Node node) except -1
    cdef int reset(self) except -1