from lpdec.decoders.base cimport Decoder
from lpdec.decoders.branchcut.node cimport Node
from lpdec.persistence cimport JSONDecodable

cdef class BranchResult:

    cdef int index, childInfeasible, canPrune


cdef class BranchingRule(JSONDecodable):

    cdef:
        object code
        Decoder bcDecoder
        int lamb
        public double ub
        double mu
        Node node
        public int index
        public bint canPrune
        int[:] candInds

    cdef int[:] candidates(self)
    cdef double calculateScore(self, int index)
    cdef int computeBranchIndex(self, Node node, double ub, double[::1] solution) except -1
    cdef double score(self, double qminus, double qplus)
    cdef int beginScoreComputation(self) except -1
    cdef int endScoreComputation(self) except -1
    cdef int callback(self, Node node) except -1
    cdef int rootCallback(self, int, int) except -1
    cdef int reset(self) except -1