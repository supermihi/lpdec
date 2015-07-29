cimport numpy as np

cdef struct ConnectInfo:
    void *vertex
    int *outPositions
    int *inPositions
    int numWires


cdef class TurboVertex:
    cdef:
        public list inArcs, outArcs
        public str name
        int _outSize
        int _inSize
        public np.int_t[::1] _inWord, _outWord
        ConnectInfo *_connects
        int _numConnects
        public object cplexID # used by some decoders
    cdef void finalize(self)


cdef class InformationSource(TurboVertex):
    cdef:
        public int infobits
        int *_numTrellises
        void ***_trellises
        int **_segments
    cdef void finalize(self)


cdef class CodeVertex(TurboVertex):
    cdef:
        void **_trellises
        int *_labels
        int *_segments
    cdef void finalize(self)


cdef class TurboLikePrivate:
    cdef:
        void **_stoppers
        public int _numStoppers
        public CodeVertex codeVertex
        public InformationSource infoVertex