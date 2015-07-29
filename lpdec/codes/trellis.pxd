cimport numpy as np

cdef enum tailSpec:
    INPUT
    OUTPUT
    DROP

cdef int _INFO = 0, _PARITY = 1

cdef class Node #forward
cdef class Trellis

cdef class Arc:
    cdef readonly Node tail, head
    cdef readonly int infobit, parity
    cdef readonly int pos, state
    
    # for shortest path algorithms
    cdef public double cost
    cdef public double orig_cost
    cdef public object kpaths_candidate
    
    # for LP decoders
    cdef public object lp_var

cdef class Node(object):
    cdef readonly int pos
    cdef readonly int state
    
    cdef public Arc outArc_0, outArc_1
    cdef public Arc inArc_0, inArc_1
    
    # for (k-)shortest path algorithms
    cdef public Arc pred_arc
    cdef public object kpaths
    cdef public object binaryState
    cdef public double path_cost


cdef class Segment:

    cdef readonly int pos         # time-axis position (starting by 0)
    cdef readonly int states      # number of states
    cdef readonly Trellis trellis # Trellis object this segment belongs to
    
    cdef public int fix_info      # value to which info bit should be fixed on this
                                  # segment; -1 (the default value) means no fixing
    cdef public int fix_parity    # same as above but for the parity bit
    

    # attributes used for constrained shortest paths problems
    cdef public double g_info, g_parity
    cdef public int g_info_index, g_parity_index
    
    cdef public int info_code_bit
    cdef public int parity_code_bit
    cdef public double info_code_ratio
    cdef public double parity_code_ratio
    
    cdef void **nodes             # array of pointers to the segment's Node objects
    
    cdef void setNode(self, Node node, int state)
     

cdef class Trellis:

    cdef readonly int length
    cdef public int states
    cdef readonly int tailbits
    cdef readonly int insize
    cdef readonly int outsize
    cdef public str name
    cdef void **segments  # array of pointers to Segment objects
    cdef tailSpec _parityTail
    cdef tailSpec _infoTail
    cdef public Node initVertex, endVertex
    cdef void setNode(self, Node node, int pos, int state)
    cdef int segmentFromPosAndDirection(self, int, tailSpec)
    cdef int labelFromPosAndDirection(self, int, tailSpec)
    cdef object encoder
    cpdef np.int_t[::1] encode(self, np.int_t[::1] input)