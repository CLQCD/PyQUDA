cdef extern from "qcu.h":
    ctypedef struct QcuParam:
        int lattice_size[4]

    void dslashQcu(void *fermion_out, void *fermion_in, void *gauge, QcuParam *param, int parity)