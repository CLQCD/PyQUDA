cimport qcu
include "pointer.pxi"

cdef class QcuParam:
    cdef qcu.QcuParam param

    def __init__(self):
        pass

    @property
    def lattice_size(self):
        return self.param.lattice_size

    @lattice_size.setter
    def lattice_size(self, value):
        self.param.lattice_size = value

def dslashQcu(Pointer fermion_out, Pointer fermion_in, Pointer gauge, QcuParam param, int parity):
    qcu.dslashQcu(fermion_out.ptr, fermion_in.ptr, gauge.ptr, &param.param, parity)
