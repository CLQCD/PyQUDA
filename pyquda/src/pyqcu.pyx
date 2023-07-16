import numpy
cimport numpy


cdef extern from "qcu.h":
    void my_dslash_interface(void *U_ptr, void *a_ptr, void *b_ptr, int Lx, int Ly, int Lz, int Lt, int Nd, int Ns, int Nc)


def my_dslash(numpy.ndarray[numpy.complex128_t, ndim=7] U, numpy.ndarray[numpy.complex128_t, ndim=6] a, numpy.ndarray[numpy.complex128_t,ndim=6]b, int Lx,int Ly, int Lz,int Lt,int Nd, int Ns, int Nc):
    cdef size_t ptr_uint64
    ptr_uint64 = U.ctypes.data
    cdef void *U_ptr = <void *>ptr_uint64
    ptr_uint64 = a.ctypes.data
    cdef void *a_ptr = <void *>ptr_uint64
    ptr_uint64 = b.ctypes.data
    cdef void *b_ptr = <void *>ptr_uint64
    my_dslash_interface(U_ptr, a_ptr, b_ptr, Lx, Ly, Lz, Lt, Nd, Ns, Nc)
