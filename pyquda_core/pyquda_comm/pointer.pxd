cdef class Pointer:
    cdef void *ptr
    cdef str dtype

    cdef set_ptr(self, void *ptr)

cdef class Pointers(Pointer):
    cdef unsigned int n0
    cdef void **ptrs

    cdef set_ptrs(self, void **ptrs)

cdef class Pointerss(Pointer):
    cdef unsigned int n0, n1
    cdef void ***ptrss

    cdef set_ptrss(self, void ***ptrss)

cdef class _NDArray:
    cdef unsigned int n0, n1
    cdef void *ptr
    cdef void **ptrs
    cdef void ***ptrss
