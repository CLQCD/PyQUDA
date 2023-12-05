cdef class Pointer:
    cdef void *ptr
    cdef str dtype

    cdef set_ptr(self, void *ptr)

cdef class Pointers(Pointer):
    cdef unsigned int n1
    cdef void **ptrs

    cdef set_ptrs(self, void **ptrs)

cdef class Pointerss(Pointer):
    cdef unsigned int n1, n2
    cdef void ***ptrss

    cdef set_ptrss(self, void ***ptrss)
