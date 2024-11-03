from libc.stdlib cimport malloc, free

cdef class Pointer:
    def __cinit__(self, str dtype, *args):
        self.dtype = dtype
        self.ptr = NULL

    cdef set_ptr(self, void *ptr):
        self.ptr = ptr

    def __repr__(self) -> str:
        return f"Pointer at 0x{<size_t>self.ptr:016x}"

cdef class Pointers(Pointer):
    def __cinit__(self, str dtype, unsigned int n1):
        self.n1 = n1
        if n1 > 0:
            self.ptrs = <void **>malloc(n1 * sizeof(void *))
            for i in range(n1):
                self.ptrs[i] = NULL
        else:
            self.ptrs = <void **>NULL

    def __dealloc__(self):
        if self.ptrs:
            free(self.ptrs)

    cdef set_ptrs(self, void **ptrs):
        for i in range(self.n1):
            self.ptrs[i] = ptrs[i]
        self.ptr = <void *>self.ptrs

cdef class Pointerss(Pointer):
    def __cinit__(self, str dtype, unsigned int n1, unsigned int n2):
        self.n1 = n1
        self.n2 = n2
        if n1 > 0 and n2 > 0:
            self.ptrss = <void ***>malloc(n1 * sizeof(void **))
            for i in range(n1):
                self.ptrss[i] = <void **>malloc(n2 * sizeof(void *))
                for j in range(n2):
                    self.ptrss[i][j] = NULL
        else:
            self.ptrss = <void ***>NULL

    def __dealloc__(self):
        if self.ptrss:
            for i in range(self.n1):
                free(self.ptrss[i])
            free(self.ptrss)

    cdef set_ptrss(self, void ***ptrss):
        for i in range(self.n1):
            for j in range(self.n2):
                self.ptrss[i][j] = ptrss[i][j]
        self.ptr = <void *>self.ptrss

def ndarrayPointer(ndarray, as_void=False):
    ndarray_type = ".".join([type(ndarray).__module__, type(ndarray).__name__])
    if ndarray_type == "numpy.ndarray":
        gpu = None
        dtype = ndarray.dtype.str
    elif ndarray_type == "cupy.ndarray":
        gpu = "cupy"
        dtype = ndarray.dtype.str
    elif ndarray_type == "torch.Tensor":
        gpu = "torch"
        dtype = f"<{'c' if ndarray.dtype.is_complex else 'f' if ndarray.dtype.is_floating_point else 'i' if ndarray.dtype.is_signed else 'u'}{ndarray.dtype.itemsize}"
    else:
        raise TypeError(f"ndarrayPointer: ndarray has unsupported type={type(ndarray)}")

    if not as_void:
        if dtype == "<i4":
            dtype = "int"
        elif dtype == "<f8":
            dtype = "double"
        elif dtype == "<c16":
            dtype = "double_complex"
        else:
            raise TypeError(f"ndarrayPointer: ndarray has unsupported dtype={dtype}")
    else:
        dtype = "void"

    shape = ndarray.shape
    ndim = ndarray.ndim
    cdef size_t ptr_uint64
    cdef void **ptrs
    cdef void ***ptrss
    if ndim == 1:
        ptr1 = Pointer(dtype)
        if gpu == "cupy":
            ptr_uint64 = ndarray.data.ptr
        elif gpu == "torch":
            ptr_uint64 = ndarray.data_ptr()
        else:
            ptr_uint64 = ndarray.ctypes.data
        ptr1.set_ptr(<void *>ptr_uint64)
        return ptr1
    elif ndim == 2:
        ptr2 = Pointers(dtype, shape[0])
        ptrs = <void **>malloc(shape[0] * sizeof(void *))
        if gpu == "cupy":
            for i in range(shape[0]):
                ptr_uint64 = ndarray[i].data.ptr
                ptrs[i] = <void *>ptr_uint64
        elif gpu == "torch":
            for i in range(shape[0]):
                ptr_uint64 = ndarray[i].data_ptr()
                ptrs[i] = <void *>ptr_uint64
        else:
            for i in range(shape[0]):
                ptr_uint64 = ndarray[i].ctypes.data
                ptrs[i] = <void *>ptr_uint64
        ptr2.set_ptrs(ptrs)
        free(ptrs)
        return ptr2
    elif ndim == 3:
        ptr3 = Pointerss(dtype, shape[0], shape[1])
        ptrss = <void ***>malloc(shape[0] * sizeof(void **))
        for i in range(shape[0]):
            ptrss[i] = <void **>malloc(shape[1] * sizeof(void *))
        if gpu == "cupy":
            for i in range(shape[0]):
                for j in range(shape[1]):
                    ptr_uint64 = ndarray[i, j].data.ptr
                    ptrss[i][j] = <void *>ptr_uint64
        elif gpu == "torch":
            for i in range(shape[0]):
                for j in range(shape[1]):
                    ptr_uint64 = ndarray[i, j].data_ptr()
                    ptrss[i][j] = <void *>ptr_uint64
        else:
            for i in range(shape[0]):
                for j in range(shape[1]):
                    ptr_uint64 = ndarray[i, j].ctypes.data
                    ptrss[i][j] = <void *>ptr_uint64
        ptr3.set_ptrss(ptrss)
        for i in range(shape[0]):
            free(ptrss[i])
        free(ptrss)
        return ptr3
    else:
        raise NotImplementedError("ndarray.ndim > 3 not implemented yet")

ndarrayDataPointer = ndarrayPointer
