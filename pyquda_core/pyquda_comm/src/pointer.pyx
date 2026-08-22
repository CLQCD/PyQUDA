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
    def __cinit__(self, str dtype, unsigned int n0):
        self.n0 = n0
        if n0 > 0:
            self.ptrs = <void **>malloc(n0 * sizeof(void *))
            for i in range(n0):
                self.ptrs[i] = <void *>NULL
        else:
            self.ptrs = <void **>NULL
        self.ptr = <void *>self.ptrs

    def __dealloc__(self):
        if self.ptrs:
            free(self.ptrs)

    cdef set_ptrs(self, void **ptrs):
        for i in range(self.n0):
            self.ptrs[i] = ptrs[i]

cdef class Pointerss(Pointer):
    def __cinit__(self, str dtype, unsigned int n0, unsigned int n1):
        self.n0 = n0
        self.n1 = n1
        if n0 > 0 and n1 > 0:
            self.ptrss = <void ***>malloc(n0 * sizeof(void **))
            for i in range(n0):
                self.ptrss[i] = <void **>malloc(n1 * sizeof(void *))
                for j in range(n1):
                    self.ptrss[i][j] = <void *>NULL
        else:
            self.ptrss = <void ***>NULL
        self.ptr = <void *>self.ptrss

    def __dealloc__(self):
        if self.ptrss:
            for i in range(self.n0):
                free(self.ptrss[i])
            free(self.ptrss)

    cdef set_ptrss(self, void ***ptrss):
        for i in range(self.n0):
            for j in range(self.n1):
                self.ptrss[i][j] = ptrss[i][j]

def _data_ptr(_data, _backend):
    if _backend == "numpy":
        assert _data.flags["C_CONTIGUOUS"]
        return _data.ctypes.data
    elif _backend == "cupy":
        assert _data.flags["C_CONTIGUOUS"]
        return _data.data.ptr
    elif _backend == "dpnp":
        assert _data.flags["C_CONTIGUOUS"]
        return _data.data.ptr
        # return _data._array_obj._pointer  # ? dpnp<0.19.0
    elif _backend == "torch":
        assert _data.is_contiguous()
        return _data.data_ptr()

cdef class _NDArray:
    def __cinit__(self, ndarray, int ndim = 0):
        ndarray_type = ".".join([type(ndarray).__module__, type(ndarray).__name__])
        if ndarray_type == "numpy.ndarray":
            backend = "numpy"
            dtype = ndarray.dtype.str
        elif ndarray_type == "cupy.ndarray":
            backend = "cupy"
            dtype = ndarray.dtype.str
        elif ndarray_type == "dpnp.dpnp_array.dpnp_array":
            backend = "dpnp"
            dtype = ndarray.dtype.str
        elif ndarray_type == "torch.Tensor":
            backend = "torch"
            dtype = f"<{'c' if ndarray.dtype.is_complex else 'f' if ndarray.dtype.is_floating_point else 'i' if ndarray.dtype.is_signed else 'u'}{ndarray.dtype.itemsize}"
        else:
            raise TypeError(f"_NDArray: ndarray has unsupported type={type(ndarray)}")

        if ndim == 0:
            ndim = ndarray.ndim
            # if backend != "numpy":
            #     raise ValueError(f"_NDArray: typed ndarray has unexpected type={type(ndarray)}")
        elif ndim != ndarray.ndim:
            raise ValueError(f"_NDArray: untyped ndarray has unexpected ndim={ndarray.ndim}")
        elif dtype != "<i4" and dtype != "<f8" and dtype != "<c16":
            raise ValueError(f"_NDArray: untyped ndarray has unexpected dtype={dtype}")
        shape = ndarray.shape
        cdef size_t ptr_uint64
        if ndim == 1:
            self.n0, self.n1 = 0, 0
            if shape[0] > 0:
                ptr_uint64 = _data_ptr(ndarray, backend)
                self.ptr = <void *>ptr_uint64
            else:
                self.ptr = <void *>NULL
        elif ndim == 2:
            self.n0, self.n1 = shape[0], 0
            if shape[0] > 0:
                self.ptrs = <void **>malloc(shape[0] * sizeof(void *))
                for i in range(shape[0]):
                    if shape[1] > 0:
                        ptr_uint64 = _data_ptr(ndarray[i], backend)
                        self.ptrs[i] = <void *>ptr_uint64
                    else:
                        self.ptrs[i] = <void *>NULL
            else:
                self.ptrs = <void **>NULL
            self.ptr = <void *>self.ptrs
        elif ndim == 3:
            self.n0, self.n1 = shape[0], shape[1]
            if shape[0] > 0 and shape[1] > 0:
                self.ptrss = <void ***>malloc(shape[0] * sizeof(void **))
                for i in range(shape[0]):
                    self.ptrss[i] = <void **>malloc(shape[1] * sizeof(void *))
                    for j in range(shape[1]):
                        if shape[2] > 0:
                            ptr_uint64 = _data_ptr(ndarray[i, j], backend)
                            self.ptrss[i][j] = <void *>ptr_uint64
                        else:
                            self.ptrss[i][j] = <void *>NULL
            else:
                self.ptrss = <void ***>NULL
            self.ptr = <void *>self.ptrss
        else:
            raise NotImplementedError("ndarray.ndim > 3 not implemented yet")

    def __dealloc__(self):
        if self.ptrs:
            free(self.ptrs)
        if self.ptrss:
            for i in range(self.n0):
                free(self.ptrss[i])
            free(self.ptrss)

def ndarrayPointer(ndarray, as_void=False):
    _ndarray = _NDArray(ndarray)

    ndarray_type = ".".join([type(ndarray).__module__, type(ndarray).__name__])
    if ndarray_type == "numpy.ndarray":
        dtype = ndarray.dtype.str
    elif ndarray_type == "cupy.ndarray":
        dtype = ndarray.dtype.str
    elif ndarray_type == "dpnp.dpnp_array.dpnp_array":
        dtype = ndarray.dtype.str
    elif ndarray_type == "torch.Tensor":
        dtype = f"<{'c' if ndarray.dtype.is_complex else 'f' if ndarray.dtype.is_floating_point else 'i' if ndarray.dtype.is_signed else 'u'}{ndarray.dtype.itemsize}"

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

    ndim = ndarray.ndim
    if ndim == 1:
        ptr1 = Pointer(dtype)
        ptr1.set_ptr(_ndarray.ptr)
        return ptr1
    elif ndim == 2:
        ptr2 = Pointers(dtype, _ndarray.n0)
        ptr2.set_ptrs(_ndarray.ptrs)
        return ptr2
    elif ndim == 3:
        ptr3 = Pointerss(dtype, _ndarray.n0, _ndarray.n1)
        ptr3.set_ptrss(_ndarray.ptrss)
        return ptr3
