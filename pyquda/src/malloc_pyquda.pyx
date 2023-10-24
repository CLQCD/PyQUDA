cimport malloc_quda

def pyquda_cupy_malloc(size_t size, int device_id) -> int:
    cdef size_t ptr = <size_t>malloc_quda.device_malloc_("pyquda_cupy_malloc", "pyquda/src/malloc_pyquda.pyx", 4, size)
    return ptr

def pyquda_cupy_free(size_t ptr, int device_id) -> None:
    malloc_quda.device_free_("pyquda_cupy_malloc", "pyquda/src/malloc_pyquda.pyx", 8, <void *>ptr)
