cdef extern from "malloc_quda.h" namespace "quda::pool":
    void *device_malloc_(const char *func, const char *file, int line, size_t size)
    void device_free_(const char *func, const char *file, int line, void *ptr)
