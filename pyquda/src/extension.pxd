cdef extern from "sample.h":
    void my_dslash_interface(void *U_ptr, void *a_ptr, void *b_ptr, int Lx, int Ly, int Lz, int Lt, int Nd, int Ns, int Nc)
