from numpy cimport ndarray
from pyquda.pointer cimport Pointer, Pointers, _NDArray
cimport gwu

def init(ndarray[int, ndim=1] latt_size):
    _latt_size = _NDArray(latt_size)
    gwu.gwu_init_machine(<int *>_latt_size.ptr)

def shutdown():
    gwu.gwu_shutdown_machine()

def build_hw(Pointer links_in, double kappa):
    gwu.gwu_build_hw(links_in.ptr, kappa)

def load_hw_eigen(int hw_eignum, double hw_eigprec, ndarray[double complex, ndim=1] hw_eigvals, Pointers hw_eigvecs):
    _hw_eigvals = _NDArray(hw_eigvals)
    gwu.gwu_load_hw_eigen(hw_eignum, hw_eigprec, <double complex *>_hw_eigvals.ptr, hw_eigvecs.ptr)

def build_ov(double ov_poly_prec, int ov_use_fp32):
    gwu.gwu_build_ov(ov_poly_prec, ov_use_fp32)

def load_ov_eigen(int ov_eignum, double ov_eigprec, ndarray[double complex, ndim=1] ov_eigvals, Pointers ov_eigvecs):
    _ov_eigvals = _NDArray(ov_eigvals)
    gwu.gwu_load_ov_eigen(ov_eignum, ov_eigprec, <double complex *>_ov_eigvals.ptr, ov_eigvecs.ptr)

def build_hw_eigen(int hw_eignum, double hw_eigprec, int hw_extra_krylov, int maxiter, int chebyshev_order, double chebyshev_cut, int iseed):
    gwu.gwu_build_hw_eigen(hw_eignum, hw_eigprec, hw_extra_krylov, maxiter, chebyshev_order, chebyshev_cut, iseed)

def invert_overlap(
    Pointers propag_in,
    Pointers source_in,
    ndarray[double, ndim=1] masses,
    double tol,
    int maxiter,
    int one_minus_half_d,
    int mode,
):
    cdef int _num_mass = masses.size
    _masses = _NDArray(masses)
    gwu.gwu_invert_overlap(
        propag_in.ptr,
        source_in.ptr,
        _num_mass,
        <double *>_masses.ptr,
        tol,
        maxiter,
        one_minus_half_d,
        mode,
    )
