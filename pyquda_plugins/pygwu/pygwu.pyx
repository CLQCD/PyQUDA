from libc.stdlib cimport malloc, free
from numpy cimport ndarray
from pyquda.pointer cimport Pointer, Pointers, _NDArray
cimport gwu

def init():
    gwu.gwu_init_machine()

def shutdown():
    gwu.gwu_shutdown_machine()

def invert_overlap(
    Pointers propag_in,
    Pointers source_in,
    Pointer links_in,
    double kappa,
    ndarray[int, ndim=1] latt_size,
    ndarray[double, ndim=1] masses,
    double tol,
    int maxiter,
    double ov_ploy_prec,
    int ov_use_fp32,
    int ov_test,
    int one_minus_half_d,
    ndarray[double complex, ndim=1] hw_eigvals,
    Pointers hw_eigvecs,
    double hw_eigprec,
    ndarray[double complex, ndim=1] ov_eigvals,
    Pointers ov_eigvecs,
    double ov_eigprec,
):
    _latt_size = _NDArray(latt_size)
    cdef int _num_mass = masses.size
    _masses = _NDArray(masses)
    cdef int _hw_eignum = hw_eigvals.size
    _hw_eigvals = _NDArray(hw_eigvals)
    cdef int _ov_eignum = ov_eigvals.size
    _ov_eigvals = _NDArray(ov_eigvals)
    gwu.gwu_invert_overlap(
        propag_in.ptr,
        source_in.ptr,
        links_in.ptr,
        <int *>_latt_size.ptr,
        kappa,
        _num_mass,
        <double *>_masses.ptr,
        tol,
        maxiter,
        ov_ploy_prec,
        ov_use_fp32,
        ov_test,
        one_minus_half_d,
        _hw_eignum,
        _hw_eigvals.ptr,
        hw_eigvecs.ptr,
        hw_eigprec,
        _ov_eignum,
        _ov_eigvals.ptr,
        ov_eigvecs.ptr,
        ov_eigprec,
    )
