# Plugins for PyQUDA

PyQUDA can build Cython warpper files from a C header file.

## Example

Consider that we want to expose some functions declaried in `interface_overlap_inverter.h`, and the definitions are compiled into a shared library `libgwu.so`. We have the path for the header `GWU_INCLUDE_PATH` and the library `GWU_LIBRARY_PATH`.

`interface_overlap_inverter.h` looks like:
```cpp
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

void gwu_init_machine(const int *latt_size);
void gwu_shutdown_machine(void);
void gwu_build_hw(const void *links_in, double kappa);
void gwu_load_hw_eigen(int hw_eignum, double hw_eigprec, const double _Complex *hw_eigvals, const void **hw_eigvecs);
void gwu_build_ov(double ov_poly_prec, int ov_use_fp32);
void gwu_load_ov_eigen(int ov_eignum, double ov_eigprec, const double _Complex *ov_eigvals, const void **ov_eigvecs);
void gwu_build_hw_eigen(int hw_eignum, double hw_eigprec, int hw_extra_krylov, int maxiter, int chebyshev_order,
                        double chebyshev_cut, int iseed);
void gwu_invert_overlap(void **propag_in, const void **source_in, int num_mass, double *masses, double tol,
                        int maxiter, int one_minus_half_d, int mode);

#ifdef __cplusplus
}
#endif
```

After installing PyQUDA-Utils by `pip install pyquda-utils`, you can use the following command to build the plugin to wrap `libgwu.so`:
```bash
python3 -m pyquda_plugins -l gwu -i interface_overlap_inverter.h -L $GWU_LIBRARY_PATH -I $GWU_INCLUDE_PATH
```
`pyquda_plugins` will build Cython source files `gwu.pxd`, `_pygwu.pyx` along with the stub file `_pygwu.pyi` and then compile them into `pyquda_plugins.pygwu._pygwu` module. Arguments of C basic type will be transfered from corresponding Python objects as input, and pointers will be converted from `numpy.ndarray`, `cupy.ndarray` or `torch.Tensor`.

`gwu.pxd`
```python
cdef extern from "interface_overlap_inverter.h":
    void gwu_init_machine(const int *latt_size)
    void gwu_shutdown_machine()
    void gwu_build_hw(const void *links_in, double kappa)
    void gwu_load_hw_eigen(int hw_eignum, double hw_eigprec, const double complex *hw_eigvals, const void **hw_eigvecs)
    void gwu_build_ov(double ov_poly_prec, int ov_use_fp32)
    void gwu_load_ov_eigen(int ov_eignum, double ov_eigprec, const double complex *ov_eigvals, const void **ov_eigvecs)
    void gwu_build_hw_eigen(int hw_eignum, double hw_eigprec, int hw_extra_krylov, int maxiter, int chebyshev_order, double chebyshev_cut, int iseed)
    void gwu_invert_overlap(void **propag_in, const void **source_in, int num_mass, double *masses, double tol, int maxiter, int one_minus_half_d, int mode)
```

`_pygwu.pyx`
```python
from libcpp cimport bool
from numpy cimport ndarray
from pyquda_comm.pointer cimport Pointer, _NDArray
cimport gwu


def gwu_init_machine(ndarray[int, ndim=1] latt_size):
    _latt_size = _NDArray(latt_size)
    gwu.gwu_init_machine(<const int *>_latt_size.ptr)

def gwu_shutdown_machine():
    gwu.gwu_shutdown_machine()

def gwu_build_hw(links_in, double kappa):
    _links_in = _NDArray(links_in, 1)
    gwu.gwu_build_hw(<const void *>_links_in.ptr, kappa)

def gwu_load_hw_eigen(int hw_eignum, double hw_eigprec, ndarray[double complex, ndim=1] hw_eigvals, hw_eigvecs):
    _hw_eigvals = _NDArray(hw_eigvals)
    _hw_eigvecs = _NDArray(hw_eigvecs, 2)
    gwu.gwu_load_hw_eigen(hw_eignum, hw_eigprec, <const double complex *>_hw_eigvals.ptr, <const void **>_hw_eigvecs.ptrs)

def gwu_build_ov(double ov_poly_prec, int ov_use_fp32):
    gwu.gwu_build_ov(ov_poly_prec, ov_use_fp32)

def gwu_load_ov_eigen(int ov_eignum, double ov_eigprec, ndarray[double complex, ndim=1] ov_eigvals, ov_eigvecs):
    _ov_eigvals = _NDArray(ov_eigvals)
    _ov_eigvecs = _NDArray(ov_eigvecs, 2)
    gwu.gwu_load_ov_eigen(ov_eignum, ov_eigprec, <const double complex *>_ov_eigvals.ptr, <const void **>_ov_eigvecs.ptrs)

def gwu_build_hw_eigen(int hw_eignum, double hw_eigprec, int hw_extra_krylov, int maxiter, int chebyshev_order, double chebyshev_cut, int iseed):
    gwu.gwu_build_hw_eigen(hw_eignum, hw_eigprec, hw_extra_krylov, maxiter, chebyshev_order, chebyshev_cut, iseed)

def gwu_invert_overlap(propag_in, source_in, int num_mass, ndarray[double, ndim=1] masses, double tol, int maxiter, int one_minus_half_d, int mode):
    _propag_in = _NDArray(propag_in, 2)
    _source_in = _NDArray(source_in, 2)
    _masses = _NDArray(masses)
    gwu.gwu_invert_overlap(<void **>_propag_in.ptrs, <const void **>_source_in.ptrs, num_mass, <double *>_masses.ptr, tol, maxiter, one_minus_half_d, mode)
```

```python
from numpy import int32, float64, complex128
from numpy.typing import NDArray

def gwu_init_machine(latt_size: NDArray[int32]) -> None: ...
def gwu_shutdown_machine() -> None: ...
def gwu_build_hw(links_in: NDArray, kappa: float) -> None: ...
def gwu_load_hw_eigen(hw_eignum: int, hw_eigprec: float, hw_eigvals: NDArray[complex128], hw_eigvecs: NDArray) -> None: ...
def gwu_build_ov(ov_poly_prec: float, ov_use_fp32: int) -> None: ...
def gwu_load_ov_eigen(ov_eignum: int, ov_eigprec: float, ov_eigvals: NDArray[complex128], ov_eigvecs: NDArray) -> None: ...
def gwu_build_hw_eigen(hw_eignum: int, hw_eigprec: float, hw_extra_krylov: int, maxiter: int, chebyshev_order: int, chebyshev_cut: float, iseed: int) -> None: ...
def gwu_invert_overlap(propag_in: NDArray, source_in: NDArray, num_mass: int, masses: NDArray[float64], tol: float, maxiter: int, one_minus_half_d: int, mode: int) -> None: ...
```

Then you can use functions from `libgwu.so`. We recommand to build a pure Python file for better user interface. You can check [pygwu](./pyquda_plugins/pygwu/__init__.py) as an example.

## Limitations

Because we compile the header in C mode, ensure that all the declarations are wrapped with
```cpp
#ifdef __cplusplus
extern "C" {
#endif

// function declarations

#ifdef __cplusplus
}
#endif
```

Because of Cython's [automatic type conversions](https://cython.readthedocs.io/en/latest/src/userguide/language_basics.html#automatic-type-conversions), we have some limitations on the argument type:

| C type                                                      | Python type           |
| ----------------------------------------------------------- | --------------------- |
| `[unsigned] int`, `[unsigned] long`, `[unsigned] long long` | `int`                 |
| `float`, `double`                                           | `float`               |
| `float _Complex`, `double _Complex`                         | `complex`             |
| `int *`                                                     | `ndarray[int32]`      |
| `double *`                                                  | `ndarray[float64]`    |
| `double _Complex *`                                         | `ndarray[complex128]` |
| `const char *`, `const char []`                             | `bytes`               |

- `struct`, `union` in C is unacceptable as the arguments for now

[pycparser](https://github.com/eliben/pycparser) uses `fake_libc_include` to parse standard headers, which cannot be easily install with `pip install pycparser`, so you cannot include any standard library header such as `stdlib.h` in the header.
