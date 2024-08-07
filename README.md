# PyQUDA

Python wrapper for [QUDA](https://github.com/lattice/quda) written in Cython.

This project aims to benefit from the optimized linear algebra library [CuPy](https://github.com/cupy/cupy) in Python based on CUDA. CuPy and QUDA will allow us to perform most lattice QCD research operations with high performance. [PyTorch](https://github.com/pytorch/pytorch) is an alternative option.

This project is based on the latest QUDA `develop` branch. PyQUDA should be compatible with any commit of QUDA after May 2023, but leave some features disabled.

## Installation

Refer to https://github.com/CLQCD/PyQUDA/wiki/Installation for detailed instruction to install PyQUDA from source.

## Benchmark

Refer to https://github.com/CLQCD/PyQUDA/wiki/Benchmark for detailed instruction to run PyQUDA benchmark.

## Development

We recommend building PyQUDA using in-place mode instead of installing PyQUDA for development.

```bash
git clone https://github.com/CLQCD/PyQUDA.git
cd PyQUDA
export QUDA_PATH=/path/to/quda/build/usqcd
python3 setup.py build_ext --inplace
```

Now you can modify Python files in the project and immediately get the new result by running scripts. Adding the root directory to `sys.path` is needed if you are running scripts from other directories.

## Maintenance

Function definitions (mainly in `quda.in.pxd` and `pyquda.in.pyx`) and most docstrings (mainly in `pyquda.pyi` and `enum_quda.in.py`) should be manually updated as they cannot be autogenerated now. This also means PyQUDA should work well with future versions if the API remains unchanged.
