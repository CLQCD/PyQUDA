# PyQuda

Python wrapper for [QUDA](https://github.com/lattice/quda) written in Cython.

This project aims to benifit from the optimized linear algebra library [CuPy](https://github.com/cupy/cupy) in Python based on CUDA. CuPy and QUDA will allow us to do most operation in lattice QCD research with high performance.

This project is based on the latest QUDA release v1.1.0.

Use `pip install .` to build and install the wrapper library, or use `python3 setup.py build_ext --inplace` to build the library in the repo folder and not install it. You need to build `libquda.so` and move it to the repo folder as a prerequisite.
