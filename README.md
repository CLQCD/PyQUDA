# PyQuda

Python wrapper for [QUDA](https://github.com/lattice/quda) written in Cython.

This project aims to benifit from the optimized linear algebra library [CuPy](https://github.com/cupy/cupy) in Python based on CUDA. CuPy and QUDA will allow us to do most operation in lattice QCD research with high performance.

This project is based on the latest QUDA release v1.1.0.

Use `pip install .` to build and install the wrapper library, or use `python3 setup.py build_ext --inplace` to build the library in the repo folder and not install it. You need to build `libquda.so` and move it to the repo folder as a prerequisite.

## Installation

### QUDA

This is an example to build QUDA for single GPU.

```bash
git clone https://github.com/SaltyChiang/quda.git
pushd quda
git checkout feature/gauss-mom
mkdir build
pushd build
cmake .. -DQUDA_DIRAC_DOMAIN_WALL=OFF -DQUDA_DIRAC_NDEG_TWISTED_CLOVER=OFF -DQUDA_DIRAC_NDEG_TWISTED_MASS=OFF -DQUDA_DIRAC_STAGGERED=OFF -DQUDA_DIRAC_TWISTED_CLOVER=OFF -DQUDA_DIRAC_TWISTED_MASS=OFF -DQUDA_INTERFACE_MILC=OFF -DQUDA_MULTIGRID=ON
cmake --build . -j8
popd
popd
```

### PyQuda

Build, install and run the example.

`chroma` is needed here for generating the reference file.

```bash
git clone https://github.com/IHEP-LQCD/PyQuda.git
pushd PyQuda
python3 -m pip install -r requirements.txt
cp ../quda/build/lib/libquda.so ./
python3 -m pip install .

chroma -i tests/test.clover.ini.xml
python3 tests/test.clover.py
popd
```
