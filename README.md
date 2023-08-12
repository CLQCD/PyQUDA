# PyQuda

Python wrapper for [QUDA](https://github.com/lattice/quda) written in Cython.

This project aims to benifit from the optimized linear algebra library [CuPy](https://github.com/cupy/cupy) in Python based on CUDA. CuPy and QUDA will allow us to do most operation in lattice QCD research with high performance.

This project is based on the latest QUDA release v1.1.0.

Use `pip install .` to build and install the wrapper library, or use `python3 setup.py build_ext --inplace` to build the library in the repo folder and not install it. You need to build `libquda.so` and move it to the repo folder as a prerequisite.

## Installation

### QUDA

This is an example to build QUDA for single GPU. The default install directory of QUDA should be `/path/to/quda/build/usqcd`.

```bash
git clone https://github.com/CLQCD/quda.git
pushd quda
git checkout b47950dd
mkdir build
pushd build
cmake .. -DQUDA_DIRAC_DOMAIN_WALL=OFF -DQUDA_CLOVER_DYNAMIC=OFF -DQUDA_CLOVER_RECONSTRUCT=OFF -DQUDA_DIRAC_NDEG_TWISTED_CLOVER=OFF -DQUDA_DIRAC_NDEG_TWISTED_MASS=OFF -DQUDA_DIRAC_TWISTED_CLOVER=OFF -DQUDA_DIRAC_TWISTED_MASS=OFF -DQUDA_INTERFACE_MILC=OFF -DQUDA_LAPLACE=ON -DQUDA_MULTIGRID=ON -DQUDA_LAPLACE=ON
cmake --build . -j12
cmake --install .
popd
popd
```

### PyQuda

Install PyQuda with your Python interpreter.
Here we copy `libquda.so` built previously to the root directory of PyQuda to simplify the environ.

```bash
export LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH
git clone https://github.com/IHEP-LQCD/PyQuda.git
pushd PyQuda
python3 -m pip install -r requirements.txt
cp ../quda/build/lib/libquda.so ./
python3 -m pip install .
popd
```

`chroma` is needed here for generating the reference files used by test scripts.

```bash
export LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH
pushd PyQuda
chroma -i tests/test.clover.ini.xml
python3 tests/test.clover.py
popd
```

## Development

I recommand you to build PyQuda using inplace mode instead of installing PyQuda.

```bash
export LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH
git clone https://github.com/IHEP-LQCD/PyQuda.git
pushd PyQuda
python3 -m pip install -r requirements.txt
cp ../quda/build/lib/libquda.so ./
python3 setup.py build_ext --inplace
popd
```

Now you can modify Python files in the project and immediately get the result by running scripts. Adding the root directory to `sys.path` list is needed if you are running scripts from other directories.
