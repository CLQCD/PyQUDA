# 简介与示例

## 0. Environment configuration

*We recommand using Python standard library `venv` instead of `conda` to manage your virtual environment.*

### a. Ensure Python version >= 3.7

```bash
python3 --version
```

### b. Initialize a vritual environment of Python.

venv version:
```bash
python3 -m venv .venv
```

conda version:
```bash
conda create .venv
```

### c. Activate the virtual environment

venv version:
```bash
source .venv/bin/activate
```

conda version:
```bash
conda activate .venv
```

### d. Install prerequisites

```bash
pip install numpy mpi4py
```

#### i. Install CuPy with CUDA Toolkit
Install prebuilt CuPy with your CUDA Toolkit version
```bash
pip install cupy-cuda110
pip install cupy-cuda111
pip install cupy-cuda11x
pip install cupy-cuda12x
```

#### ii. Install CuPy with DTK

Download the source code of CuPy version 12.3.0
```bash
wget https://github.com/cupy/cupy/releases/download/v12.3.0/cupy-12.3.0.tar.gz
tar -xzvf cupy-12.3.0.tar.gz
cd cupy-12.3.0
```

Apply the patch
```diff
diff --git a/install/cupy_builder/_features.py b/install/cupy_builder/_features.py
index d12de78c3..8c9ac830a 100644
--- a/install/cupy_builder/_features.py
+++ b/install/cupy_builder/_features.py
@@ -173,7 +173,7 @@ def get_features(ctx: Context) -> Dict[str, Feature]:
             'hiprand',
             'hipsparse',
             'rocfft',
-            'roctx64',
+            #'roctx64',
             'rocblas',
             'rocsolver',
             'rocsparse',
```

Build and install CuPy from source
```bash
export CUPY_INSTALL_USE_HIP=1
export ROCM_HOME=/path/to/dtk-23.10
pip install .
```

### e. Build QUDA

#### i. From source

```bash
git clone https://github.com/lattice/quda.git
mkdir build && cd build
export GPU_TARGET=sm_70
cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DQUDA_GPU_ARCH=${GPU_TARGET} \
    -DQUDA_MPI=ON \
    -DQUDA_CONTRACT=ON \
    -DQUDA_COVDEV=ON \
    -DQUDA_CLOVER_DYNAMIC=OFF \
    -DQUDA_CLOVER_RECONSTRUCT=OFF \
    -DQUDA_DIRAC_CLOVER_HASENBUSCH=OFF \
    -DQUDA_DIRAC_DOMAIN_WALL=OFF \
    -DQUDA_DIRAC_TWISTED_CLOVER=OFF \
    -DQUDA_DIRAC_TWISTED_MASS=OFF \
    -DQUDA_DIRAC_NDEG_TWISTED_CLOVER=OFF \
    -DQUDA_DIRAC_NDEG_TWISTED_MASS=OFF \
    -DQUDA_LAPLACE=ON -DQUDA_MULTIGRID=ON \
    -DQUDA_MULTIGRID_NVEC_LIST="6,24,32,64,96" \
    ..
cmake --build . -j16 && cmake --install .
```

#### ii. Using Build-USQCD-SciDAC repo

```bash
git clone https://github.com/SaltyChiang/Build-USQCD-SciDAC.git
cd Build-USQCD-SciDAC
./pyquda.sh dtk
scp scidac.tgz remote:~/
```

```bash
ssh remote
tar -xzvf scidac.tgz
cd scidac
./build_pyquda.sh
```

You should edit the first few lines to match your devices.
```
BUILD_SHAREDLIB=ON
GPU_TARGET=gfx906
HETEROGENEOUS_ATOMIC=OFF
LLVM_VERSION=15  # Match the LLVM version shipped with DTK
JOBS=16  # Match the number of CPU cores
QUDA_JOBS=16  # Match the number of CPU cores (Reduce this if the memory is limited)
OFFLINE=1  # Should be 1 if the host is offline
```

### f. Build and install PyQUDA

```bash
git clone --recursive https://github.com/CLQCD/PyQUDA.git
cd PyQUDA.git
export QUDA_PATH=/path/to/scidac/install/pyquda-${GPU_TARGET}
pip install .
```

Make sure everything is fine
```bash
./tests/bin/chroma -i tests/test.clover.ini.xml
python3 tests/test.clover.py
```

### g. Use QUDA functions in Python

Most functions in `quda.h` could be used by PyQUDA
```bash
from pyquda import quda

quda.initQuda(-1)
quda.endQuda()
```

## 0.5. Module `pyquda.field`

### Data format

#### `LatticeGauge.data`
Could be `numpy.ndarray`, `cupy.ndarray` or `torch.Tensor` with dtype `complex128` and shape `(Nd, 2, Lt, Lz, Ly, Lx // 2, Nc, Nc)`, row-column order.

#### `LatticePropagator.data`
Could be `numpy.ndarray`, `cupy.ndarray` or `torch.Tensor` with dtype `complex128` and shape `(2, Lt, Lz, Ly, Lx // 2, Ns, Ns, Nc, Nc)`, row-column order.

#### `LatticeFermion.data`
Could be `numpy.ndarray`, `cupy.ndarray` or `torch.Tensor` with dtype `complex128` and shape `(2, Lt, Lz, Ly, Lx // 2, Ns, Nc)`, row-column order.

### `LatticeInfo`

`LatticeInfo` is used to handle the basic information including size, grid, t boundary condition and anisotropy of a lattice.

Most class in PyQUDA shoule use `LatticeInfo` as the input to instantiate a object.

### `LatticeField`

#### `LatticeField.latt_info`

#### `LatticeField.backend`

#### `LatticeField.location`

#### `LatticeField.data`

Get the `numpy.ndarray`/`cupy.ndarray`/`torch.Tensor` which handles the actual lattice data.

#### `LatticeField.copy()`

Get a copy of the field.

#### `LatticeField.toDevice()`

Move the data from `numpy.ndarray` to `cupy.ndarray`/`torch.Tensor`.

#### `LatticeField.toHost()`

Move the data from `cupy.ndarray`/`torch.Tensor` to `numpy.ndarray`.

#### `LatticeField.getHost()`

Returns a `numpy.ndarray` copy of `LatticeFied.data`.

### `LatticeGauge`
`LatticeGauge` is the class to handle the gauge data along with some utility pure gauge functions such as gauge fixing and smearing. A `LatticeGauge` is usually generated by `io.readGauge`.

#### `LatticeGauge(latt_info, value)`

Instantiates a `LatticeGauge` with `latt_info`, and the value of `data` is initialized with `value`. Will instantiate a unit gauge field when `value=None`.


#### `LatticeGauge.lexico()`

Returns a `numpy.ndarray` copy of `LatticeField.data`, but without even-odd preconditioning.

#### `LatticeGauge.covDev(x, covdev_mu)`

Applies the covariant derivative on `x` in direction `covdev_mu`. `x` should be `LatticeFermion`. 0/1/2/3 represent +x/+y/+z/+t and 4/5/6/7 represent -x/-y/-z/-t. The covariant derivative is defined as $\psi'(x)=U_\mu(x)\psi(x+\hat{\mu})$.

#### `LatticeGauge.laplace(x, laplace3D)`

Applies the Laplacian operator on `x`, and `laplace3D` takes 3 or 4 to apply Laplacian on spacial or all directions. `x` should be `LatticeStaggeredFermion`. The Laplacian operator is defined as $\psi'(x)=\frac{1}{N_\mathrm{Lap}}\sum_\mu\psi(x)-\dfrac{1}{2}\left[U_\mu(x)\psi(x+\hat{\mu})+U_\mu^\dagger(x-\hat{\mu})\psi(x-\hat{\mu})\right]$

#### `LatticeGauge.staggeredPhase()`

Applies the staggered phase to the gauge field. The convention is controld by `LatticeGauge.pure_gauge.gauge_param.staggered_phase_type`, which is `QudaStaggeredPhase.QUDA_STAGGERED_PHASE_MILC` by default.

#### `LatticeGauge.projectSU3(tol)`

Projects the gauge field onto SU(3) matrix. `tol` is the tolerance of how matrix deviates from SU(3). `2e-15` (which is 10x the epsilon of fp64) should be a good choice.

#### `LatticeGauge.path(paths, coeff)`

`paths` is a list of length 4, which is `[paths_x, paths_y, paths_z, paths_t]`. All `paths_*` should have the same shape. `paths_x` is a list of any length.

#### `LatticeGauge.loopTrace(paths)`

`paths` is similar to `paths_x` in `LatticeGauge.path`, but the function returns the traces of all loops. The traces is a ndarray of complex number, and every element is definded as $\sum_x\mathrm{Tr}W(x)$.

#### `LatticeGauge.apeSmear(n_steps, alpha, dir_ignore)`

Applies the APE smearing to the gauge field. `alpha` is the smearing strength.

#### `LatticeGauge.smearAPE(n_steps, factor, dir_ignore)`

Similar to `LatticeGauge.apeSmear()` but `factor` matchs Chroma.

#### `LatticeGauge.stoutSmear(n_steps, rho, dir_ignore`

Applies the stout smearing to the gauge field. `rho` is the smearing strength.

#### `LatticeGauge.smearSTOUT(n_steps, rho, dir_ignore)`

Similar to `LatticeGauge.stoutSmear()`.

#### `LatticeGauge.hypSmear(n_steps, alpha1, alpha2, alpha3, dir_ignore)`

Applies the stout smearing to the gauge field. `alpha1/alpha2/alpha3` is the smearing strength on level 3/2/1.

#### `LatticeGauge.smearHYP(n_steps, alpha1, alpha2, alpha3, dir_ignore)`

Similar to `LatticeGauge.hypSmear()`.

#### `LatticeGauge.wilsonFlow(n_steps, epsilon)`

Applies the Wilson flow to the gauge field. Returns the energy (all, spacial, temporal) for every step.

#### `LatticeGauge.wilsonFlowScale(max_steps, epsilon)`

Returns $t_0$ and $w_0$ with up to `max_steps` Wilson flow step.

#### `LatticeGauge.symanzikFlow(n_steps, epsilon)`

Applies the Symanzik flow to the gauge field. Returns the energy (all, spacial, temporal) for every step.

#### `LatticeGauge.symanzikFlowScale(max_steps, epsilon)`

Returns $t_0$ and $w_0$ with up to `max_steps` Symanzik flow step.

#### `LatticeGauge.plaquette()`

Returns the plaquette (all, spacial, temporal) of the gauge field.

#### `LatticeGauge.polyakovLoop()`

Retruns the Polyakov loop (real, image) of the gauge field.

#### `LatticeGauge.energy()`

Returns the energy (all, spacial, temporal) of the gauge field.

#### `LatticeGauge.qcharge()`

Returns the topological charge of the gauge field.

<!-- #### `LatticeGauge.qcharge()`

Returns the topological charge density (with shape `(Lt, Lz, Ly, Lx)`)of the gauge field. -->

#### `LatticeGauge.gauss(seed, sigma)`

Fills the gauge field with random SU(3) matrices. `sigma=1` corresponds to the standard normal distribution.

#### `LatticeGauge.fixingOVR(gauge_dir, Nsteps, verbose_interval, relax_boost, tolerance, reunit_interval, stopWtheta)`

Applies the gauge gixing to the gauge field and over relaxation is used to speed up the operation.

    gauge_dir: {3, 4}
        3 for Coulomb gauge fixing, 4 for Landau gauge fixing
    Nsteps: int
        maximum number of steps to perform gauge fixing
    verbose_interval: int
        print gauge fixing info when iteration count is a multiple of this
    relax_boost: float
        gauge fixing parameter of the overrelaxation method, most common value is 1.5 or 1.7.
    tolerance: float
        torelance value to stop the method, if this value is zero then the method stops when
        iteration reachs the maximum number of steps defined by Nsteps
    reunit_interval: int
        reunitarize gauge field when iteration count is a multiple of this
    stopWtheta: int
        0 for MILC criterion and 1 to use the theta value

#### `LatticeGauge.fixingFFT(gauge_dir, Nsteps, verbose_interval, alpha, autotune, tolerance, stopWtheta)`

Applies the gauge gixing to the gauge field and fast Fourier transform is used to speed up the operation.

    gauge_dir: {3, 4}
        3 for Coulomb gauge fixing, 4 for Landau gauge fixing
    Nsteps: int
        maximum number of steps to perform gauge fixing
    verbose_interval: int
        print gauge fixing info when iteration count is a multiple of this
    alpha: float
        gauge fixing parameter of the method, most common value is 0.08
    autotune: int
        1 to autotune the method, i.e., if the Fg inverts its tendency we decrease the alpha value
    tolerance: float
        torelance value to stop the method, if this value is zero then the method stops when
        iteration reachs the maximum number of steps defined by Nsteps
    stopWtheta: int
        0 for MILC criterion and 1 to use the theta value


### `LatticePropagator`

A `LatticePropagator` is usually generated by `source` and `invert` functions.


### `LatticeFermion`

A `LatticeFermion` should always be a intermediate variant.

## 1. Quenched HMC and dynamical HMC (Optional)

```bash
python3 1.py
```

## 2. 在产生的规范场组态上做HYP smearing，然后算静态势并大致估计格距

## 3. 在HYP smeared的组态上跑clover传播子（csw=1），计算pion和nucleon的2pt（mpi～300 MeV，700MeV，3000MeV）

```bash
python3 3_Pion_Proton_2pt.py
```

## 4. 基于PCAC计算三种pion mass下的quark mass

```bash
python3 4_Pion_PCAC.py
```

## 5. 计算不同动量的2pt，确定色散关系，计算quasi-DA矩阵元（可选）

```bash
python3 5_Pion_Dispersion.py
```

## 6. 计算seq source，计算nucleon的gV和gA

## 7. 计算quasi-PDF矩阵元（可选）
