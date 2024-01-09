import os
import sys
import numpy as np
import cupy as cp

test_dir = os.path.dirname(os.path.abspath(__file__))
# sys.path.insert(1, os.path.join(test_dir, ".."))
from pyquda import init, core, quda
from pyquda.field import Ns, Nc, LatticeInfo, LatticeGauge, LatticeFermion
from pyquda.enum_quda import QudaParity

os.environ["QUDA_RESOURCE_PATH"] = ".cache"

init()
latt_info = LatticeInfo([16, 16, 16, 32])
Lx, Ly, Lz, Lt = latt_info.size


def gaussGauge(latt_info: LatticeInfo, seed: int):
    gauge = LatticeGauge(latt_info, None)

    dslash = core.getDslash(latt_info.size, 0, 0, 0, anti_periodic_t=False)
    dslash.gauge_param.use_resident_gauge = 0
    quda.loadGaugeQuda(gauge.data_ptrs, dslash.gauge_param)
    dslash.gauge_param.use_resident_gauge = 1
    quda.gaussGaugeQuda(seed, 1.0)
    quda.saveGaugeQuda(gauge.data_ptrs, dslash.gauge_param)

    return gauge


def applyDslash(Mp, p, U_seed):
    # Set parameters in Dslash and use m=-3.5 to make kappa=1
    dslash = core.getDslash(latt_info.size, -3.5, 0, 0, anti_periodic_t=False)

    # Generate gauge and then load it
    U = gaussGauge(latt_info, U_seed)
    dslash.loadGauge(U)

    # Load a from p and allocate b
    a = LatticeFermion(latt_info, cp.asarray(core.cb2(p, [0, 1, 2, 3])))
    b = LatticeFermion(latt_info)

    # Dslash a = b
    quda.dslashQuda(b.even_ptr, a.odd_ptr, dslash.invert_param, QudaParity.QUDA_EVEN_PARITY)
    quda.dslashQuda(b.odd_ptr, a.even_ptr, dslash.invert_param, QudaParity.QUDA_ODD_PARITY)

    # Save b to Mp
    Mp[:] = b.lexico()

    # Return gauge as a ndarray with shape (Nd, Lt, Lz, Ly, Lx, Ns, Ns)
    return U.lexico()


p = np.zeros((Lt, Lz, Ly, Lx, Ns, Nc), np.complex128)
p[0, 0, 0, 0, 0, 0] = 1
Mp = np.zeros((Lt, Lz, Ly, Lx, Ns, Nc), np.complex128)

U = applyDslash(Mp, p, 0)
print(Mp[0, 0, 0, 1])
