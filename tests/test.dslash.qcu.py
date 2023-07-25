import os
import sys

# test_dir = os.path.dirname(os.path.abspath(__file__))
# sys.path.insert(0, os.path.join(test_dir, ".."))

import numpy as np

from pyquda import init, qcu

os.environ["QUDA_RESOURCE_PATH"] = ".cache"
init()

Lx, Ly, Lz, Lt = 16, 16, 16, 32
Nd, Ns, Nc = 4, 4, 3
latt_size = [Lx, Ly, Lz, Lt]


def applyDslash(Mp, p, U_seed):
    import cupy as cp
    from pyquda import core, quda
    from pyquda.enum_quda import QudaParity
    from pyquda.field import LatticeFermion
    from pyquda.utils import gauge_utils

    # Set parameters in Dslash and use m=-3.5 to make kappa=1
    dslash = core.getDslash(latt_size, -3.5, 0, 0, anti_periodic_t=False)

    # Generate gauge and then load it
    U = gauge_utils.gaussGauge(latt_size, U_seed)
    dslash.loadGauge(U)

    # Load a from p and allocate b
    a = LatticeFermion(latt_size, cp.asarray(core.cb2(p, [0, 1, 2, 3])))
    b = LatticeFermion(latt_size)

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

Mp1 = np.zeros((Lt, Lz, Ly, Lx, Ns, Nc), np.complex128)
param = qcu.QcuParam()
param.lattice_size = latt_size
qcu.dslashQcu(Mp1, p, U, param)
print(Mp[0, 0, 0, 1])

print(np.linalg.norm(Mp - Mp1))
