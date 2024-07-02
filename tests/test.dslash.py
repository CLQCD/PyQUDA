import numpy as np
import cupy as cp

from check_pyquda import weak_field

from pyquda import init, core, quda
from pyquda.field import Ns, Nc
from pyquda.enum_quda import QudaParity

init([1, 1, 1, 1], [16, 16, 16, 32], 1, 1.0, resource_path=".cache")
latt_info = core.getDefaultLattice()
Lx, Ly, Lz, Lt = latt_info.size


def applyDslash(Mp, p, U_seed):
    # Set parameters in Dslash and use m=-3.5 to make kappa=1
    dslash = core.getDefaultDirac(-3.5, 0, 0)

    # Generate gauge and then load it
    U = core.LatticeGauge(latt_info)
    U.gauss(U_seed, 1.0)
    dslash.loadGauge(U)

    # Load a from p and allocate b
    a = core.LatticeFermion(latt_info, cp.asarray(core.cb2(p, [0, 1, 2, 3])))
    b = core.LatticeFermion(latt_info)

    # Dslash a = b
    quda.dslashQuda(b.even_ptr, a.odd_ptr, dslash.invert_param, QudaParity.QUDA_EVEN_PARITY)
    quda.dslashQuda(b.odd_ptr, a.even_ptr, dslash.invert_param, QudaParity.QUDA_ODD_PARITY)

    # Save b to Mp
    Mp[:] = b.lexico()

    # Return gauge as a ndarray with shape (Nd, Lt, Lz, Ly, Lx, Ns, Ns)
    return U.lexico()


p = np.zeros((Lt, Lz, Ly, Lx, Ns, Nc), "<c16")
p[0, 0, 0, 0, 0, 0] = 1
Mp = np.zeros((Lt, Lz, Ly, Lx, Ns, Nc), "<c16")

U = applyDslash(Mp, p, 0)
print(Mp[0, 0, 0, 1])
