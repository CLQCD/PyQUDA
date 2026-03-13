import numpy as np
import cupy as cp

from check_pyquda import weak_field

from pyquda.field import Ns, Nc
from pyquda_utils import core

core.init([1, 1, 1, 1], resource_path=".cache/quda")
latt_info = core.LatticeInfo([16, 16, 16, 32], 1, 1.0)
Lx, Ly, Lz, Lt = latt_info.size


def applyDslash(Mp, p, U_seed):
    # Set parameters in Dslash and use m=-3.5 to make kappa=1
    dirac = core.getWilson(latt_info, -3.5, 0, 0)

    # Generate gauge and then load it
    U = core.LatticeGauge(latt_info)
    U.gauss(U_seed, 1.0)

    # Load a from p and allocate b
    a = core.LatticeFermion(latt_info, cp.asarray(latt_info.evenodd(p, False)))
    b = core.LatticeFermion(latt_info)

    # Dslash a = b
    with dirac.useGauge(U):
        b = dirac.dslash(a)

    # Save b to Mp
    Mp[:] = b.lexico()

    # Return gauge as a ndarray with shape (Nd, Lt, Lz, Ly, Lx, Ns, Ns)
    return U.lexico()


p = np.zeros((Lt, Lz, Ly, Lx, Ns, Nc), "<c16")
p[0, 0, 0, 0, 0, 0] = 1
Mp = np.zeros((Lt, Lz, Ly, Lx, Ns, Nc), "<c16")

U = applyDslash(Mp, p, 0)
print(Mp[0, 0, 0, 1])
