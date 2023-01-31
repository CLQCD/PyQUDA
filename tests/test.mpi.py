import os
import sys
import numpy as np
import cupy as cp

test_dir = os.path.dirname(os.path.abspath(__file__))
# sys.path.insert(0, os.path.join(test_dir, ".."))

from pyquda import core, mpi
from pyquda.utils import gauge_utils

os.environ["QUDA_RESOURCE_PATH"] = ".cache"

latt_size = [4, 4, 4, 8]
grid_size = [1, 1, 1, 2]
Lx, Ly, Lz, Lt = latt_size
Gx, Gy, Gz, Gt = grid_size
latt_size = [Lx // Gx, Ly // Gy, Lz // Gz, Lt // Gt]
Lx, Ly, Lz, Lt = latt_size
Vol = Lx * Ly * Lz * Lt

mpi.init(grid_size)

xi_0, nu = 2.464, 0.95
kappa = 0.115
coeff = 1.17
coeff_r, coeff_t = 0.91, 1.07
mass = 1 / (2 * kappa) - 4
dslash = core.getDslash(latt_size, mass, 1e-9, 1000, xi_0, nu, coeff_t, coeff_r)

gauge = gauge_utils.readIldg(os.path.join(test_dir, "weak_field.lime"))

dslash.loadGauge(gauge)

propagator = core.invert(dslash, "point", [0, 0, 0, 0])

propagator_all = mpi.gather(propagator.data, [1, 2, 3, 4])

if mpi.rank == 0:
    propagator_chroma = np.fromfile("pt_prop_1", ">c16", offset=8).astype("<c16")
    print(np.linalg.norm(propagator_all.transpose(0, 1, 2, 3, 4, 6, 5, 8, 7).reshape(-1) - propagator_chroma))
