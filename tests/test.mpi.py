from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

import os
import sys
import numpy as np
import cupy as cp

test_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(test_dir, ".."))

from pyquda import quda, core
from pyquda.core import Nc, Nd, Ns
from pyquda.utils import gauge_utils, prop_utils

os.environ["QUDA_RESOURCE_PATH"] = ".cache"

latt_size = [4, 4, 4, 8]
grid_size = [1, 1, 1, 2]
Lx, Ly, Lz, Lt = latt_size
Gx, Gy, Gz, Gt = grid_size
latt_size = [Lx // Gx, Ly // Gy, Lz // Gz, Lt // Gt]
Lx, Ly, Lz, Lt = latt_size
Vol = Lx * Ly * Lz * Lt

cp.cuda.Device(rank).use()
quda.initCommsGridQuda(4, grid_size)

xi_0, nu = 2.464, 0.95
kappa = 0.115
coeff = 1.17
coeff_r, coeff_t = 0.91, 1.07
mass = 1 / (2 * kappa) - 4
loader = core.QudaFieldLoader(latt_size, mass, 1e-9, 1000, xi_0, nu, coeff_t, coeff_r)

gauge = gauge_utils.readIldg(os.path.join(test_dir, "weak_field.lime"), grid_size, rank)

quda.initQuda(rank)

loader.loadGauge(gauge)

propagator = core.LatticePropagator(latt_size)
data = propagator.data.reshape(Vol, Ns, Ns, Nc, Nc)
for spin in range(Ns):
    for color in range(Nc):
        b = core.LatticeFermion(latt_size)
        b_data = b.data.reshape(Vol, Ns, Nc)
        if rank == 0:
            b_data[0, spin, color] = 1
        x = loader.invert(b)
        data[:, :, spin, :, color] = x.data.reshape(Vol, Ns, Nc)

quda.endQuda()

propagator_all = prop_utils.collect(propagator, grid_size, comm, rank)

if rank == 0:
    propagator_chroma = cp.array(np.fromfile("pt_prop_1", ">c16", offset=8).astype("<c16"))
    print(cp.linalg.norm(propagator_all.transpose() - propagator_chroma))
