import os
import numpy as np
import cupy as cp
from pyquda import core, quda, gauge_utils, LatticePropagator

os.environ["QUDA_RESOURCE_PATH"] = ".cache"

Nc, Ns, Nd = 3, 4, 4

xi_0, nu = 4.8965, 0.86679
mass = 0.09253
coeff_r, coeff_t = 2.32582045, 0.8549165664

gauge = gauge_utils.readIldg("s1.0_cfg_100.lime")
latt_size = gauge.latt_size
Lx, Ly, Lz, Lt = latt_size
Vol = Lx * Ly * Lz * Lt

loader = core.QudaFieldLoader(latt_size, mass, 1e-9, 1000, xi_0, nu, coeff_t, coeff_r)

quda.initQuda(0)

loader.loadGauge(gauge)

propagator = LatticePropagator(latt_size)
data = propagator.data.reshape(Vol, Ns, Ns, Nc, Nc)
for spin in range(Ns):
    for color in range(Nc):
        b = core.source(latt_size, "wall", 0, spin, color)
        x = loader.invert(b)
        data[:, spin, :, color, :] = x.data.reshape(Vol, Ns, Nc)

quda.endQuda()

propagator_chroma = cp.array(np.fromfile("wl_prop_1", ">c16", offset=8).astype("<c16")).reshape(Vol, Ns, Ns, Nc, Nc)
print(cp.linalg.norm(propagator.data.reshape(Vol, Ns, Ns, Nc, Nc) - propagator_chroma))
