import os
import sys
import cupy as cp

test_dir = os.path.dirname(os.path.abspath(__file__))
# sys.path.insert(0, os.path.join(test_dir, ".."))
from pyquda import core, init
from pyquda.utils import io

os.environ["QUDA_RESOURCE_PATH"] = ".cache"
init()

latt_size = [4, 4, 4, 8]
Lx, Ly, Lz, Lt = latt_size
Vol = Lx * Ly * Lz * Lt

xi_0, nu = 2.464, 0.95
kappa = 0.115
coeff = 1.17
coeff_r, coeff_t = 0.91, 1.07
alpha0 = 0.4
t0 = 0

mass = 1 / (2 * kappa) - 4

from pyquda.dirac import general

dslash = core.getDslash(latt_size, mass, 1e-9, 1000, xi_0, nu, coeff_t, coeff_r, multigrid=False)
general.cuda_prec_sloppy = 8
dslash.invert_param.distance_pc_alpha0 = alpha0
dslash.invert_param.distance_pc_t0 = t0
gauge = io.readQIOGauge(os.path.join(test_dir, "weak_field.lime"))


dslash.loadGauge(gauge)

propagator = core.invert(dslash, "point", [0, 0, 0, t0])

dslash.destroy()

propagator_chroma = io.readQIOPropagator("pt_prop_1")
propagator_chroma.toDevice()
print(cp.linalg.norm(propagator.data - propagator_chroma.data))
