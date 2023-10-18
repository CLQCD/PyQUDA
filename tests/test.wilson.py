import os
import sys
import cupy as cp

test_dir = os.path.dirname(os.path.abspath(__file__))
# sys.path.insert(0, os.path.join(test_dir, ".."))
from pyquda import core, init
from pyquda.utils import io

os.environ["QUDA_RESOURCE_PATH"] = ".cache"

latt_size = [4, 4, 4, 8]
Lx, Ly, Lz, Lt = latt_size
Vol = Lx * Ly * Lz * Lt
init()

xi_0, nu = 2.464, 0.95
kappa = 0.135
mass = 1 / (2 * kappa) - 4

dslash = core.getDslash(latt_size, mass, 1e-9, 1000, xi_0, nu, multigrid=False)
gauge = io.readQIOGauge(os.path.join(test_dir, "weak_field.lime"))


dslash.loadGauge(gauge)

propagator = core.invert(dslash, "point", [0, 0, 0, 0])

dslash.destroy()

propagator_chroma = io.readQIOPropagator("pt_prop_0")
propagator_chroma.toDevice()
print(cp.linalg.norm(propagator.data - propagator_chroma.data))
