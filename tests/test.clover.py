import os
import sys
import cupy as cp

test_dir = os.path.dirname(os.path.abspath(__file__))
# sys.path.insert(1, os.path.join(test_dir, ".."))
from pyquda import core, init
from pyquda.utils import io
from pyquda.field import LatticeInfo

os.environ["QUDA_RESOURCE_PATH"] = ".cache"

init()
latt_info = LatticeInfo([4, 4, 4, 8])

xi_0, nu = 2.464, 0.95
kappa = 0.115
coeff = 1.17
coeff_r, coeff_t = 0.91, 1.07

mass = 1 / (2 * kappa) - 4

dslash = core.getDslash(latt_info.size, mass, 1e-12, 1000, xi_0, nu, coeff_t, coeff_r, multigrid=False)
gauge = io.readQIOGauge(os.path.join(test_dir, "weak_field.lime"))


dslash.loadGauge(gauge)

propagator = core.invert(dslash, "point", [0, 0, 0, 0])

dslash.destroy()

propagator_chroma = io.readQIOPropagator("pt_prop_1")
propagator_chroma.toDevice()
print(cp.linalg.norm(propagator.data - propagator_chroma.data))
