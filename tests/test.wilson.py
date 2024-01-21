import os
import cupy as cp

from check_pyquda import test_dir

from pyquda import core, init
from pyquda.utils import io

init(resource_path=".cache")

xi_0, nu = 2.464, 0.95
kappa = 0.135
mass = 1 / (2 * kappa) - 4

core.setDefaultLattice([4, 4, 4, 8], -1, xi_0 / nu)

dslash = core.getDiracDefault(mass, 1e-12, 1000, xi_0, multigrid=False)
gauge = io.readQIOGauge(os.path.join(test_dir, "weak_field.lime"))

dslash.loadGauge(gauge)

propagator = core.invert(dslash, "point", [0, 0, 0, 0])

dslash.destroy()

propagator_chroma = io.readQIOPropagator("pt_prop_0")
propagator_chroma.toDevice()
print(cp.linalg.norm(propagator.data - propagator_chroma.data))
