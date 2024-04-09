import cupy as cp

from check_pyquda import weak_field

from pyquda import core, init
from pyquda.utils import io

xi_0, nu = 2.464, 0.95
kappa = 0.115
mass = 1 / (2 * kappa) - 4
coeff = 1.17
coeff_r, coeff_t = 0.91, 1.07

init([1, 1, 1, 1], [4, 4, 4, 8], -1, xi_0 / nu, resource_path=".cache")

dslash = core.getDefaultDirac(mass, 1e-12, 1000, xi_0, coeff_t, coeff_r)
gauge = io.readQIOGauge(weak_field)

dslash.loadGauge(gauge)

propagator = core.invert(dslash, "point", [0, 0, 0, 0])

dslash.destroy()

propagator_chroma = io.readQIOPropagator("pt_prop_1")
propagator_chroma.toDevice()
print(cp.linalg.norm(propagator.data - propagator_chroma.data))
