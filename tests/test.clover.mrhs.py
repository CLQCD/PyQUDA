import cupy as cp

from check_pyquda import weak_field

from pyquda import init
from pyquda_utils import core, io, source

xi_0, nu = 2.464, 0.95
kappa = 0.115
mass = 1 / (2 * kappa) - 4
coeff = 1.17
coeff_r, coeff_t = 0.91, 1.07

init(None, [4, 4, 4, 8], -1, xi_0 / nu, resource_path=".cache")
latt_info = core.getDefaultLattice()

dslash = core.getDefaultDirac(mass, 1e-12, 1000, xi_0, coeff_t, coeff_r)
gauge = io.readQIOGauge(weak_field)

dslash.loadGauge(gauge)

point_source = source.multiFermion(latt_info, "point", [0, 0, 0, 0])
propagator = dslash.invertMultiSrc(point_source).toPropagator()

dslash.destroy()

propagator_chroma = io.readQIOPropagator("pt_prop_1")
propagator_chroma.toDevice()
print(cp.linalg.norm(propagator.data - propagator_chroma.data))
