from check_pyquda import weak_field, data

from pyquda_utils import core, io

xi_0, nu = 2.464, 0.95
kappa = 0.125
mass = 1 / (2 * kappa) - 4

core.init(None, [4, 4, 4, 8], resource_path=".cache")

gauge = io.readQIOGauge(weak_field)
latt_info = core.LatticeInfo([4, 4, 4, 8], -1, xi_0 / nu)
dirac = core.getWilson(latt_info, mass, 1e-12, 1000)

with dirac.useGauge(gauge):
    propagator = core.invert(dirac, "point", [0, 0, 0, 0])

propagator_chroma = io.readQIOPropagator(data("pt_prop_0"))
propagator_chroma.toDevice()
print((propagator - propagator_chroma).norm2() ** 0.5)
