from check_pyquda import weak_field, data

from pyquda_utils import core, io

xi_0, nu = 1.0, 1.0
kappa = 0.115
mass = 1 / (2 * kappa) - 4
coeff_r, coeff_t = 1.17, 1.17

core.init(None, [4, 4, 4, 8], resource_path=".cache/quda")

gauge = io.readQIOGauge(weak_field)
latt_info = core.LatticeInfo([4, 4, 4, 8], -1, xi_0 / nu)
dirac = core.getClover(latt_info, mass, 1e-12, 1000, xi_0, coeff_t, coeff_r)

with dirac.useGauge(gauge) as d:
    propagator = core.invert(d, "point", [0, 0, 0, 0])

propagator_chroma = io.readQIOPropagator(data("pt_prop_3"))
propagator_chroma.toDevice()
print((propagator - propagator_chroma).norm2() ** 0.5)
