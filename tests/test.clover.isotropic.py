from check_pyquda import weak_field

from pyquda_utils import core, io

xi_0, nu = 1.0, 1.0
kappa = 0.115
mass = 1 / (2 * kappa) - 4
coeff_r, coeff_t = 1.17, 1.17

core.init(None, [4, 4, 4, 8], -1, xi_0 / nu, resource_path=".cache")

gauge = io.readQIOGauge(weak_field)

dslash = core.getDefaultDirac(mass, 1e-12, 1000, xi_0, coeff_t, coeff_r)
dslash.loadGauge(gauge)
propagator = core.invert(dslash, "point", [0, 0, 0, 0])
dslash.destroy()

propagator_chroma = io.readQIOPropagator("pt_prop_3")
propagator_chroma.toDevice()
print((propagator - propagator_chroma).norm2() ** 0.5)
