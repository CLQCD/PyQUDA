from check_pyquda import weak_field, data

from pyquda_utils import core, io, source

xi_0, nu = 2.464, 0.95
kappa = 0.115
mass = 1 / (2 * kappa) - 4
coeff_r, coeff_t = 0.91, 1.07

core.init(None, [4, 4, 4, 8], -1, xi_0 / nu, resource_path=".cache")

gauge = io.readQIOGauge(weak_field)

rho, n_steps = 2.0, 5
point_source = source.propagator(core.getDefaultLattice(), "point", [0, 0, 0, 0])
shell_source = source.gaussianSmear(point_source, gauge, rho, n_steps)

dslash = core.getDefaultDirac(mass, 1e-12, 1000, xi_0, coeff_t, coeff_r)
dslash.loadGauge(gauge)
propagator = core.invertPropagator(dslash, shell_source)
dslash.destroy()

propagator_chroma = io.readQIOPropagator(data("pt_prop_4"))
propagator_chroma.toDevice()
print((propagator - propagator_chroma).norm2() ** 0.5)
