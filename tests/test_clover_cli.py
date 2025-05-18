#!/usr/bin/env -S python3 -m pyquda -l 4 4 4 8 -t -1 -a 2.593684210526316 -p .cache
from tests.check_pyquda import weak_field, data

from pyquda_utils import core, io

xi_0, nu = 2.464, 0.95
kappa = 0.115
mass = 1 / (2 * kappa) - 4
coeff_r, coeff_t = 0.91, 1.07

gauge = io.readQIOGauge(weak_field)

dirac = core.getDefaultDirac(mass, 1e-12, 1000, xi_0, coeff_t, coeff_r)
dirac.loadGauge(gauge)
propagator = core.invert(dirac, "point", [0, 0, 0, 0])
dirac.destroy()

propagator_chroma = io.readQIOPropagator(data("pt_prop_1"))
propagator_chroma.toDevice()
print((propagator - propagator_chroma).norm2() ** 0.5)
