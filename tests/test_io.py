from check_pyquda import weak_field

from pyquda_utils import core, io, convert, checksum

xi_0, nu = 2.464, 0.95
kappa = 0.115
mass = 1 / (2 * kappa) - 4
coeff = 1.17
coeff_r, coeff_t = 0.91, 1.07

core.init(None, [4, 4, 4, 8], -1, xi_0 / nu, resource_path=".cache")

dslash = core.getDefaultDirac(mass, 1e-12, 1000, xi_0, coeff_t, coeff_r)
gauge = io.readQIOGauge(weak_field)
dslash.loadGauge(gauge)
propagator = core.invert(dslash, "point", [0, 0, 0, 0])
dslash.destroy()

hash = [checksum.checksumSciDAC(gauge[i]) for i in range(4)]
print([(f"{i:08x}", f"{j:08x}") for i, j in hash])
gauge.saveH5("weak_field.h5")
propagator.saveH5("pt_prop_1.h5", 0)
convert.propagatorToMultiFermion(propagator).appendH5("pt_prop_1.h5", range(12))
gauge.save("weak_field.npy")
propagator.save("pt_prop_1.npy")

propagator.toHost()
gauge_h5 = core.LatticeGauge.loadH5("weak_field.h5")
propagator_h5 = core.LatticePropagator.loadH5("pt_prop_1.h5", 0)
multifermion_h5 = convert.multiFermionToPropagator(core.MultiLatticeFermion.loadH5("pt_prop_1.h5", range(12)))
fermion_h5 = core.LatticeFermion.loadH5("pt_prop_1.h5", 5)
propagator_chroma = io.readQIOPropagator("./tests/data/pt_prop_1")
assert isinstance(propagator_chroma, core.LatticePropagator)
gauge_npy = core.LatticeGauge.load("weak_field.npy")
propagator_npy = core.LatticePropagator.load("pt_prop_1.npy")
gauge_npy_old = io.readNPYGauge("weak_field.npy")
propagator_npy_old = io.readNPYPropagator("pt_prop_1.npy")
print((gauge_h5 - gauge).norm2() ** 0.5)
print((propagator - propagator_chroma).norm2() ** 0.5)
print((propagator_h5 - propagator_chroma).norm2() ** 0.5)
print((multifermion_h5 - propagator_chroma).norm2() ** 0.5)
print((fermion_h5 - propagator_chroma.getFermion(1, 2)).norm2() ** 0.5)
print((gauge_npy - gauge).norm2() ** 0.5)
print((propagator_npy - propagator).norm2() ** 0.5)
print((gauge_npy_old - gauge).norm2() ** 0.5)
print((propagator_npy_old - propagator).norm2() ** 0.5)
