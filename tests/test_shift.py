from check_pyquda import weak_field

from pyquda_utils import core, convert
from pyquda_utils.core import X, Y, Z, T
import cupy as cp
from time import perf_counter

core.init(None, [4, 4, 4, 8], resource_path=".cache")

latt_info = core.LatticeInfo([4, 4, 4, 8])
unit = core.LatticeGauge(latt_info)
unit.gauge_dirac.loadGauge(unit)
propagator = core.LatticePropagator(latt_info, cp.random.random((latt_info.volume, 4, 4, 3, 3 * 2), "<f8").view("<c16"))
multi_fermion = convert.propagatorToMultiFermion(propagator)

for n, mu in zip([1, 1], [-X, T]):
    propagator_covdev = core.LatticePropagator(latt_info)
    cp.cuda.runtime.deviceSynchronize()
    s = perf_counter()
    for spin in range(4):
        for color in range(3):
            propagator_covdev.setFermion(unit.gauge_dirac.covDev(propagator.getFermion(spin, color), mu), spin, color)
    cp.cuda.runtime.deviceSynchronize()
    e = perf_counter()
    print(f"Time for GaugeField.covDev(LatticePropagator) on n={n}, mu={mu}:", e - s)

    multi_fermion_covdev = convert.propagatorToMultiFermion(propagator)
    cp.cuda.runtime.deviceSynchronize()
    s = perf_counter()
    for spin in range(4):
        for color in range(3):
            multi_fermion_covdev[spin * 3 + color] = unit.gauge_dirac.covDev(multi_fermion[spin * 3 + color], mu)
    cp.cuda.runtime.deviceSynchronize()
    e = perf_counter()
    print(f"Time for GaugeField.covDev(MultiLatticeFermion) on n={n}, mu={mu}:", e - s)
    multi_fermion_covdev = convert.multiFermionToPropagator(multi_fermion_covdev)

    multi_fermion_shift = core.MultiLatticeFermion(latt_info, 12)
    cp.cuda.runtime.deviceSynchronize()
    s = perf_counter()
    for spin in range(4):
        for color in range(3):
            multi_fermion_shift[spin * 3 + color] = multi_fermion[spin * 3 + color].shift(n, mu)
    cp.cuda.runtime.deviceSynchronize()
    e = perf_counter()
    print(f"Time for MultiLatticeFermion.shift on n={n}, mu={mu}:", e - s)
    multi_fermion_shift = convert.multiFermionToPropagator(multi_fermion_shift)

    cp.cuda.runtime.deviceSynchronize()
    s = perf_counter()
    propagator_shift = propagator.shift(n, mu)
    cp.cuda.runtime.deviceSynchronize()
    e = perf_counter()
    print(f"Time for LatticePropagator.shift on n={n}, mu={mu}:", e - s)

    print(
        "Difference between propagator_covdev and propagator_shift",
        (propagator_covdev - propagator_shift).norm2() ** 0.5,
    )
    print(
        "Difference between propagator_covdev and multi_fermion_covdev",
        (propagator_covdev - multi_fermion_covdev).norm2() ** 0.5,
    )
    print(
        "Difference between propagator_covdev and multi_fermion_shift",
        (propagator_shift - multi_fermion_shift).norm2() ** 0.5,
    )
