from math import exp
from random import random
from time import perf_counter

from check_pyquda import test_dir

from pyquda import init, core
from pyquda.hmc import HMC, O4Nf5Ng0V
from pyquda.action import wilson_gauge
from pyquda.utils.io import writeNPYGauge

beta, u_0 = 6.0, 1.0
start, stop, warm, save = 0, 2000, 500, 5
t, n_steps = 1.0, 10

init(resource_path=".cache")
latt_info = core.LatticeInfo([4, 4, 4, 8], t_boundary=1, anisotropy=1.0)

monomials = [
    wilson_gauge.WilsonGauge(latt_info, beta, u_0),
]

hmc = HMC(latt_info, monomials, O4Nf5Ng0V)
gauge = core.LatticeGauge(latt_info)
hmc.initialize(gauge)

print("\n" f"Trajectory {start}:\n" f"Plaquette = {hmc.plaquette()}\n")
for i in range(start, stop):
    s = perf_counter()

    hmc.gaussMom(i)
    hmc.samplePhi(i)

    kinetic_old, potential_old = hmc.actionMom(), hmc.actionGauge()
    energy_old = kinetic_old + potential_old

    hmc.integrate(t, n_steps)
    hmc.reunitGauge(1e-15)

    kinetic, potential = hmc.actionMom(), hmc.actionGauge()
    energy = kinetic + potential

    accept = random() < exp(energy_old - energy)
    if accept or i < warm:
        hmc.saveGauge(gauge)
    else:
        hmc.loadGauge(gauge)

    print(
        f"Trajectory {i + 1}:\n"
        f"Plaquette = {hmc.plaquette()}\n"
        f"P_old = {potential_old}, K_old = {kinetic_old}\n"
        f"P = {potential}, K = {kinetic}\n"
        f"Delta_P = {potential - potential_old}, Delta_K = {kinetic - kinetic_old}\n"
        f"Delta_E = {energy - energy_old}\n"
        f"acceptance rate = {min(1, exp(energy_old - energy)) * 100:.2f}%\n"
        f"accept? {accept or i < warm}\n"
        f"HMC time = {perf_counter() - s:.3f} secs\n"
    )

    if (i + 1) % save == 0:
        writeNPYGauge(f"./DATA/cfg/cfg_{i + 1}.npy", gauge)
