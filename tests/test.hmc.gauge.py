from time import perf_counter

import numpy as np

from check_pyquda import test_dir

from pyquda import init
from pyquda.hmc import HMC, O4Nf5Ng0V
from pyquda.action import symanzik_gauge
from pyquda.field import LatticeInfo, LatticeGauge

init(resource_path=".cache")
latt_info = LatticeInfo([16, 16, 16, 32], t_boundary=1, anisotropy=1.0)

monomials = [
    symanzik_gauge.SymanzikGauge(latt_info, beta=6.2, u_0=0.855453),
]
gauge = LatticeGauge(latt_info)

hmc = HMC(latt_info, monomials, O4Nf5Ng0V)
hmc.setVerbosity(0)
hmc.initialize(gauge)

start = 0
stop = 2000
warm = 500
save = 5

print("\n" f"Trajectory {start}:\n" f"plaquette = {hmc.plaquette()}\n")

t = 1.0
steps = 10
for i in range(start, stop):
    s = perf_counter()

    hmc.gaussMom(i)
    hmc.samplePhi(i)

    kinetic = hmc.actionMom()
    potential = hmc.actionGauge()
    energy = kinetic + potential

    hmc.integrate(t, steps)
    hmc.reunitGauge(1e-15)

    kinetic1 = hmc.actionMom()
    potential1 = hmc.actionGauge()
    energy1 = kinetic1 + potential1

    accept = np.random.rand() < np.exp(energy - energy1)
    if accept or i < warm:
        hmc.saveGauge(gauge)
    else:
        hmc.loadGauge(gauge)

    print(
        f"Trajectory {i + 1}:\n"
        f"plaquette = {hmc.plaquette()}\n"
        f"PE_old = {potential}, KE_old = {kinetic}\n"
        f"PE = {potential1}, KE = {kinetic1}\n"
        f"Delta_PE = {potential1 - potential}, Delta_KE = {kinetic1 - kinetic}\n"
        f"Delta_E = {energy1 - energy}\n"
        f"accept rate = {min(1, np.exp(energy - energy1))*100:.2f}%\n"
        f"accept? {accept or i < warm}\n"
        f"HMC time = {perf_counter() - s:.3f} secs\n"
    )

    if (i + 1) % save == 0:
        np.save(f"./DATA/cfg/cfg_{i + 1}.npy", gauge.lexico())
