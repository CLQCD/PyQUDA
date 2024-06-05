from time import perf_counter

import numpy as np

from pyquda import init
from pyquda.hmc import HMC, O4Nf5Ng0V
from pyquda.action import symanzik_gauge, two_flavor_clover, one_flavor_clover
from pyquda.field import LatticeInfo, LatticeGauge

init(resource_path=".cache")
latt_info = LatticeInfo([16, 16, 16, 32], t_boundary=-1, anisotropy=1.0)

monomials = [
    symanzik_gauge.SymanzikGauge(latt_info, beta=6.2, u_0=0.855453),
    two_flavor_clover.TwoFlavorClover(latt_info, mass=-0.2770, tol=1e-9, maxiter=1000, clover_csw=1.160920226),
    one_flavor_clover.OneFlavorClover(latt_info, mass=-0.2400, tol=1e-9, maxiter=1000, clover_csw=1.160920226),
]
gauge = LatticeGauge(latt_info)

hmc = HMC(latt_info, monomials=monomials, integrator=O4Nf5Ng0V)
hmc.setVerbosity(0)
hmc.loadGauge(gauge)
hmc.loadMom(gauge)

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

    kinetic_old = hmc.actionMom()
    potential_old = hmc.actionGauge()
    energy_old = kinetic_old + potential_old

    hmc.integrate(t, steps)
    hmc.reunitGauge(1e-15)

    kinetic = hmc.actionMom()
    potential = hmc.actionGauge()
    energy = kinetic + potential

    accept = np.random.rand() < np.exp(energy_old - energy)
    if accept or i < warm:
        hmc.saveGauge(gauge)
    else:
        hmc.loadGauge(gauge)

    print(
        f"Trajectory {i + 1}:\n"
        f"plaquette = {hmc.plaquette()}\n"
        f"P_old = {potential_old}, K_old = {kinetic_old}\n"
        f"P = {potential}, K = {kinetic}\n"
        f"Delta_P = {potential - potential_old}, Delta_K = {kinetic - kinetic_old}\n"
        f"Delta_E = {energy - energy_old}\n"
        f"accept rate = {min(1, np.exp(energy_old - energy))*100:.2f}%\n"
        f"accept? {accept or i < warm}\n"
        f"HMC time = {perf_counter() - s:.3f} secs\n"
    )

    if (i + 1) % save == 0:
        np.save(f"./DATA/cfg/cfg_{i + 1}.npy", gauge.lexico())
