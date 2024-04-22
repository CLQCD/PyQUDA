from time import perf_counter

import numpy as np

from check_pyquda import test_dir

from pyquda import init, setGPUID
from pyquda.hmc import HMC
from pyquda.action import symanzik_gauge, one_flavor_clover, two_flavor_clover
from pyquda.field import LatticeInfo, LatticeGauge

setGPUID(0)
init(resource_path=".cache")
beta = 6.2
u_0 = 0.855453
mass_l = -0.2770
mass_s = -0.2400
csw = 1.160920226
latt_info = LatticeInfo([16, 16, 16, 32], -1, 1.0)

monomials = [
    symanzik_gauge.SymanzikGauge(latt_info, beta, u_0),
    one_flavor_clover.OneFlavorClover(latt_info, mass_s, 1e-9, 1000, csw),
    two_flavor_clover.TwoFlavorClover(latt_info, mass_l, 1e-9, 1000, csw),
]
gauge = LatticeGauge(latt_info, None)

hmc = HMC(latt_info, monomials)
hmc.setVerbosity(0)
hmc.loadGauge(gauge)
hmc.loadMom(gauge)

rho_ = 0.2539785108410595
theta_ = -0.03230286765269967
vartheta_ = 0.08398315262876693
lambda_ = 0.6822365335719091

start = 0
stop = 2000
warm = 500
save = 5

print("\n" f"Trajectory {start}:\n" f"plaquette = {hmc.plaquette()}\n")

t = 1.0
steps = 10
dt = t / steps
for i in range(start, stop):
    s = perf_counter()

    hmc.gaussMom(i)
    hmc.samplePhi(i)

    kinetic = hmc.actionMom()
    potential = hmc.actionGauge()
    energy = kinetic + potential

    for step in range(steps):
        hmc.updateMom(vartheta_ * dt)
        hmc.updateGauge(rho_ * dt)
        hmc.updateMom(lambda_ * dt)
        hmc.updateGauge(theta_ * dt)
        hmc.updateMom((0.5 - (lambda_ + vartheta_)) * dt)
        hmc.updateGauge((1.0 - 2 * (theta_ + rho_)) * dt)
        hmc.updateMom((0.5 - (lambda_ + vartheta_)) * dt)
        hmc.updateGauge(theta_ * dt)
        hmc.updateMom(lambda_ * dt)
        hmc.updateGauge(rho_ * dt)
        hmc.updateMom(vartheta_ * dt)

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
