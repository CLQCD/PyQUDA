from time import perf_counter

import numpy as np

from check_pyquda import test_dir

from pyquda import init, setGPUID
from pyquda.hmc import HMC
from pyquda.action import symanzik_gauge
from pyquda.field import LatticeInfo, LatticeGauge

setGPUID(0)
init(resource_path=".cache")
beta = 6.2
u_0 = 0.855453
latt_info = LatticeInfo([16, 16, 16, 32], 1, 1.0)

monomials = [symanzik_gauge.SymanzikGauge(latt_info, beta, u_0)]
gauge = LatticeGauge(latt_info, None)

hmc = HMC(latt_info, monomials)
hmc.loadGauge(gauge)
hmc.loadMom(gauge)

# input_path = [
#     [0, 1, 7, 6],
#     [0, 2, 7, 5],
#     [0, 3, 7, 4],
#     [1, 2, 6, 5],
#     [1, 3, 6, 4],
#     [2, 3, 5, 4],
# ]
# input_coeffs = [
#     -1,
#     -1,
#     -1,
#     -1,
#     -1,
#     -1,
# ]

rho_ = 0.2539785108410595
theta_ = -0.03230286765269967
vartheta_ = 0.08398315262876693
lambda_ = 0.6822365335719091

plaquette = hmc.plaquette()
print(f"\nplaquette = {plaquette}\n")

t = 1.0
steps = 10
dt = t / steps
warm = 500
for i in range(2000):
    s = perf_counter()

    hmc.gaussMom(i)

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
    if warm > 0:
        warm -= 1
    if accept or warm:
        hmc.saveGauge(gauge)
    else:
        hmc.loadGauge(gauge)

    plaquette = hmc.plaquette()

    print(
        f"Step {i}:\n"
        f"PE_old = {potential}, KE_old = {kinetic}\n"
        f"PE = {potential1}, KE = {kinetic1}\n"
        f"Delta_PE = {potential1 - potential}, Delta_KE = {kinetic1 - kinetic}\n"
        f"Delta_E = {energy1 - energy}\n"
        f"accept rate = {min(1, np.exp(energy - energy1))*100:.2f}%\n"
        f"accept? {accept or not not warm}\n"
        f"plaquette = {plaquette}\n"
        f"HMC time = {perf_counter() - s:.3f} secs\n"
    )
