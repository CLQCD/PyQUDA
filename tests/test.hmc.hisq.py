from math import exp
from random import random
from time import perf_counter

from check_pyquda import test_dir

from pyquda import init, core
from pyquda.hmc import StaggeredHMC, O2Nf2Ng0P
from pyquda.action import symanzik_gauge, two_flavor_hisq
from pyquda.utils.io import readMILCGauge, writeNPYGauge

import cupy as cp

beta, u_0 = 7.4 / (5 / 3) * 0.890**4, 0.890
tol, maxiter = 1e-6, 1000
start, stop, warm, save = 0, 500, 500, 500
t = 0.2

init(resource_path=".cache", enable_force_monitor=True)
latt_info = core.LatticeInfo([6, 6, 6, 6], t_boundary=-1, anisotropy=1.0)

monomials = [
    symanzik_gauge.SymanzikGauge(latt_info, beta, u_0),
    # two_flavor_hisq.TwoFlavorHISQ(latt_info, 0.05, tol, maxiter),
]

hmc = StaggeredHMC(latt_info, monomials, O2Nf2Ng0P(5))
gauge = readMILCGauge("/home/jiangxy/milc_qcd/binary_samples/lat.sample.l6666.hisq")
gauge.toDevice()
gauge.projectSU3(2e-15)
hmc.initialize(gauge)

print("\n" f"Trajectory {start}:\n" f"Plaquette = {hmc.plaquette()}\n")
for i in range(start, stop):
    s = perf_counter()

    hmc.gaussMom(i)
    hmc.samplePhi(i)

    kinetic_old, potential_old = hmc.actionMom(), hmc.actionGauge()
    energy_old = kinetic_old + potential_old

    hmc.integrate(t)
    hmc.reunitGauge(2e-15)

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
