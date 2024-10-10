from math import exp
from time import perf_counter

from check_pyquda import test_dir

from pyquda import init, getLogger, core
from pyquda.hmc import HMC, INT_3G1F
from pyquda.action import SymanzikTreeGauge, HISQFermion
from pyquda.utils.io import readMILCGauge, writeNPYGauge

beta, u_0 = 7.3, 0.880
tol, maxiter = 1e-6, 2500
start, stop, warm, save = 0, 2000, 500, 5
t = 0.48

init([1, 1, 1, 1], resource_path=".cache", enable_force_monitor=True)
latt_info = core.LatticeInfo([4, 4, 4, 8], t_boundary=-1, anisotropy=1.0)

monomials = [
    SymanzikTreeGauge(latt_info, beta, u_0),
    HISQFermion(latt_info, [0.0012, 0.0323, 0.2], [2, 1, -3], 100 * tol, maxiter),
    HISQFermion(latt_info, 0.2, 1, tol, maxiter),
    HISQFermion(latt_info, 0.2, 1, tol, maxiter),
    HISQFermion(latt_info, 0.2, 1, tol, maxiter),
    HISQFermion(latt_info, 0.432, 1, tol, maxiter, naik_epsilon=-0.116203),
]

hmc = HMC(latt_info, monomials, INT_3G1F(24))
gauge = core.LatticeGauge(latt_info)
# gauge = readMILCGauge("./s16t32_beta7.3_ml0.0012ms0.0323mc0.432.600")
# gauge.toDevice()
# gauge.projectSU3(2e-15)
hmc.initialize(10086, gauge)

plaq = hmc.plaquette()
getLogger().info(f"Trajectory {start}:\n" f"Plaquette = {plaq}\n")
for i in range(start, stop):
    s = perf_counter()

    hmc.gaussMom()
    hmc.samplePhi()

    kinetic_old, potential_old = hmc.actionMom(), hmc.actionGauge()
    energy_old = kinetic_old + potential_old

    hmc.integrate(t, 2e-15)

    kinetic, potential = hmc.actionMom(), hmc.actionGauge()
    energy = kinetic + potential

    accept = hmc.accept(energy - energy_old)
    if accept or i < warm:
        hmc.saveGauge(gauge)
    else:
        hmc.loadGauge(gauge)

    plaq = hmc.plaquette()
    getLogger().info(
        f"Trajectory {i + 1}:\n"
        f"Plaquette = {plaq}\n"
        f"P_old = {potential_old}, K_old = {kinetic_old}\n"
        f"P = {potential}, K = {kinetic}\n"
        f"Delta_P = {potential - potential_old}, Delta_K = {kinetic - kinetic_old}\n"
        f"Delta_E = {energy - energy_old}\n"
        f"acceptance rate = {min(1, exp(energy_old - energy)) * 100:.2f}%\n"
        f"accept? {accept}\n"
        f"warmup? {i < warm}\n"
        f"HMC time = {perf_counter() - s:.3f} secs\n"
    )

    if (i + 1) % save == 0:
        writeNPYGauge(f"./DATA/cfg/cfg_{i + 1}.npy", gauge)
