from math import exp
from time import perf_counter

from check_pyquda import test_dir

from pyquda.hmc import HMC, INT_3G1F
from pyquda.action import GaugeAction, HISQAction
from pyquda.enum_quda import QudaVerbosity
from pyquda_utils import core
from pyquda_utils.hmc_param import (
    symanzikTreeGaugeLoopParam as loopParam,
    staggeredFermionRationalParam as rationalParam,
)
from pyquda_utils.io import readMILCGauge, writeNPYGauge

beta, u_0 = 7.3, 0.880
tol, maxiter = 1e-6, 2500
start, stop, warm, save = 0, 1000, 500, 5
t = 0.48

core.init([1, 1, 1, 2], resource_path=".cache", enable_force_monitor=True)
latt_info = core.LatticeInfo([4, 4, 4, 12], t_boundary=-1, anisotropy=1.0)

monomials = [
    GaugeAction(latt_info, loopParam(u_0), beta),
    HISQAction(latt_info, rationalParam((0.0012, 0.0323, 0.2), (2, 1, -3), 9, 11, 1e-15, 90, 100), 100 * tol, maxiter),
    HISQAction(latt_info, rationalParam((0.2,), (1,), 7, 9, 1e-15, 90, 75), tol, maxiter),
    HISQAction(latt_info, rationalParam((0.2,), (1,), 7, 9, 1e-15, 90, 75), tol, maxiter),
    HISQAction(latt_info, rationalParam((0.2,), (1,), 7, 9, 1e-15, 90, 75), tol, maxiter),
    HISQAction(latt_info, rationalParam((0.432,), (1,), 7, 9, 1e-15, 90, 75), tol, maxiter, naik_epsilon=-0.116203),
]

hmc = HMC(latt_info, monomials, INT_3G1F(24))
hmc.setFermionVerbosity(QudaVerbosity.QUDA_SILENT)
gauge = core.LatticeGauge(latt_info)
# gauge = readMILCGauge("./s16t32_beta7.3_ml0.0012ms0.0323mc0.432.600")
hmc.initialize(10086, gauge)

plaq = hmc.plaquette()
core.getLogger().info(f"Trajectory {start}:\n" f"Plaquette = {plaq}\n")
for i in range(start, stop):
    s = perf_counter()

    hmc.gaussMom()
    hmc.samplePhi()

    kinetic_old, potential_old = hmc.momAction(), hmc.gaugeAction() + hmc.fermionAction()
    energy_old = kinetic_old + potential_old

    hmc.integrate(t, 2e-15)

    kinetic, potential = hmc.momAction(), hmc.gaugeAction() + hmc.fermionAction()
    energy = kinetic + potential

    accept = hmc.accept(energy - energy_old)
    if accept or i < warm:
        hmc.saveGauge(gauge)
    else:
        hmc.loadGauge(gauge)

    plaq = hmc.plaquette()
    core.getLogger().info(
        f"Trajectory {i + 1}:\n"
        f"Plaquette = {plaq}\n"
        f"P_old = {potential_old}, K_old = {kinetic_old}\n"
        f"P = {potential}, K = {kinetic}\n"
        f"Delta_P = {potential - potential_old}, Delta_K = {kinetic - kinetic_old}\n"
        f"Delta_E = {energy - energy_old}\n"
        f"acceptance rate = {exp(min(energy_old - energy, 0)) * 100:.2f}%\n"
        f"accept? {accept}\n"
        f"warmup? {i < warm}\n"
        f"HMC time = {perf_counter() - s:.3f} secs\n"
    )

    if (i + 1) % save == 0:
        writeNPYGauge(f"./DATA/cfg/cfg_{i + 1}.npy", gauge)
