from math import exp
from time import perf_counter

from check_pyquda import test_dir

from pyquda import init, getLogger
from pyquda.hmc import HMC, O4Nf5Ng0V
from pyquda.action import GaugeAction, CloverWilsonAction
from pyquda_utils import core
from pyquda_utils.hmc_param import symanzik_tree_gauge, wilson_rational_param
from pyquda_utils.io import writeNPYGauge

beta, u_0 = 7.4, 0.890
clover_csw = 1 / u_0**3
tol, maxiter = 1e-6, 1000
start, stop, warm, save = 0, 2000, 500, 5
t = 1.0

init([1, 1, 1, 1], resource_path=".cache", enable_force_monitor=True)
latt_info = core.LatticeInfo([4, 4, 4, 8], t_boundary=-1, anisotropy=1.0)

monomials = [
    GaugeAction(latt_info, symanzik_tree_gauge(u_0), beta, u_0),
    CloverWilsonAction(latt_info, wilson_rational_param[2], 0.3, 2, tol, maxiter, clover_csw),
    CloverWilsonAction(latt_info, wilson_rational_param[1], 0.5, 1, tol, maxiter, clover_csw),
]

# hmc_inner = HMC(latt_info, monomials[:1], O4Nf5Ng0V(4))
# hmc = HMC(latt_info, monomials[1:], O4Nf5Ng0V(3), hmc_inner)
hmc = HMC(latt_info, monomials, O4Nf5Ng0V(5))
gauge = core.LatticeGauge(latt_info)
hmc.initialize(10086, gauge)

plaq = hmc.plaquette()
getLogger().info(f"Trajectory {start}:\n" f"Plaquette = {plaq}\n")
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
