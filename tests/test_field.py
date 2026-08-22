from math import exp
from time import perf_counter

from check_pyquda import data

from pyquda.hmc import HMC, INT_3G1F
from pyquda.action import GaugeAction, HISQAction
from pyquda_utils import core, io
from pyquda_utils.hmc_param import (
    symanzikOneLoopGaugeLoopParam as loopParam,
    staggeredFermionRationalParam as rationalParam,
)

beta, u_0 = 6.00, 0.86372
tol, maxiter = 1e-7, 2500
start, stop, warm, save = 0, 1000, 300, 5
t, n_steps = 1.0, 28

core.init(None, [24, 24, 24, 64], resource_path=".cache/quda", enable_force_monitor=True)
latt_info = core.LatticeInfo([24, 24, 24, 64], -1, 1.0)

ml, ms, mc = 0.0102, 0.0509, 0.635
# naik_epsilon = (-27 / 40) * mc**2 + (327 / 1120) * mc**4 - (15607 / 268800) * mc**6 - (73697 / 3942400) * mc**8
naik_epsilon = (-27 / 40) * mc**2 + (327 / 1120) * mc**4 - (5843 / 53760) * mc**6 + (153607 / 3942400) * mc**8

monomials = [
    GaugeAction(latt_info, loopParam(u_0, "hisq", 4), beta),
    HISQAction(latt_info, rationalParam((ml, ms, 0.2), (2, 1, -3), 7, 9, 1e-15, 90, 100), 100 * tol, maxiter),
    HISQAction(latt_info, rationalParam((0.2,), (1,), 6, 8, 1e-15, 90, 75), tol, maxiter),
    HISQAction(latt_info, rationalParam((0.2,), (1,), 6, 8, 1e-15, 90, 75), tol, maxiter),
    HISQAction(latt_info, rationalParam((0.2,), (1,), 6, 8, 1e-15, 90, 75), tol, maxiter),
    HISQAction(latt_info, rationalParam((mc,), (1,), 5, 7, 1e-15, 90, 75), tol, maxiter, naik_epsilon=naik_epsilon),
]

hmc = HMC(latt_info, monomials, INT_3G1F(n_steps))
if start == 0:
    gauge = core.LatticeGauge(latt_info)
else:
    gauge = io.readKYUGauge(data(f"a12m310/l2464f211b600m0102m0509m635a.{start}"), latt_info.global_size)
    gauge.toDevice()
hmc.initialize(10086 + start, gauge)

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
        io.writeKYUGauge(data(f"a12m310/l2464f211b600m0102m0509m635a.{i + 1}"), gauge)
