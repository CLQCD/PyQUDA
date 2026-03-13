from math import exp
from time import perf_counter

from pyquda.action import GaugeAction, abstract
from pyquda.hmc import HMC, Integrator
from pyquda_utils import core, io
from pyquda_utils.core import X, Y, Z, T

core.init(resource_path=".cache/quda")
latt_info = core.LatticeInfo([16, 16, 16, 32])

u_0 = 0.855453
beta = 6.20
loop_param = abstract.LoopParam(
    [
        [X, Y, -X, -Y],
        [X, Z, -X, -Z],
        [Y, Z, -Y, -Z],
        [X, T, -X, -T],
        [Y, T, -Y, -T],
        [Z, T, -Z, -T],
    ],
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
)
start, stop, warm, save = 0, 2000, 500, 5
t, n_steps = 1.0, 100


class O2Nf1Ng0V(Integrator):
    R"""https://doi.org/10.1016/S0010-4655(02)00754-3
    BAB: Eq.(23), \xi=0"""

    def integrate(self, updateGauge, updateMom, t: float):
        dt = t / self.n_steps
        for _ in range(self.n_steps):
            updateMom(dt / 2)
            updateGauge(dt)
            updateMom(dt / 2)


hmc = HMC(latt_info, [GaugeAction(latt_info, loop_param, beta)], O2Nf1Ng0V(n_steps))
gauge = core.LatticeGauge(latt_info)
hmc.initialize(10086, gauge)

plaq = hmc.plaquette()
core.getLogger().info(f"Trajectory {start}:\n" f"plaquette = {plaq}\n")

for i in range(start, stop):
    s = perf_counter()

    hmc.gaussMom()

    kinetic_old, potential_old = hmc.momAction(), hmc.gaugeAction() + hmc.fermionAction()
    energy_old = kinetic_old + potential_old

    hmc.integrate(t, 2e-14)

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
        f"acceptance rate = {min(1, exp(energy_old - energy)) * 100:.2f}%\n"
        f"accept? {accept or i < warm}\n"
        f"HMC time = {perf_counter() - s:.3f} secs\n"
    )

    if (i + 1) % save == 0:
        io.writeNPYGauge(f"./cfg/cfg_{i + 1}.npy", gauge)
