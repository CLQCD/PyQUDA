from time import perf_counter
import numpy as np
import cupy as cp

from check_pyquda import weak_field

from pyquda import core, init
from pyquda.utils import gamma, phase, io

xi_0, nu = 4.8965, 0.86679
mass = 0.09253
kappa = 0.5 / (mass + 1 + 3 / (xi_0 / nu))
coeff_r, coeff_t = 2.32582045, 0.8549165664

init([1, 1, 1, 1], [4, 4, 4, 8], -1, xi_0 / nu, resource_path=".cache")

latt_info = core.getDefaultLattice()
Lx, Ly, Lz, Lt = latt_info.size
Vol = latt_info.volume
Nc, Ns, Nd = 3, 4, 4

gamma1 = gamma.gamma(1)
gamma2 = gamma.gamma(2)
gamma3 = gamma.gamma(4)
gamma4 = gamma.gamma(8)
gamma5 = gamma.gamma(15)
gammai = [gamma1, gamma2, gamma3, gamma4]
gamma_insertion = [(gamma5, gamma5), (gamma4 @ gamma5, gamma4 @ gamma5)]

mom_list = phase.getMomList(9)
mom_num = len(mom_list)
mom_phase = phase.MomentumPhase(latt_info)
phase_list = mom_phase.getPhases(mom_list)

dslash = core.getDefaultDirac(mass, 1e-12, 1000, xi_0, coeff_t, coeff_r)
twopt = np.zeros((Lt, Lt, len(gamma_insertion), mom_num), "<c16")


s = perf_counter()
gauge = io.readQIOGauge(weak_field)
dslash.loadGauge(gauge)
print(f"Read and load gauge configuration: {perf_counter()-s:.2f}sec.")

for t in range(Lt):
    s = perf_counter()
    propagator = core.invert(dslash, "wall", t)
    print(f"Invertion for wall source at t={t}: {perf_counter()-s:.2f}sec.")

    s = perf_counter()
    gamma_idx = 0
    for gamma_src, gamma_snk in gamma_insertion:
        tmp = cp.einsum(
            "ij,xkjba,kl,xliba->x",
            gamma_src @ gamma5,
            propagator.data.reshape(Vol, Ns, Ns, Nc, Nc).conj(),
            gamma5 @ gamma_snk,
            propagator.data.reshape(Vol, Ns, Ns, Nc, Nc),
            optimize=True,
        )
        for p in range(mom_num):
            res = cp.einsum(
                "wtzyx,wtzyx->t",
                phase_list[p],
                tmp.reshape(2, Lt, Lz, Ly, Lx // 2),
                optimize=True,
            )
            res = cp.roll(res, -t)
            twopt[t, :, gamma_idx, p] = res.get()
        gamma_idx += 1
    print(f"Contraction for {len(gamma_insertion)} gamma insertions: {perf_counter()-s:.2f}sec.")
