import os
import sys
from time import time
import numpy as np
import cupy as cp

test_dir = os.path.dirname(os.path.abspath(__file__))
# sys.path.insert(0, os.path.join(test_dir, ".."))
from pyquda import core, mpi
from pyquda.utils import gamma, phase, gauge_utils

os.environ["QUDA_RESOURCE_PATH"] = ".cache"

Nc, Ns, Nd = 3, 4, 4

xi_0, nu = 4.8965, 0.86679
mass = 0.09253
coeff_r, coeff_t = 2.32582045, 0.8549165664

kappa = 0.5 / (mass + 1 + 3 / (xi_0 / nu))

latt_size = [4, 4, 4, 8]
Lx, Ly, Lz, Lt = latt_size
Vol = Lx * Ly * Lz * Lt

gamma1 = gamma.gamma(1)
gamma2 = gamma.gamma(2)
gamma3 = gamma.gamma(4)
gamma4 = gamma.gamma(8)
gamma5 = gamma.gamma(15)
gammai = [gamma1, gamma2, gamma3, gamma4]
gamma_insertion = [(gamma5, gamma5), (gamma4 @ gamma5, gamma4 @ gamma5)]

mom_list = phase.getMomList(9)
mom_num = len(mom_list)
mom_phase = phase.Phase(latt_size)
phase_list = mom_phase.cache(mom_list)

dslash = core.getDslash(latt_size, mass, 1e-9, 1000, xi_0, nu, coeff_t, coeff_r, multigrid=True)
twopt = np.zeros((Lt, Lt, len(gamma_insertion), mom_num), "<c16")

mpi.init()

s = time()
gauge = gauge_utils.readIldg(os.path.join(test_dir, "weak_field.lime"))
dslash.loadGauge(gauge)
print(f"Read and load gauge configuration: {time()-s:.2f}sec.")

for t in range(Lt):
    s = time()
    propagator = core.invert(dslash, "wall", t)
    print(f"Invertion for wall source at t={t}: {time()-s:.2f}sec.")

    s = time()
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
                "etzyx,etzyx->t",
                phase_list[p],
                tmp.reshape(2, Lt, Lz, Ly, Lx // 2),
                optimize=True,
            )
            res = cp.roll(res, -t)
            twopt[t, :, gamma_idx, p] = res.get()
        gamma_idx += 1
    print(f"Contraction for {len(gamma_insertion)} gamma insertions: {time()-s:.2f}sec.")
