import os
import sys
import numpy as np
import cupy as cp

test_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(test_dir, ".."))

import pyquda
from pyquda import core
from pyquda.utils import source, gauge_utils

os.environ["QUDA_RESOURCE_PATH"] = ".cache"
pyquda.init()

latt_size = [16, 16, 16, 128]
Lx, Ly, Lz, Lt = latt_size
rho = 2.0
nsteps = 5
x, y, z, t = 0, 0, 0, 0

filename = "/dg_hpc/LQCD/gongming/productions/confs/light.20200720.b20.16_128/s1.0_cfg_1000.lime"

xi = 5.2
xi_0 = 5.65
nu = xi_0 / xi
u_s = 0.780268
dslash = core.getDslash(latt_size, 0, 0, 0, xi_0, nu / u_s, anti_periodic_t=False)
gauge = gauge_utils.readIldg(filename)
dslash.loadGauge(gauge)

sh_src12 = source.source12(latt_size, "gaussian", [x, y, z, t], rho=rho, nsteps=nsteps, xi=xi * u_s)

data = sh_src12.lexico()

# def Laplacian(F, U, U_dag, sigma):
#     return (
#         (1 - sigma * 6) * F + sigma * (
#             cp.einsum("zyxab,zyxsb->zyxsa", U[0], cp.roll(F, -1, 2)) +
#             cp.einsum("zyxab,zyxsb->zyxsa", U[1], cp.roll(F, -1, 1)) +
#             cp.einsum("zyxab,zyxsb->zyxsa", U[2], cp.roll(F, -1, 0)) +
#             cp.roll(cp.einsum("zyxab,zyxsb->zyxsa", U_dag[0], F), 1, 2) +
#             cp.roll(cp.einsum("zyxab,zyxsb->zyxsa", U_dag[1], F), 1, 1) +
#             cp.roll(cp.einsum("zyxab,zyxsb->zyxsa", U_dag[2], F), 1, 0)
#         )
#     )

# from pyquda.field import Ns, Nc
# from pyquda.core import lexico
# gauge = cp.asarray(lexico(gauge.data, [1, 2, 3, 4, 5]))
# pt_src = cp.zeros((Lt, Lz, Ly, Lx, Nc, Nc))
# pt_src = source.source12(latt_size, "point", [x, y, z, t])
# pt_src.data = cp.asarray(pt_src.lexico())
# U = gauge[:, t].copy()
# U_dag = U.conj().transpose(0, 1, 2, 3, 5, 4)
# for step in range(nsteps):
#     for color in range(Nc):
#         for spin in range(Ns):
#             F = pt_src.data[t, :, :, :, :, spin, :, color].copy()
#             pt_src.data[t, :, :, :, :, spin, :, color] = Laplacian(F, U, U_dag, rho**2 / 4 / nsteps)
# data_cupy = pt_src.data.get()

# print(np.linalg.norm(data - data_cupy))
