import os
import cupy as cp

from check_pyquda import test_dir

from pyquda import core, init
from pyquda.utils import source, io
from pyquda.field import LatticeInfo

init(resource_path=".cache")
latt_info = LatticeInfo([4, 4, 4, 8])

rho = 2.0
nsteps = 5
x, y, z, t = 0, 0, 0, 0

xi = 5.2
xi_0 = 5.65
nu = xi_0 / xi
u_s = 0.780268
dslash = core.getDslash(latt_info.size, 0, 0, 0, xi_0, nu / u_s, anti_periodic_t=False)
# dslash = core.getDslash(latt_info.size, 0, 0, 0, anti_periodic_t=False)  #* This is used for isotropic lattice
gauge = io.readQIOGauge(os.path.join(test_dir, "weak_field.lime"))
dslash.loadGauge(gauge)

shell_source = source.source12(latt_info.size, "gaussian", [0, 0, 0, 0], rho=2.0, nsteps=5, xi=xi * u_s)
# shell_source = source.source12(
#     latt_info.size, "gaussian", [0, 0, 0, 0], rho=2.0, nsteps=5
# )  # * This is used for isotropic lattice


shell_source_chroma = io.readQIOPropagator("pt_prop_4")
shell_source_chroma.toDevice()
print(cp.linalg.norm(shell_source.data - shell_source_chroma.data))

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
# pt_src = source.source12(latt_info.size, "point", [x, y, z, t])
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
