from itertools import permutations
import numpy as np
import cupy as cp
from opt_einsum import contract
from matplotlib import pyplot as plt

from pyquda import init, core, LatticeInfo
from pyquda.utils import io, gamma

init([1, 1, 1, 2], resource_path=".cache")

latt_info = LatticeInfo([24, 24, 24, 72], -1, 1.0)
dirac = core.getDirac(latt_info, -0.2770, 1e-12, 1000, 1.0, 1.160920226, 1.160920226, [[6, 6, 6, 4], [4, 4, 4, 9]])
gauge = io.readChromaQIOGauge("/public/ensemble/C24P29/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_48000.lime")
gauge.stoutSmear(1, 0.125, 4)
dirac.loadGauge(gauge)

C = gamma.gamma(2) @ gamma.gamma(8)
G0 = gamma.gamma(0)
G4 = gamma.gamma(8)
G5 = gamma.gamma(15)
P = cp.zeros((72, 4, 4), "<c16")
P[:36] = (G0 + G4) / 2
P[36:] = (G0 - G4) / 2
T = cp.ones((2 * 72), "<f8")
T[:] = -1
T[36 : 36 + 72] = 1
t_src_list = list(range(0, 72, 72))
pion = cp.zeros((len(t_src_list), latt_info.Lt), "<c16")
proton = cp.zeros((len(t_src_list), latt_info.Lt), "<c16")

for t_idx, t_src in enumerate(t_src_list):
    propag = core.invert(dirac, "wall", t_src)

    pion[t_idx] += contract(
        "wtzyxjiba,jk,wtzyxklba,li->t",
        propag.data.conj(),
        G5 @ G5,
        propag.data,
        G5 @ G5,
    )

    P_ = cp.roll(P, t_src, 0)[latt_info.gt * latt_info.Lt : (latt_info.gt + 1) * latt_info.Lt]
    T_ = T[72 - t_src : 2 * 72 - t_src][latt_info.gt * latt_info.Lt : (latt_info.gt + 1) * latt_info.Lt]
    for a, b, c in permutations(tuple(range(3))):
        for d, e, f in permutations(tuple(range(3))):
            sign = 1 if b == (a + 1) % 3 else -1
            sign *= 1 if e == (d + 1) % 3 else -1
            proton[t_idx] += (sign * T_) * (
                contract(
                    "ij,kl,tmn,wtzyxik,wtzyxjl,wtzyxmn->t",
                    C @ G5,
                    C @ G5,
                    P_,
                    propag.data[:, :, :, :, :, :, :, a, d],
                    propag.data[:, :, :, :, :, :, :, b, e],
                    propag.data[:, :, :, :, :, :, :, c, f],
                )
                + contract(
                    "ij,kl,tmn,wtzyxik,wtzyxjn,wtzyxml->t",
                    C @ G5,
                    C @ G5,
                    P_,
                    propag.data[:, :, :, :, :, :, :, a, d],
                    propag.data[:, :, :, :, :, :, :, b, e],
                    propag.data[:, :, :, :, :, :, :, c, f],
                )
            )

dirac.destroy()

tmp = core.gatherLattice(pion.real.get(), [1, -1, -1, -1])
if latt_info.mpi_rank == 0:
    for t_idx, t_src in enumerate(t_src_list):
        tmp[t_idx] = np.roll(tmp[t_idx], -t_src, 0)
    twopt = tmp.mean(0)
    plt.plot(np.arange(72), twopt, ",-")
    plt.yscale("log")
    plt.savefig("pion_2pt.svg")
    plt.clf()
    mass = np.arccosh((twopt[2:] + twopt[:-2]) / (2 * twopt[1:-1]))
    plt.plot(np.arange(1, 72 - 1), mass, ",-")
    # plt.ylim(0, 1)
    plt.savefig("pion_mass.svg")
    plt.clf()
    # np.save("pion.npy", tmp)
    print(tmp)
tmp = core.gatherLattice(proton.real.get(), [1, -1, -1, -1])
if latt_info.mpi_rank == 0:
    for t_idx, t_src in enumerate(t_src_list):
        tmp[t_idx] = np.roll(tmp[t_idx], -t_src, 0)
    twopt = tmp.mean(0)
    plt.plot(np.arange(72), twopt, ",-")
    plt.yscale("log")
    plt.savefig("proton_2pt.svg")
    plt.clf()
    mass = np.arccosh((twopt[2:] + twopt[:-2]) / (2 * twopt[1:-1]))
    plt.plot(np.arange(1, 72 - 1), mass, ",-")
    # plt.ylim(0, 1)
    plt.savefig("proton_mass.svg")
    plt.clf()
    # np.save("proton.npy", tmp)
    print(tmp)
