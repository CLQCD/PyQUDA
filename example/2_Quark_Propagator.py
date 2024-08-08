import numpy as np
import cupy as cp
from opt_einsum import contract
from matplotlib import pyplot as plt

from pyquda import init, core, LatticeInfo
from pyquda.utils import io, gamma, source
from pyquda.field import LatticePropagator, Ns, Nc

init([1, 1, 1, 2], resource_path=".cache")

latt_info = LatticeInfo([24, 24, 24, 72], -1, 1.0)
dirac = core.getDirac(latt_info, -0.2770, 1e-12, 1000, 1.0, 1.160920226, 1.160920226, [[6, 6, 6, 4], [4, 4, 4, 9]])
gauge = io.readChromaQIOGauge("/public/ensemble/C24P29/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_48000.lime")
gauge.stoutSmear(1, 0.125, 4)
dirac.loadGauge(gauge)

G5 = gamma.gamma(15)
t_src_list = list(range(0, 72, 1))
pion = cp.zeros((len(t_src_list), latt_info.Lt), "<f8")

propag = LatticePropagator(latt_info)
for t_idx, t_src in enumerate(t_src_list):
    for spin in range(Ns):
        for color in range(Nc):
            b = source.wall(latt_info, t_src, spin, color)
            x = dirac.invert(b)
            propag.setFermion(x, spin, color)
    pion[t_idx] += contract("wtzyxjiba,wtzyxjiba->t", propag.data.conj(), propag.data).real
dirac.destroy()

tmp = core.gatherLattice(pion.get(), [1, -1, -1, -1])
if latt_info.mpi_rank == 0:
    for t_idx, t_src in enumerate(t_src_list):
        tmp[t_idx] = np.roll(tmp[t_idx], -t_src, 0)
    twopt = tmp.mean(0)
    plt.plot(np.arange(72), twopt, ",-")
    plt.yscale("log")
    plt.savefig("pion_2pt.svg")
    np.save("pion.npy", tmp)
    print(tmp)
