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

G4 = gamma.gamma(8)
G5 = gamma.gamma(15)
t_src_list = list(range(0, 72, 72))
pion = cp.zeros((len(t_src_list), latt_info.Lt), "<c16")
pionA4 = cp.zeros((len(t_src_list), latt_info.Lt), "<c16")

for t_idx, t_src in enumerate(t_src_list):
    propag = core.invert(dirac, "wall", t_src)

    pion[t_idx] += contract(
        "wtzyxjiba,jk,wtzyxklba,li->t",
        propag.data.conj(),
        G5 @ G5,
        propag.data,
        G5 @ G5,
    )

    pionA4[t_idx] += contract(
        "wtzyxjiba,jk,wtzyxklba,li->t",
        propag.data.conj(),
        G5 @ G5 @ G4,
        propag.data,
        G5 @ G5,
    )

dirac.destroy()

tmp = core.gatherLattice(pion.real.get(), [1, -1, -1, -1])
tmpA4 = core.gatherLattice(pionA4.real.get(), [1, -1, -1, -1])
if latt_info.mpi_rank == 0:
    for t_idx, t_src in enumerate(t_src_list):
        tmp[t_idx] = np.roll(tmp[t_idx], -t_src, 0)
        tmpA4[t_idx] = np.roll(tmpA4[t_idx], -t_src, 0)
    twopt = tmp.mean(0)
    twoptA4 = tmpA4.mean(0)
    ratio = twoptA4 / twopt
    plt.plot(np.arange(72), ratio, ",-")
    plt.savefig("pion_pcac.svg")
    plt.clf()
