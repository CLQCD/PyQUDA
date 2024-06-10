import numpy as np

from pyquda import init
from pyquda.field import LatticeInfo, LatticeGauge, cb2
from pyquda.utils import io

init(resource_path=".cache")

# latt_info = LatticeInfo([16, 16, 16, 32], t_boundary=-1, anisotropy=1.0)
# gauge = LatticeGauge(latt_info, cb2(np.load("./DATA/cfg/cfg_1210.npy"), [1, 2, 3, 4]))
gauge = io.readChromaQIOGauge("/public/ensemble/C24P29/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_10000.lime")

# t0, w0 = gauge.wilsonFlowScale(1000, 0.01)
# print(0.14464 / t0**0.5)
# print(0.17355 / w0)

retr = (
    gauge.loopTrace([[0, 3, 7, 4], [1, 3, 6, 4], [2, 3, 5, 4], [0, 1, 7, 6], [0, 2, 7, 5], [1, 2, 6, 5]]).real
    / gauge.latt_info.global_volume
)
print(retr[:3].mean() / 3, retr[3:].mean() / 3)
print(gauge.plaquette())
