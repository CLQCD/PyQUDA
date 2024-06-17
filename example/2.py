import sys

sys.path.insert(1, "/home/jiangxy/PyQUDA/")
import numpy as np
from pyquda import init
from pyquda.utils import io
from pyquda.field import LatticeInfo, LatticeGauge, LatticeFermion

init([1, 1, 1, 1], backend="torch", enable_tuning=False)
latt_info = LatticeInfo([24, 24, 24, 72], t_boundary=1, anisotropy=1.0)
gauge = io.readChromaQIOGauge("/public/ensemble/C24P29/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_10000.lime")

gauge_ape = gauge.copy()
gauge_ape.smearAPE(1, 2.5, 4)
gauge_tmp = io.readChromaQIOGauge("./ape.lime")
print(np.linalg.norm(gauge_ape.data - gauge_tmp.data))

gauge_stout = gauge.copy()
gauge_stout.stoutSmear(1, 0.125, 4)
gauge_tmp = io.readChromaQIOGauge("./stout.lime")
print(np.linalg.norm(gauge_stout.data - gauge_tmp.data))

gauge_hyp = gauge.copy()
gauge_hyp.hypSmear(1, 0.75, 0.6, 0.3, 4)
gauge_tmp = io.readChromaQIOGauge("./hyp.lime")
print(np.linalg.norm(gauge_hyp.data - gauge_tmp.data))

gauge_wflow = gauge.copy()
gauge_wflow.wilsonFlow(20, 0.01)
gauge_tmp = io.readChromaQIOGauge("./wflow.lime")
print(np.linalg.norm(gauge_wflow.data - gauge_tmp.data))

t0, w0 = gauge.wilsonFlowScale(1000, 0.01)
print(t0, w0)

print(gauge.plaquette())

print([value / 3 for value in gauge.polyakovLoop()])

retr = (
    gauge.loopTrace(
        [
            [0, 1, 4, 5],
            [0, 2, 4, 6],
            [1, 2, 5, 6],
            [0, 3, 4, 7],
            [1, 3, 5, 7],
            [2, 3, 6, 7],
        ]
    ).real
    / gauge.latt_info.global_volume
    / 3
)
print(retr.mean(), retr[:3].mean(), retr[3:].mean())
