import numpy as np
import gvar as gv

import sys

sys.path.insert(1, "/home/jiangxy/PyQUDA/")

from pyquda import init
from pyquda.field import LatticeInfo, LatticeGauge, cb2
from pyquda.utils import io

init(resource_path=".cache", enable_tuning=False)

with open("./cfglist.txt", "r") as f:
    cfglist = [line.strip() for line in f.readlines() if line != ""]

s_max = 6
t_max = 18
retr = np.zeros((len(cfglist), s_max, t_max), "<f8")
for idx, cfg in enumerate(cfglist):
    gauge = io.readChromaQIOGauge(cfg)

    # t0, w0 = gauge.wilsonFlowScale(1000, 0.01)
    # print(0.14464 / t0**0.5)
    # print(0.17355 / w0)

    for s in range(1, s_max + 1):
        for dir in range(3):
            loops = []
            for t in range(1, t_max + 1):
                loops.append(
                    [dir for _ in range(s)]
                    + [3 for _ in range(t)]
                    + [dir + 4 for _ in range(s)]
                    + [7 for _ in range(t)]
                )
            retr[idx, s - 1] += gauge.loopTrace(loops).real / gauge.latt_info.global_volume
        retr[idx, s - 1] /= 3
        print(retr[idx, s - 1])
retr = gv.dataset.avg_data(retr)
lnretr = gv.log(retr)
dtlnretr = lnretr[:, :-1] - lnretr[:, 1:]
print(dtlnretr)
# print(retr[:3].mean() / 3, retr[3:].mean() / 3)
# print(gauge.plaquette())
