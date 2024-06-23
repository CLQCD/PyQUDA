import sys
import numpy as np
import gvar as gv
from matplotlib import pyplot as plt

sys.path.insert(1, "/home/jiangxy/PyQUDA/")
from pyquda import init
from pyquda.field import Nc
from pyquda.utils import io

plt.set_loglevel(level="warning")
init(resource_path=".cache", enable_tuning=False)

with open("./cfglist.txt", "r") as f:
    cfglist = [line.strip() for line in f.readlines() if line != ""]


path = {
    1: [[0], [1], [2]],
    2: [[0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1]],
    3: [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]],
    4: [[0, 0], [1, 1], [2, 2]],
    5: [[0, 0, 1], [0, 0, 2], [1, 1, 0], [1, 1, 2], [2, 2, 0], [2, 2, 1]],
    6: [[0, 0, 1, 2]],
    8: [[0, 0, 1, 1]],
    9: [[0, 0, 1, 1, 2], [0, 0, 2, 2, 1], [1, 1, 0, 0, 2], [1, 1, 2, 2, 0], [2, 2, 0, 0, 1], [2, 2, 1, 1, 0]],
}

t_min = 4
t_max = 6
retr = np.zeros((len(cfglist), len(path), t_max + 1 - t_min), "<f8")
for idx, cfg in enumerate(cfglist):
    print(cfg)
    gauge = io.readChromaQIOGauge(cfg)

    # t0, w0 = gauge.wilsonFlowScale(1000, 0.01)
    # print(0.14464 / t0**0.5)
    # print(0.17355 / w0)

    i = -1
    for key, val in path.items():
        i += 1
        for choice in val:
            loops = []
            for t in range(t_min, t_max + 1):
                loops.append(
                    [dir for dir in choice]
                    + [3 for _ in range(t)]
                    + [dir + 4 for dir in choice]
                    + [7 for _ in range(t)]
                )
            retr[idx, i] += gauge.loopTrace(loops).real / (gauge.latt_info.global_volume * Nc)
        retr[idx, i] /= len(val)

ratio = retr[:, :, 1:] / retr[:, :, :-1]
ratio = gv.dataset.avg_data(ratio)
for i, key in enumerate(path.keys()):
    plt.errorbar(
        np.arange(t_min, t_max) + 0.5 + i * 0.1,
        gv.mean(ratio[i, :]),
        gv.sdev(ratio[i, :]),
        fmt="x",
        label=f"{key ** 0.5}",
    )
plt.legend()
plt.ylim(0, 1)
plt.show()
plt.clf()

r = np.array(list(path.keys())) ** 0.5
r0 = (r[1:] + r[:-1]) / 2
potential = -gv.log(ratio[:, 0])
r2force = ((r[1:] + r[:-1]) / 2) ** 2 * (potential[1:] - potential[:-1]) / (r[1:] - r[:-1])
plt.errorbar(r, gv.mean(potential), gv.sdev(potential), fmt="x")
plt.show()
plt.clf()
plt.errorbar(r0, gv.mean(r2force), gv.sdev(r2force), fmt="x")
plt.show()
plt.clf()
