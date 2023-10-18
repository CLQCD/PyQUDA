import os
import sys
import numpy as np

test_dir = os.path.dirname(os.path.abspath(__file__))
# sys.path.insert(0, os.path.join(test_dir, ".."))
from pyquda import core, quda, init
from pyquda.utils import io

os.environ["QUDA_RESOURCE_PATH"] = ".cache"

latt_size = [4, 4, 4, 8]
Lx, Ly, Lz, Lt = latt_size
Vol = Lx * Ly * Lz * Lt
init()

dslash = core.getDslash(latt_size, 0, 0, 0, anti_periodic_t=False)
gauge = io.readQIOGauge(os.path.join(test_dir, "weak_field.lime"))


timeinfo = [0.0, 0.0, 0.0]
quda.computeGaugeFixingOVRQuda(gauge.data_ptrs, 4, 1000, 1, 1.0, 1e-15, 1, 1, dslash.gauge_param, timeinfo)
print(timeinfo)

land_gauge = io.readQIOGauge("coul_cfg.lime")
print(np.linalg.norm(land_gauge.data - gauge.data))
