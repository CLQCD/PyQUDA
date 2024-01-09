import os
import sys
import numpy as np

test_dir = os.path.dirname(os.path.abspath(__file__))
# sys.path.insert(1, os.path.join(test_dir, ".."))
from pyquda import core, quda, init
from pyquda.utils import io
from pyquda.field import LatticeInfo

os.environ["QUDA_RESOURCE_PATH"] = ".cache"

init()
latt_info = LatticeInfo([4, 4, 4, 8])

dslash = core.getDslash(latt_info.size, 0, 0, 0, anti_periodic_t=False)
gauge = io.readQIOGauge(os.path.join(test_dir, "weak_field.lime"))


quda.computeGaugeFixingOVRQuda(gauge.data_ptrs, 4, 1000, 1, 1.0, 2e-15, 1, 1, dslash.gauge_param)

land_gauge = io.readQIOGauge("coul_cfg.lime")
print(np.linalg.norm(land_gauge.data - gauge.data))
