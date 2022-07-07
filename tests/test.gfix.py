import os
import sys
import numpy as np
import cupy as cp

test_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(test_dir, ".."))
from pyquda import core, quda
from pyquda.utils import gauge_utils

os.environ["QUDA_RESOURCE_PATH"] = ".cache"

latt_size = [4, 4, 4, 8]
Lx, Ly, Lz, Lt = latt_size
Vol = Lx * Ly * Lz * Lt

xi_0, nu = 2.464, 0.95
kappa = 0.135
mass = 1 / (2 * kappa) - 4

loader = core.QudaFieldLoader(latt_size, mass, 1e-9, 1000, xi_0, nu)
gauge = gauge_utils.readIldg(os.path.join(test_dir, "weak_field.lime"))

quda.initQuda(0)

loader.loadGauge(gauge)
timeinfo = np.zeros((10))
quda.computeGaugeFixingOVRQuda(gauge.data_ptrs, 3, 10000, 10, 1.8, 1e-9, 10, 1, loader.gauge_param, timeinfo)
print(timeinfo)

quda.endQuda()
