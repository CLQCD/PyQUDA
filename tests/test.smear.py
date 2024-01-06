import os
import sys
import cupy as cp


test_dir = os.path.dirname(os.path.abspath(__file__))
# sys.path.insert(1, os.path.join(test_dir, ".."))
from pyquda import init
from pyquda.utils import io
from pyquda.field import LatticeInfo

os.environ["QUDA_RESOURCE_PATH"] = ".cache"

init()
latt_info = LatticeInfo([4, 4, 4, 8])

gauge = io.readQIOGauge(os.path.join(test_dir, "weak_field.lime"))
gauge.smearSTOUT(1, 0.241, 3)
# gauge.setAntiPeroidicT()  # for fermion smearing

gauge_chroma = io.readQIOGauge("stout.lime")
print(cp.linalg.norm(gauge.data - gauge_chroma.data))
