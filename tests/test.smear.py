import os
import cupy as cp

from check_pyquda import test_dir

from pyquda import init
from pyquda.utils import io
from pyquda.field import LatticeInfo

init(resource_path=".cache")
latt_info = LatticeInfo([4, 4, 4, 8])

gauge = io.readQIOGauge(os.path.join(test_dir, "weak_field.lime"))
gauge.smearSTOUT(1, 0.241, 3)
# gauge.setAntiPeroidicT()  # for fermion smearing

gauge_chroma = io.readQIOGauge("stout.lime")
print(cp.linalg.norm(gauge.data - gauge_chroma.data))
