import os
import cupy as cp

from check_pyquda import test_dir

from pyquda import init
from pyquda.utils import io

init(resource_path=".cache")

gauge = io.readQIOGauge(os.path.join(test_dir, "weak_field.lime"))
gauge.smearSTOUT(1, 0.241, 3)
# gauge.setAntiPeroidicT()  # for anti peroidic t fermion smearing

gauge_chroma = io.readQIOGauge("stout.lime")
print(cp.linalg.norm(gauge.data - gauge_chroma.data))
