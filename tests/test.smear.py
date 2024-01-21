import cupy as cp

from check_pyquda import weak_field

from pyquda import init
from pyquda.utils import io

init(resource_path=".cache")

gauge = io.readQIOGauge(weak_field)
gauge.smearSTOUT(1, 0.241, 3)
# gauge.setAntiPeroidicT()  # for anti peroidic t fermion smearing

gauge_chroma = io.readQIOGauge("stout.lime")
print(cp.linalg.norm(gauge.data - gauge_chroma.data))
