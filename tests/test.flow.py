import cupy as cp

from check_pyquda import weak_field

from pyquda_utils import core, io

core.init(resource_path=".cache")

gauge = io.readQIOGauge(weak_field)
gauge_wflow = gauge.copy()
energy = gauge_wflow.flowWilson(100, 1.0)

gauge_chroma = io.readQIOGauge("wflow.lime")
print(cp.linalg.norm(gauge_wflow.data - gauge_chroma.data))

print(energy)
