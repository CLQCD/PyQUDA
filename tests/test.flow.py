import cupy as cp

from check_pyquda import weak_field

from pyquda import init
from pyquda.utils import io

init(resource_path=".cache")

gauge = io.readQIOGauge(weak_field)
gauge_wflow = gauge.copy()
gauge_wflow.flowWilson(1, 1 / 100)
for i in range(1, 100):
    gauge_wflow.flowWilson(1, 1 / 100, i / 100, True)

gauge_chroma = io.readQIOGauge("wflow.lime")
print(cp.linalg.norm(gauge_wflow.data - gauge_chroma.data))
