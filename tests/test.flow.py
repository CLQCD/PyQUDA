from check_pyquda import weak_field

from pyquda_utils import core, io

core.init(resource_path=".cache")

gauge = io.readQIOGauge(weak_field)

gauge_wflow = gauge.copy()
energy = gauge_wflow.flowWilson(100, 1.0)

gauge_chroma = io.readQIOGauge("wflow.lime")
print((gauge_wflow - gauge_chroma).norm2() ** 0.5)
# print(energy)
