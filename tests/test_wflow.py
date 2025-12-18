from check_pyquda import weak_field, data

from pyquda_utils import core, io

core.init(resource_path=".cache")

gauge = io.readQIOGauge(weak_field)

gauge_wflow = gauge.copy()
energy = gauge_wflow.wilsonFlowChroma(100, 1.0)
print(energy)

gauge_chroma = io.readQIOGauge(data("wflow.lime"))
print((gauge_wflow - gauge_chroma).norm2() ** 0.5)
# print(energy)
