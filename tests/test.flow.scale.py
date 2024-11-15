from check_pyquda import weak_field

from pyquda_utils import core, io

core.init(resource_path=".cache")

gauge = io.readQIOGauge(weak_field)
gauge_wflow = gauge.copy()
t0, w0 = gauge_wflow.wilsonFlowScale(100, 0.01)

print(t0, w0)
