from check_pyquda import weak_field, data

from pyquda_utils import core, io

core.init(resource_path=".cache")

gauge = io.readQIOGauge(weak_field)
gauge.fixingOVR(4, 1000, 1, 1.0, 2e-15, 1, 1)

land_gauge = io.readQIOGauge(data("coul_cfg.lime"))
print((land_gauge - gauge).norm2() ** 0.5)
