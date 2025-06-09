from check_pyquda import weak_field, data

from pyquda_utils import core, io

core.init(resource_path=".cache")

gauge = io.readQIOGauge(weak_field)

gauge_wflow = gauge.copy()
energy = gauge_wflow.wilsonFlow(100, 0.01)

fermion = core.MultiLatticeFermion(gauge.latt_info, 1)
fermion.data[0, 0, 0, 0, 0, 0, 0, 0] = 1
fermion = gauge.gradientFlow(fermion, "wilson", 100, 0.01, True)
print(fermion.data[0, 0, 0, 0, 0, 0, 0, 0])
fermion = gauge.adjointGradientFlowHierarchy(fermion, "wilson", 100, 0.01)
print(fermion.data[0, 0, 0, 0, 0, 0, 0, 0])

gauge_chroma = io.readQIOGauge(data("wflow.lime"))
print((gauge_wflow - gauge_chroma).norm2() ** 0.5)
# print(energy)
