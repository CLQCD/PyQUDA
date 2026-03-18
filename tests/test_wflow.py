from check_pyquda import weak_field, data

from pyquda_utils import core, io

core.init(resource_path=".cache/quda")

gauge = io.readQIOGauge(weak_field)

gauge_wflow = gauge.copy()
# energy = gauge_wflow.wilsonFlow(100, 0.01)
energy = gauge_wflow.wilsonFlowChroma(100, 1.0)

fermion = core.MultiLatticeFermion(gauge.latt_info, 1)
fermion.data[0, 0, 0, 0, 0, 0, 0, 0] = 1 + 1j
gauge_wflow2 = gauge.copy()
fermion = gauge_wflow2.gradientFlow(fermion, "wilson", 100, 0.01, True)
print(fermion.data[0, 0, 0, 0, 0, 0, 0, 0])

fermion1 = gauge.adjointGradientFlowSafe(fermion, "wilson", 100, 0.01)
print(fermion1.data[0, 0, 0, 0, 0, 0, 0, 0])
fermion2 = gauge.adjointGradientFlowHierarchy(fermion, "wilson", 100, 0.01)
print(fermion2.data[0, 0, 0, 0, 0, 0, 0, 0])

gauge_chroma = io.readQIOGauge(data("wflow.lime"))
print((gauge_wflow - gauge_chroma).norm2() ** 0.5)
print((gauge_wflow2 - gauge_chroma).norm2() ** 0.5)
