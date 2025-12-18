from check_pyquda import weak_field, data

from pyquda_utils import core, io

core.init(None, [4, 4, 4, 8], resource_path=".cache")

gauge = io.readQIOGauge(weak_field)
gauge_ape = gauge.copy()
gauge_ape.apeSmearChroma(1, 2.5, 4)
# gauge_ape.apeSmear(1, (4 - 1) / (4 - 1 + 2.5 / 2), 4)
gauge_stout = gauge.copy()
gauge_stout.stoutSmear(1, 0.241, 3)
gauge_hyp = gauge.copy()
gauge_hyp.hypSmear(1, 0.75, 0.6, 0.3, 4)
# gauge.setAntiPeriodicT()  # for fermion smearing

gauge_chroma = io.readQIOGauge(data("ape.lime"))
print((gauge_ape - gauge_chroma).norm2() ** 0.5)

gauge_chroma = io.readQIOGauge(data("stout.lime"))
print((gauge_stout - gauge_chroma).norm2() ** 0.5)

gauge_chroma = io.readQIOGauge(data("hyp.lime"))
print((gauge_hyp - gauge_chroma).norm2() ** 0.5)
