import cupy as cp

from check_pyquda import weak_field

from pyquda import init
from pyquda.utils import io

init(resource_path=".cache")

gauge = io.readQIOGauge(weak_field)
gauge_ape = gauge.copy()
gauge_ape.smearAPE(1, 2.5, 4)
gauge_stout = gauge.copy()
gauge_stout.smearSTOUT(1, 0.241, 3)
gauge_hyp = gauge.copy()
gauge_hyp.smearHYP(1, 0.75, 0.6, 0.3, 4)
# gauge.setAntiPeriodicT()  # for fermion smearing

gauge_chroma = io.readQIOGauge("ape.lime")
print(cp.linalg.norm(gauge_ape.data - gauge_chroma.data))

gauge_chroma = io.readQIOGauge("stout.lime")
print(cp.linalg.norm(gauge_stout.data - gauge_chroma.data))

gauge_chroma = io.readQIOGauge("hyp.lime")
print(cp.linalg.norm(gauge_hyp.data - gauge_chroma.data))
