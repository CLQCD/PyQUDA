import os
import sys
import cupy as cp


test_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, os.path.join(test_dir, ".."))
from pyquda import init
from pyquda.utils import io
from pyquda.field import LatticeInfo

os.environ["QUDA_RESOURCE_PATH"] = ".cache"

init()
latt_info = LatticeInfo([4, 4, 4, 8])

gauge = io.readQIOGauge(os.path.join(test_dir, "weak_field.lime"))
gauge_ape = gauge.copy()
gauge_ape.smearAPE(1, 2.5, 4)
gauge_stout = gauge.copy()
gauge_stout.smearSTOUT(1, 0.241, 3)
gauge_hyp = gauge.copy()
gauge_hyp.smearHYP(1, 0.75, 0.6, 0.3, 4)
# gauge.setAntiPeroidicT()  # for fermion smearing

gauge_chroma = io.readQIOGauge("ape.lime")
print(cp.linalg.norm(gauge_ape.data - gauge_chroma.data))

gauge_chroma = io.readQIOGauge("stout.lime")
print(cp.linalg.norm(gauge_stout.data - gauge_chroma.data))

gauge_chroma = io.readQIOGauge("hyp.lime")
print(cp.linalg.norm(gauge_hyp.data - gauge_chroma.data))
