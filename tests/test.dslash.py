import os
import sys

test_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(test_dir, ".."))

from pyquda import init, core, quda
from pyquda.enum_quda import QudaParity
from pyquda.field import LatticeFermion
from pyquda.utils import gauge_utils, source

os.environ["QUDA_RESOURCE_PATH"] = ".cache"
init()

latt_size = [16, 16, 16, 32]

# Use m=-3.5 to make kappa=1
dslash = core.getDslash(latt_size, -3.5, 0, 0, anti_periodic_t=False)
kappa = dslash.invert_param.kappa

seed = 0
U = gauge_utils.gaussGauge(latt_size, seed)
a = source.source(latt_size, "point", [0, 0, 0, 0], 0, 0)
b = LatticeFermion(latt_size)

dslash.loadGauge(U)

# Dslash a = b
quda.dslashQuda(b.even_ptr, a.odd_ptr, dslash.invert_param, QudaParity.QUDA_EVEN_PARITY)
quda.dslashQuda(b.odd_ptr, a.even_ptr, dslash.invert_param, QudaParity.QUDA_ODD_PARITY)

print(b.lexico()[0, 0, 0, 1])
