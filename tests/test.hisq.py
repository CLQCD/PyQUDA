import os
import sys
import cupy as cp

test_dir = os.path.dirname(os.path.abspath(__file__))
# sys.path.insert(0, os.path.join(test_dir, ".."))
from pyquda import core, init
from pyquda.dslash import general
from pyquda.utils import io

general.link_recon = 18
general.link_recon_sloppy = 18

os.environ["QUDA_RESOURCE_PATH"] = ".cache"

latt_size = [4, 4, 4, 8]
Lx, Ly, Lz, Lt = latt_size
Vol = Lx * Ly * Lz * Lt
init()

mass = 0.0102

dslash = core.getStaggeredDslash(latt_size, mass, 1e-12, 1000, 1.0, 0.0, False)
gauge = io.readQIOGauge(os.path.join(test_dir, "weak_field.lime"))

dslash.loadGauge(gauge)

# import cupy as cp
# Cx = np.arange(Lx).reshape(1, 1, 1, Lx).repeat(Ly, 2).repeat(Lz, 1).repeat(Lt, 0)
# Cy = np.arange(Ly).reshape(1, 1, Ly, 1).repeat(Lx, 3).repeat(Lz, 1).repeat(Lt, 0)
# Cz = np.arange(Lz).reshape(1, Lz, 1, 1).repeat(Lx, 3).repeat(Ly, 2).repeat(Lt, 0)
# Ct = np.arange(Lt).reshape(Lt, 1, 1, 1).repeat(Lx, 3).repeat(Ly, 2).repeat(Lz, 1)
# Convert from CPS(QUDA, Old) to Chroma
# phase = cp.asarray(core.cb2(np.where((Cx) % 2 == 1, -1, 1), [0, 1, 2, 3]))
# Convert from MILC to Chroma
# phase = cp.asarray(core.cb2(np.where(((Cx + Cy + Cz) % 2 == 1) | (Ct % 2 == 1), -1, 1), [0, 1, 2, 3]))
# # Convert from CPS(QUDA, New) to Chroma
# phase = cp.asarray(core.cb2(np.where((Cx + Cy + Cz + Ct) % 2 == 1, -1, 1), [0, 1, 2, 3]))

propagator = core.invertStaggered(dslash, "point", [0, 0, 0, 0])

dslash.destroy()

propagator_chroma = io.readQIOPropagator("pt_prop_2")
propagator_chroma.toDevice()
print(cp.linalg.norm(propagator.data - propagator_chroma.data))
