import cupy as cp

from check_pyquda import weak_field

from pyquda import core, init
from pyquda.utils import io

init([1, 1, 1, 1], [4, 4, 4, 8], 1, 1.0, resource_path=".cache")

mass = 0.0102

dslash = core.getDefaultStaggeredDirac(mass, 1e-12, 1000, 1.0, 0.0)
gauge = io.readQIOGauge(weak_field)

dslash.loadGauge(gauge)

# Lx, Ly, Lz, Lt = latt_info.size
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

twopt = cp.einsum("wtzyxab,wtzyxab->t", propagator.data.conj(), propagator.data)
twopt_chroma = cp.einsum("wtzyxab,wtzyxab->t", propagator_chroma.data.conj(), propagator_chroma.data)
print(cp.linalg.norm(twopt - twopt_chroma))
