import cupy as cp

from check_pyquda import weak_field, data

from pyquda.enum_quda import QudaStaggeredPhase
from pyquda_utils import core, io

core.init([1, 1, 1, 1], [4, 4, 4, 8], resource_path=".cache")

mass = 0.0102

gauge = io.readQIOGauge(weak_field)
latt_info = core.LatticeInfo([4, 4, 4, 8], 1, 1.0)
dirac = core.getHISQ(latt_info, mass, 1e-12, 1000, 0.0)
dirac.gauge_param.staggered_phase_type = QudaStaggeredPhase.QUDA_STAGGERED_PHASE_CHROMA

# Lx, Ly, Lz, Lt = latt_info.size
# Cx = np.arange(Lx).reshape(1, 1, 1, Lx).repeat(Ly, 2).repeat(Lz, 1).repeat(Lt, 0)
# Cy = np.arange(Ly).reshape(1, 1, Ly, 1).repeat(Lx, 3).repeat(Lz, 1).repeat(Lt, 0)
# Cz = np.arange(Lz).reshape(1, Lz, 1, 1).repeat(Lx, 3).repeat(Ly, 2).repeat(Lt, 0)
# Ct = np.arange(Lt).reshape(Lt, 1, 1, 1).repeat(Lx, 3).repeat(Ly, 2).repeat(Lz, 1)
# Convert from CPS(QUDA, Old) to Chroma
# phase = cp.asarray(core.evenodd(np.where((Cx) % 2 == 1, -1, 1), [0, 1, 2, 3]))
# Convert from MILC to Chroma
# phase = cp.asarray(core.evenodd(np.where(((Cx + Cy + Cz) % 2 == 1) | (Ct % 2 == 1), -1, 1), [0, 1, 2, 3]))
# # Convert from CPS(QUDA, New) to Chroma
# phase = cp.asarray(core.evenodd(np.where((Cx + Cy + Cz + Ct) % 2 == 1, -1, 1), [0, 1, 2, 3]))

with dirac.useGauge(gauge):
    propagator = core.invertStaggered(dirac, "point", [0, 0, 0, 0])

propagator_chroma = io.readQIOPropagator(data("pt_prop_2"))
propagator_chroma.toDevice()
print((propagator - propagator_chroma).norm2() ** 0.5)

twopt = cp.einsum("wtzyxab,wtzyxab->t", propagator.data.conj(), propagator.data)
twopt_chroma = cp.einsum("wtzyxab,wtzyxab->t", propagator_chroma.data.conj(), propagator_chroma.data)
print(cp.linalg.norm(twopt - twopt_chroma))
