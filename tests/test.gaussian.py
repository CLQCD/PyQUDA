import cupy as cp

from check_pyquda import weak_field

from pyquda_utils import core, source, io

core.init([1, 1, 1, 2], resource_path=".cache")
latt_info = core.LatticeInfo([4, 4, 4, 8])

rho = 2.0
n_steps = 5

gauge = io.readQIOGauge(weak_field)
# point_source = source.source12(latt_info, "point", [0, 0, 0, 0])
point_source = source.propagator(latt_info, "point", [0, 0, 0, 0])
# shell_source = source.gaussian12(point_source, gauge, rho, n_steps)
shell_source = source.gaussianSmear(point_source, gauge, rho, n_steps)

shell_source_chroma = io.readQIOPropagator("pt_prop_4")
shell_source_chroma.toDevice()
print(cp.linalg.norm(shell_source.data - shell_source_chroma.data))
