import os
import numpy as np
from pyquda import core, gauge_utils, LatticePropagator

os.environ["QUDA_RESOURCE_PATH"] = ".cache"

Nc, Ns, Nd = 3, 4, 4

Lx, Ly, Lz, Lt = 16, 16, 16, 128
xi_0, nu = 4.8965, 0.86679
mass = 0.09253
coeff_r, coeff_t = 2.32582045, 0.8549165664

latt_size = [Lx, Ly, Lz, Lt]
Vol = Lx * Ly * Lz * Lt

gauge = gauge_utils.readIldg("/hpcfs/lqcd/qcd/gongming/productions/confs/light.20200720.b20.16_128/s1.0_cfg_100.lime")
assert gauge.latt_size == latt_size

inverter = core.QudaInverter(latt_size, mass, 1e-9, 1000, xi_0, nu, coeff_t, coeff_r)

inverter.initQuda(0)

inverter.loadGauge(gauge)

propagator = LatticePropagator(latt_size)
data = propagator.data.reshape(Vol, Ns, Ns, Nc, Nc)
for spin in range(Ns):
    for color in range(Nc):
        b = core.source(latt_size, "wall", 0, spin, color)
        x = inverter.invert(b)
        data[:, spin, :, color, :] = x.data.reshape(Vol, Ns, Nc)

inverter.endQuda()

propagator_chroma = np.fromfile("wl_prop_1", ">c16", offset=8).reshape(Vol, Ns, Ns, Nc, Nc)
# print(propagator[0])
# print(propagator_chroma[0])
print(np.linalg.norm(propagator.data.reshape(Vol, Ns, Ns, Nc, Nc) - propagator_chroma))
