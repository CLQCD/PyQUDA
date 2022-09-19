import os
import sys
import numpy as np
import cupy as cp

test_dir = os.path.dirname(os.path.abspath(__file__))
# sys.path.insert(0, os.path.join(test_dir, ".."))
from pyquda import core, quda, mpi, enum_quda
from pyquda.core import Nc

os.environ["QUDA_RESOURCE_PATH"] = ".cache"

ensembles = {
    "A1a": ([16, 16, 16, 16], 5.789),
    "B0a": ([24, 24, 24, 24], 6),
    "C1d": ([32, 32, 32, 64], 6.136),
    "D1d": ([48, 48, 48, 48], 6.475)
}

tag = "D1d"

latt_size = ensembles[tag][0]
Lx, Ly, Lz, Lt = latt_size
Vol = Lx * Ly * Lz * Lt

beta = ensembles[tag][1]
t = 1
dt = 0.1
steps = round(t / dt)
dt = t / steps

nullptr = quda.Pointers("void", 0)

dslash = core.getDslash(latt_size, 0, 0, 0, anti_periodic_t=False)
gauge_param = dslash.gauge_param

gauge = core.LatticeGauge(latt_size, None, True)
gauge.data[:] = cp.diag([1, 1, 1])

quda.initQuda(mpi.gpuid)

quda.loadGaugeQuda(gauge.data_ptr, gauge_param)

gauge_param.type = enum_quda.QudaLinkType.QUDA_MOMENTUM_LINKS
gauge_param.overwrite_gauge = 0
gauge_param.overwrite_mom = 0
gauge_param.use_resident_gauge = 1
gauge_param.use_resident_mom = 1
gauge_param.make_resident_gauge = 1
gauge_param.make_resident_mom = 1
gauge_param.return_result_gauge = 0
gauge_param.return_result_mom = 0
quda.momResidentQuda(gauge.data_ptr, gauge_param)

num_paths = 6
max_length = 3

input_path_buf = np.array(
    [
        [[1, 7, 6], [6, 7, 1], [2, 7, 5], [5, 7, 2], [3, 7, 4], [4, 7, 3]],
        [[0, 6, 7], [7, 6, 0], [2, 6, 5], [5, 6, 2], [3, 6, 4], [4, 6, 3]],
        [[0, 5, 7], [7, 5, 0], [1, 5, 6], [6, 5, 1], [3, 5, 4], [4, 5, 3]],
        [[0, 4, 7], [7, 4, 0], [1, 4, 6], [6, 4, 1], [2, 4, 5], [5, 4, 2]]
    ],
    dtype="<i4"
)
input_path_buf_ptr = quda.ndarrayDataPointer(input_path_buf)

input_path_buff = np.array(
    [
        [0, 1, 7, 6],
        [0, 2, 7, 5],
        [0, 3, 7, 4],
        [1, 2, 6, 5],
        [1, 3, 6, 4],
        [2, 3, 5, 4],
    ], dtype="<i4"
)
input_path_buff_ptr = quda.ndarrayDataPointer(input_path_buff)

traces = np.zeros((num_paths), "<c16")
path_length = np.zeros((num_paths), "<i4")
path_length[:] = max_length + 1
loop_coeff = np.zeros((num_paths), "<f8")
loop_coeff[:] = -beta / Nc


def computeGaugeForce(dt):
    quda.computeGaugeForceQuda(
        nullptr, nullptr, input_path_buf_ptr, quda.ndarrayDataPointer(path_length - 1),
        quda.ndarrayDataPointer(-loop_coeff), num_paths, max_length, dt, gauge_param
    )


def updateGaugeField(dt):
    quda.updateGaugeFieldQuda(nullptr, nullptr, dt, False, False, gauge_param)


def momAction():
    return quda.momActionQuda(nullptr, gauge_param)


def computeGaugeLoopTrace():
    quda.computeGaugeLoopTraceQuda(
        quda.ndarrayDataPointer(traces), input_path_buff_ptr, quda.ndarrayDataPointer(path_length),
        quda.ndarrayDataPointer(loop_coeff), num_paths, max_length + 1, 1
    )
    return traces.real.sum()


def plaq():
    ret = [0, 0, 0]
    quda.plaqQuda(ret)
    return ret[0] * Nc


rho_ = 0.2539785108410595
theta_ = -0.03230286765269967
vartheta_ = 0.08398315262876693
lambda_ = 0.6822365335719091

# rho_ = 0
# theta_ = 0
# vartheta_ = 0
# lambda_ = 0

warm = True
gauge_param.type = enum_quda.QudaLinkType.QUDA_WILSON_LINKS
for i in range(100):
    quda.gaussMomQuda(i, 1.0)
    quda.saveGaugeQuda(gauge.data_ptr, gauge_param)

    kinetic = momAction()
    potential = computeGaugeLoopTrace()
    energy = kinetic + potential

    for step in range(steps):
        computeGaugeForce(vartheta_ * dt)
        updateGaugeField(rho_ * dt)
        computeGaugeForce(lambda_ * dt)
        updateGaugeField(theta_ * dt)
        computeGaugeForce((1 - 2 * (lambda_ + vartheta_)) * dt / 2)
        updateGaugeField((1 - 2 * (theta_ + rho_)) * dt)
        computeGaugeForce((1 - 2 * (lambda_ + vartheta_)) * dt / 2)
        updateGaugeField(theta_ * dt)
        computeGaugeForce(lambda_ * dt)
        updateGaugeField(rho_ * dt)
        computeGaugeForce(vartheta_ * dt)

    kinetic1 = momAction()
    potential1 = computeGaugeLoopTrace()
    energy1 = kinetic1 + potential1

    accept = np.random.rand() < np.exp(energy - energy1) or warm
    if energy1 < energy:
        warm = False
    if not accept:
        gauge_param.use_resident_gauge = 0
        quda.loadGaugeQuda(gauge.data_ptr, gauge_param)
        gauge_param.use_resident_gauge = 1

    plaquette = plaq()

    print(
        f'''
PE_old = {potential}, KE_old = {kinetic}
PE = {potential1}, KE = {kinetic1}
Delta_PE = {potential1 - potential}, Delta_KE = {kinetic1 - kinetic}
Delta_E = {energy1 - energy}
accept = {accept}
plaquette = {plaquette}
'''
    )

dslash.destroy()
quda.endQuda()
