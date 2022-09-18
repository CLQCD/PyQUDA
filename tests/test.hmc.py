import os
import sys
import numpy as np
import cupy as cp

test_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(test_dir, ".."))
from pyquda import core, quda, mpi, enum_quda
from pyquda.core import Nc
from pyquda.utils import gauge_utils

os.environ["QUDA_RESOURCE_PATH"] = ".cache"

latt_size = [4, 4, 4, 8]
Lx, Ly, Lz, Lt = latt_size
Vol = Lx * Ly * Lz * Lt

beta = 4.0

nullptr = quda.Pointers("void", 0)

dslash = core.getDslash(latt_size, 0, 0, 0, anti_periodic_t=False)
gauge_param = dslash.gauge_param

gauge = gauge_utils.readIldg(os.path.join(test_dir, "weak_field.lime"))

quda.initQuda(mpi.gpuid)

quda.loadGaugeQuda(gauge.data_ptr, gauge_param)

gauge_param.type = enum_quda.QudaLinkType.QUDA_MOMENTUM_LINKS
gauge_param.make_resident_gauge = 1
gauge_param.make_resident_mom = 1
gauge_param.use_resident_gauge = 1
gauge_param.use_resident_mom = 1
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

# obsParam = quda.QudaGaugeObservableParam()
# obsParam.compute_gauge_loop_trace = enum_quda.QudaBoolean.QUDA_BOOLEAN_TRUE
# obsParam.traces = quda.ndarrayDataPointer(traces)
# obsParam.input_path_buff = input_path_buff_ptr
# obsParam.path_length = quda.ndarrayDataPointer(path_length)
# obsParam.loop_coeff = quda.ndarrayDataPointer(loop_coeff)
# obsParam.num_paths = num_paths
# obsParam.max_length = max_length + 1
# obsParam.factor = 1.0
# quda.gaugeObservablesQuda(obsParam)

for i in range(20):
    quda.gaussMomQuda(i, 1.0)

    kinetic = quda.momActionQuda(nullptr, gauge_param)
    quda.computeGaugeLoopTraceQuda(
        quda.ndarrayDataPointer(traces), input_path_buff_ptr, quda.ndarrayDataPointer(path_length),
        quda.ndarrayDataPointer(loop_coeff), num_paths, max_length + 1, 1
    )
    potential = traces.real.sum()
    energy = kinetic + potential

    t = 1
    steps = 20
    dt = t / steps
    for step in range(steps):
        quda.computeGaugeForceQuda(
            nullptr, nullptr, input_path_buf_ptr, quda.ndarrayDataPointer(path_length - 1),
            quda.ndarrayDataPointer(-loop_coeff), num_paths, max_length, 0.5 * dt, gauge_param
        )
        quda.updateGaugeFieldQuda(nullptr, nullptr, 1.0 * dt, False, False, gauge_param)
        quda.computeGaugeForceQuda(
            nullptr, nullptr, input_path_buf_ptr, quda.ndarrayDataPointer(path_length - 1),
            quda.ndarrayDataPointer(-loop_coeff), num_paths, max_length, 0.5 * dt, gauge_param
        )
        pass

    kinetic1 = quda.momActionQuda(nullptr, gauge_param)
    quda.computeGaugeLoopTraceQuda(
        quda.ndarrayDataPointer(traces), input_path_buff_ptr, quda.ndarrayDataPointer(path_length),
        quda.ndarrayDataPointer(loop_coeff), num_paths, max_length + 1, 1
    )
    potential1 = traces.real.sum()
    energy1 = kinetic1 + potential1

    print(
        f'''
PE_old = {potential}, KE_old = {kinetic}
PE = {potential1}, KE = {kinetic1}
Delta_PE = {potential1 - potential}, Delta_KE = {kinetic1 - kinetic}
Delta_E = {energy1 - energy}
'''
    )

dslash.destroy()
quda.endQuda()
