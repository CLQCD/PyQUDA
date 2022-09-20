from . import pyquda as quda
from .pyquda import QudaGaugeParam
from .core import LatticeGauge, LatticeFermion, LatticePropagator

nullptr = quda.Pointers("void", 0)


def loadGauge(gauge: LatticeGauge, param: QudaGaugeParam):
    use_resident_gauge = param.use_resident_gauge
    param.use_resident_gauge = 0
    quda.loadGaugeQuda(gauge.data_ptr, param)
    param.use_resident_gauge = use_resident_gauge


def saveGauge(gauge: LatticeGauge, param: QudaGaugeParam):
    quda.saveGaugeQuda(gauge.data_ptr, param)


def momResident(mom: LatticeGauge, param: QudaGaugeParam):
    make_resident_mom = param.make_resident_mom
    return_result_mom = param.return_result_mom
    param.make_resident_mom = 1
    param.return_result_mom = 0
    quda.momResidentQuda(mom.data_ptr, param)
    param.make_resident_mom = make_resident_mom
    param.return_result_mom = return_result_mom


def gaussMom(seed: int):
    quda.gaussMomQuda(seed, 1.0)


def computeGaugeForce(dt, force, lengths, coeffs, num_paths, max_length, param: QudaGaugeParam):
    quda.computeGaugeForceQuda(
        nullptr, nullptr, quda.ndarrayDataPointer(force), quda.ndarrayDataPointer(lengths),
        quda.ndarrayDataPointer(coeffs), num_paths, max_length, dt, param
    )


def computeGaugeLoopTrace(dt, path, lengths, coeffs, num_paths, max_length):
    import numpy as np
    traces = np.zeros((num_paths), "<c16")
    quda.computeGaugeLoopTraceQuda(
        quda.ndarrayDataPointer(traces), quda.ndarrayDataPointer(path), quda.ndarrayDataPointer(lengths),
        quda.ndarrayDataPointer(coeffs), num_paths, max_length, dt
    )
    return traces.real.sum()


def updateGaugeField(dt, param: QudaGaugeParam):
    quda.updateGaugeFieldQuda(nullptr, nullptr, dt, False, False, param)


def momAction(param: QudaGaugeParam):
    return quda.momActionQuda(nullptr, param)


def plaq():
    from .core import Nc
    ret = [0., 0., 0.]
    quda.plaqQuda(ret)
    return ret[0] * Nc
