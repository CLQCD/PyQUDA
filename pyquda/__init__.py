from . import pyquda as quda
from .mpi import init
from .field import LatticeGauge, LatticeFermion, LatticePropagator
import numpy as np

nullptr = quda.Pointers("void", 0)


def loadGauge(gauge: LatticeGauge, param: quda.QudaGaugeParam):
    use_resident_gauge = param.use_resident_gauge
    param.use_resident_gauge = 0
    quda.loadGaugeQuda(gauge.data_ptr, param)
    param.use_resident_gauge = use_resident_gauge


def saveGauge(gauge: LatticeGauge, param: quda.QudaGaugeParam):
    quda.saveGaugeQuda(gauge.data_ptr, param)


def momResident(mom: LatticeGauge, param: quda.QudaGaugeParam):
    make_resident_mom = param.make_resident_mom
    return_result_mom = param.return_result_mom
    param.make_resident_mom = 1
    param.return_result_mom = 0
    quda.momResidentQuda(mom.data_ptr, param)
    param.make_resident_mom = make_resident_mom
    param.return_result_mom = return_result_mom


def gaussMom(seed: int):
    quda.gaussMomQuda(seed, 1.0)


def computeCloverForce(dt, x: LatticeFermion, kappa2, ck, multiplicity, gauge_param, inv_param):
    quda.freeCloverQuda()
    quda.loadCloverQuda(nullptr, nullptr, inv_param)
    quda.invertQuda(x.even_ptr, x.odd_ptr, inv_param)
    quda.computeCloverForceQuda(
        nullptr, dt, quda.ndarrayDataPointer(x.even.reshape(1, -1), True), nullptr,
        quda.ndarrayDataPointer(np.array([1.0])), kappa2, ck, 1, multiplicity, nullptr, gauge_param, inv_param
    )


def computeGaugeForce(dt, force, lengths, coeffs, num_paths, max_length, param: quda.QudaGaugeParam):
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


def updateGaugeField(dt, param: quda.QudaGaugeParam):
    quda.updateGaugeFieldQuda(nullptr, nullptr, dt, False, False, param)


def momAction(param: quda.QudaGaugeParam):
    return quda.momActionQuda(nullptr, param)


def projectSU3(tol, param: quda.QudaGaugeParam):
    quda.projectSU3Quda(nullptr, tol, param)


def plaq():
    from .field import Nc
    ret = [0., 0., 0.]
    quda.plaqQuda(ret)
    return ret[0] * Nc
