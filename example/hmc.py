from typing import Union, List

import numpy

from pyquda.pointer import Pointers
from pyquda.pyquda import (
    gaussGaugeQuda,
    gaussMomQuda,
    loadGaugeQuda,
    momActionQuda,
    momResidentQuda,
    plaqQuda,
    saveGaugeQuda,
    setVerbosityQuda,
    updateGaugeFieldQuda,
    computeGaugeLoopTraceQuda,
    computeGaugeForceQuda,
)
from pyquda.enum_quda import QudaTboundary, QudaVerbosity
from pyquda.field import LatticeInfo, LatticeGauge, LatticeMom
from pyquda.dirac.pure_gauge import PureGauge

nullptr = Pointers("void", 0)


def getLoopsCoeffs(loops: List[List[int]], coeffs: List[float]):
    num_paths = len(loops)
    assert num_paths == len(coeffs)
    path_length = numpy.full((num_paths), -1, "<i4")
    loop_coeff = numpy.zeros((num_paths), "<f8")
    for i in range(num_paths):
        path_length[i] = len(loops[i])
        loop_coeff[i] = coeffs[i]
    max_length = int(numpy.max(path_length))
    input_path_buf = numpy.full((num_paths, max_length), -1, "<i4")
    for i in range(num_paths):
        dx = [0, 0, 0, 0]
        for j, d in enumerate(loops[i]):
            if 0 <= d < 4:
                dx[d] += 1
                input_path_buf[i, j] = d
            elif 4 <= d < 8:
                dx[d - 4] -= 1
                input_path_buf[i, j] = 7 - (d - 4)
            else:
                raise ValueError(f"path should be list of int from 0 to 7, but get {loops[i]}")
        if dx != [0, 0, 0, 0]:
            raise ValueError(f"path {loops[i]} is not a loop")
    return input_path_buf, path_length, loop_coeff, num_paths, max_length


class HMC:
    def __init__(self, latt_info: LatticeInfo) -> None:
        assert latt_info.anisotropy == 1.0
        self.latt_info = latt_info
        self._pure_gauge = PureGauge(latt_info)
        self.gauge_param = self._pure_gauge.gauge_param

    def setVerbosity(self, verbosity: QudaVerbosity):
        setVerbosityQuda(verbosity, b"\0")

    def initialize(self, gauge: Union[LatticeGauge, int, None] = None):
        if isinstance(gauge, LatticeGauge):
            self.loadGauge(gauge)
            self.loadMom(gauge)
        else:
            unit = LatticeGauge(self.latt_info)
            self.loadGauge(unit)
            self.loadMom(unit)
            if gauge is not None:
                self.gaussGauge(gauge)

    def actionGauge(self, loops: List[List[int]], coeffs: List[float]) -> float:
        input_path_buf, path_length, loop_coeff, num_paths, max_length = getLoopsCoeffs(loops, coeffs)
        traces = numpy.zeros((num_paths), "<c16")
        computeGaugeLoopTraceQuda(
            traces,
            input_path_buf,
            path_length,
            loop_coeff,
            num_paths,
            max_length,
            1,
        )
        return traces.real.sum()

    def actionMom(self) -> float:
        return momActionQuda(nullptr, self.gauge_param)

    def updateGauge(self, dt: float):
        updateGaugeFieldQuda(nullptr, nullptr, dt, False, True, self.gauge_param)
        loadGaugeQuda(nullptr, self.gauge_param)

    def updateMom(self, loops: List[List[List[int]]], coeffs: List[float], dt: float):
        input_path_buf_, path_length, loop_coeff, num_paths, max_length = getLoopsCoeffs(loops[0], coeffs)
        input_path_buf = numpy.full((4, num_paths, max_length - 1), -1, "<i4")
        for i in range(4):
            if i > 0:
                input_path_buf_, path_length_, loop_coeff_, num_paths_, max_length_ = getLoopsCoeffs(loops[i], coeffs)
                assert (path_length == path_length_).all() and num_paths == num_paths_ and max_length == max_length_
            assert (input_path_buf_[:, 0] == i).all()
            input_path_buf[i] = input_path_buf_[:, 1:]
        computeGaugeForceQuda(
            nullptr,
            nullptr,
            input_path_buf,
            path_length - 1,
            loop_coeff,
            num_paths,
            max_length - 1,
            dt,
            self.gauge_param,
        )

    def loadGauge(self, gauge: LatticeGauge):
        gauge_in = gauge.copy()
        if self.gauge_param.t_boundary == QudaTboundary.QUDA_ANTI_PERIODIC_T:
            gauge_in.setAntiPeriodicT()
        self.gauge_param.use_resident_gauge = 0
        loadGaugeQuda(gauge_in.data_ptrs, self.gauge_param)
        self.gauge_param.use_resident_gauge = 1

    def saveGauge(self, gauge: LatticeGauge):
        saveGaugeQuda(gauge.data_ptrs, self.gauge_param)
        if self.gauge_param.t_boundary == QudaTboundary.QUDA_ANTI_PERIODIC_T:
            gauge.setAntiPeriodicT()

    def gaussGauge(self, seed: int):
        gaussGaugeQuda(seed, 1.0)

    def loadMom(self, mom: LatticeMom):
        momResidentQuda(mom.data_ptrs, self.gauge_param)

    def saveMom(self, mom: LatticeMom):
        self.gauge_param.make_resident_mom = 0
        self.gauge_param.return_result_mom = 1
        momResidentQuda(mom.data_ptrs, self.gauge_param)
        self.gauge_param.make_resident_mom = 1
        self.gauge_param.return_result_mom = 0
        momResidentQuda(mom.data_ptrs, self.gauge_param)  # keep momResident

    def gaussMom(self, seed: int):
        gaussMomQuda(seed, 1.0)

    def reunitGauge(self, tol: float):
        gauge = LatticeGauge(self.latt_info)
        self.saveGauge(gauge)
        gauge.projectSU3(tol)
        self.loadGauge(gauge)

    def plaquette(self):
        return plaqQuda()[0]
