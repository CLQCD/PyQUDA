import numpy

from ..pointer import Pointers
from ..pyquda import computeGaugeLoopTraceQuda, computeGaugeForceQuda
from ..field import Nc, LatticeInfo
from ..dirac.pure_gauge import PureGauge

nullptr = Pointers("void", 0)

from . import GaugeAction


def loop_ndarray(path, num_paths, max_length):
    ret = -numpy.ones((num_paths, max_length), "<i4")
    for i in range(num_paths):
        for j in range(len(path[i])):
            ret[i, j] = path[i][j]
    return ret


def path_ndarray(path, num_paths, max_length):
    ret = -numpy.ones((4, num_paths, max_length), "<i4")
    for d in range(4):
        for i in range(num_paths):
            for j in range(len(path[d][i])):
                ret[d, i, j] = path[d][i][j]
    return ret


def path_force(path, coeffs):
    num_paths = len(path)
    lengths = []
    fpath = [[], [], [], []]
    fcoeffs = [[], [], [], []]
    flengths = [[], [], [], []]
    for i in range(num_paths):
        lengths.append(len(path[i]))
        loop = numpy.array(path[i])
        loop_dag = numpy.flip(7 - loop)
        for j in range(lengths[i]):
            if loop[j] < 4:
                fpath[loop[j]].append(numpy.roll(loop, -j)[1:])
                fcoeffs[loop[j]].append(-coeffs[i])
                flengths[loop[j]].append(lengths[i] - 1)
            else:
                fpath[loop_dag[lengths[i] - 1 - j]].append(numpy.roll(loop_dag, j + 1 - lengths[i])[1:])
                fcoeffs[loop_dag[lengths[i] - 1 - j]].append(-coeffs[i])
                flengths[loop_dag[lengths[i] - 1 - j]].append(lengths[i] - 1)
    max_length = max(lengths)
    lengths = numpy.array(lengths, dtype="<i4")
    coeffs = numpy.array(coeffs, "<f8")
    path = loop_ndarray(path, num_paths, max_length)
    assert flengths[0] == flengths[1] == flengths[2] == flengths[3], "path is not symmetric"
    flengths = numpy.array(flengths[0], "<i4")
    assert fcoeffs[0] == fcoeffs[1] == fcoeffs[2] == fcoeffs[3], "path is not symmetric"
    fcoeffs = numpy.array(fcoeffs[0], "<f8")
    num_fpaths = len(flengths)
    max_flength = max_length - 1
    fpath = path_ndarray(fpath, num_fpaths, max_flength)
    return path, num_paths, max_length, lengths, coeffs, fpath, num_fpaths, max_flength, flengths, fcoeffs


class WilsonGauge(GaugeAction):
    def __init__(self, latt_info: LatticeInfo, beta: float, u_0: float):
        super().__init__(latt_info)

        self.pure_gauge = PureGauge(latt_info)
        self.gauge_param = self.pure_gauge.gauge_param

        input_path = [
            [0, 1, 7, 6],
            [0, 2, 7, 5],
            [1, 2, 6, 5],
            [0, 3, 7, 4],
            [1, 3, 6, 4],
            [2, 3, 5, 4],
        ]
        input_coeffs = [
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
        ]
        (
            self.path,
            self.num_paths,
            self.max_length,
            self.lengths,
            self.coeffs,
            self.fpath,
            self.num_fpaths,
            self.max_flength,
            self.flengths,
            self.fcoeffs,
        ) = path_force(input_path, input_coeffs)
        self.coeffs *= beta / u_0**4 / Nc
        self.fcoeffs *= beta / u_0**4 / Nc

    def action(self) -> float:
        traces = numpy.zeros((self.num_paths), "<c16")
        computeGaugeLoopTraceQuda(
            traces,
            self.path,
            self.lengths,
            self.coeffs,
            self.num_paths,
            self.max_length,
            1,
        )
        return traces.real.sum()

    def force(self, dt: float):
        computeGaugeForceQuda(
            nullptr,
            nullptr,
            self.fpath,
            self.flengths,
            self.fcoeffs,
            self.num_fpaths,
            self.max_flength,
            dt,
            self.gauge_param,
        )
