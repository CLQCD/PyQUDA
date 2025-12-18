from typing import List, NamedTuple, Optional

import numpy
from numpy.typing import NDArray

from ..field import LatticeInfo, LatticeMom
from ..pyquda import computeGaugeLoopTraceQuda, computeGaugeForceQuda
from ..dirac import GaugeDirac

nullptr = numpy.empty((0, 0), "<c16")

from .abstract import LoopParam, Action


class PathParam(NamedTuple):
    input_path_buf: NDArray[numpy.int32]
    path_length: NDArray[numpy.int32]
    loop_coeff: NDArray[numpy.float64]
    num_paths: int
    max_length: int


def actionPath(input_path: List[List[int]], input_coeff: List[float]):
    num_paths = len(input_path)
    path_length = numpy.zeros((num_paths), "<i4")
    loop_coeff = numpy.array(input_coeff, "<f8")
    for i in range(num_paths):
        path_length[i] = len(input_path[i])
    max_length = int(numpy.max(path_length))
    input_path_buf = numpy.full((num_paths, max_length), -1, "<i4")
    for i in range(num_paths):
        dx = [0, 0, 0, 0]
        for j, d in enumerate(input_path[i]):
            if 0 <= d < 4:
                dx[d] += 1
                input_path_buf[i, j] = d
            elif 4 <= d < 8:
                dx[d - 4] -= 1
                input_path_buf[i, j] = 7 - (d - 4)
            else:
                raise ValueError(f"path should be list of int from 0 to 7, but get {input_path[i]}")
        if dx != [0, 0, 0, 0]:
            raise ValueError(f"path {input_path[i]} is not a loop")
    return PathParam(input_path_buf, path_length, loop_coeff, num_paths, max_length)


def forcePath(action_path: PathParam):
    action_input_path_buf = action_path.input_path_buf
    action_path_length = action_path.path_length
    action_loop_coeff = action_path.loop_coeff
    action_num_paths = action_path.num_paths
    force_input_path = [[], [], [], []]
    loop_coeff = [[], [], [], []]
    path_length = [[], [], [], []]
    for i in range(action_num_paths):
        length = int(action_path_length[i])
        coeff = float(action_loop_coeff[i])
        loop_fwd = numpy.array(action_input_path_buf[i, :length])
        loop_bwd = numpy.flip(7 - loop_fwd)
        for j in range(length):
            if loop_fwd[j] < 4:
                k = loop_fwd[j]
                force_input_path[k].append(numpy.roll(loop_fwd, -j)[1:])
                path_length[k].append(length - 1)
                loop_coeff[k].append(-coeff)
            else:
                k = loop_bwd[-(j + 1)]
                force_input_path[k].append(numpy.roll(loop_bwd, j + 1)[1:])
                path_length[k].append(length - 1)
                loop_coeff[k].append(-coeff)
            # if loop_bwd[j] < 4:
            #     k = loop_bwd[j]
            #     force_input_path[k].append(numpy.roll(loop_bwd, -j)[1:])
            #     path_length[k].append(length - 1)
            #     loop_coeff[k].append(-coeff)
    assert path_length[0] == path_length[1] == path_length[2] == path_length[3], "path is not symmetric"
    path_length = numpy.array(path_length[0], "<i4")
    assert loop_coeff[0] == loop_coeff[1] == loop_coeff[2] == loop_coeff[3], "path is not symmetric"
    loop_coeff = numpy.array(loop_coeff[0], "<f8")
    num_paths = len(path_length)
    max_length = int(numpy.max(path_length))
    input_path_buf = numpy.full((4, num_paths, max_length), -1, "<i4")
    for d in range(4):
        for i in range(num_paths):
            input_path_buf[d, i, : path_length[i]] = force_input_path[d][i]
    return PathParam(input_path_buf, path_length, loop_coeff, num_paths, max_length)


class GaugeAction(Action):
    dirac: GaugeDirac

    def __init__(self, latt_info: LatticeInfo, loop_param: LoopParam, beta: float):
        super().__init__(latt_info, GaugeDirac(latt_info))

        # S=\frac{\beta}{N_c}\sum_{i}c_i\mathrm{ReTr}(I-W_i)
        self.action_path = actionPath(loop_param.path, [-beta / latt_info.Nc * coeff for coeff in loop_param.coeff])
        self.force_path = forcePath(self.action_path)

    def action(self) -> float:
        traces = numpy.zeros(self.action_path.num_paths, "<c16")
        computeGaugeLoopTraceQuda(
            traces,
            self.action_path.input_path_buf,
            self.action_path.path_length,
            self.action_path.loop_coeff,
            self.action_path.num_paths,
            self.action_path.max_length,
            1,
        )
        return traces.sum().real.item()

    def force(self, dt: float, mom: Optional[LatticeMom] = None):
        if mom is not None:
            self.gauge_param.use_resident_mom = 0
            self.gauge_param.make_resident_gauge = 0
            self.gauge_param.return_result_mom = 1
        computeGaugeForceQuda(
            nullptr if mom is None else mom.data_ptrs,
            nullptr,
            self.force_path.input_path_buf,
            self.force_path.path_length,
            self.force_path.loop_coeff,
            self.force_path.num_paths,
            self.force_path.max_length,
            dt,
            self.gauge_param,
        )
        if mom is not None:
            self.gauge_param.use_resident_mom = 1
            self.gauge_param.make_resident_gauge = 1
            self.gauge_param.return_result_mom = 0
