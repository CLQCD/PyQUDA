import math
from time import perf_counter
from mpi4py import MPI

from check_pyquda import weak_field

from pyquda_io.chroma import readQIOGauge as readChromaQIOGauge
from pyquda_utils.gauge_nd_sun import init, LatticeGauge, link

GPU = False
if GPU:
    import cupy as numpy
else:
    import numpy
MODE = "link_four"

init([1, 2, 1, 2], [4, 4, 4, 8], "cupy" if GPU else "numpy")
s = perf_counter()
print([0.9948041322666998, 0.9947985783413031, 0.9948096861920964])
latt_size, gauge = readChromaQIOGauge(weak_field)
print(f"Load: {perf_counter() - s:.3f} s")
s = perf_counter()
unit = LatticeGauge(latt_size, 3, 0)
gauge = LatticeGauge(latt_size, 3, 1, gauge)
if GPU:
    gauge.toDevice()  # CUDA-aware MPI required
print(f"Prepare: {perf_counter() - s:.3f} s")

s = perf_counter()
plaq = numpy.zeros((6))
if MODE == "gauge_extend":
    plaq[0] = numpy.vdot(
        gauge[0] @ gauge.shift([1, 0, 0, 0])[1],
        gauge[1] @ gauge.shift([0, 1, 0, 0])[0],
    ).real
    plaq[1] = numpy.vdot(
        gauge[0] @ gauge.shift([1, 0, 0, 0])[2],
        gauge[2] @ gauge.shift([0, 0, 1, 0])[0],
    ).real
    plaq[2] = numpy.vdot(
        gauge[1] @ gauge.shift([0, 1, 0, 0])[2],
        gauge[2] @ gauge.shift([0, 0, 1, 0])[1],
    ).real
    plaq[3] = numpy.vdot(
        gauge[0] @ gauge.shift([1, 0, 0, 0])[3],
        gauge[3] @ gauge.shift([0, 0, 0, 1])[0],
    ).real
    plaq[4] = numpy.vdot(
        gauge[1] @ gauge.shift([0, 1, 0, 0])[3],
        gauge[3] @ gauge.shift([0, 0, 0, 1])[1],
    ).real
    plaq[5] = numpy.vdot(
        gauge[2] @ gauge.shift([0, 0, 1, 0])[3],
        gauge[3] @ gauge.shift([0, 0, 0, 1])[2],
    ).real
elif MODE == "link_shift":
    plaq[0] = numpy.vdot(gauge[0] @ gauge[1].shift(0), gauge[1] @ gauge[0].shift(1)).real
    plaq[1] = numpy.vdot(gauge[0] @ gauge[2].shift(0), gauge[2] @ gauge[0].shift(2)).real
    plaq[2] = numpy.vdot(gauge[1] @ gauge[2].shift(1), gauge[2] @ gauge[1].shift(2)).real
    plaq[3] = numpy.vdot(gauge[0] @ gauge[3].shift(0), gauge[3] @ gauge[0].shift(3)).real
    plaq[4] = numpy.vdot(gauge[1] @ gauge[3].shift(1), gauge[3] @ gauge[1].shift(3)).real
    plaq[5] = numpy.vdot(gauge[2] @ gauge[3].shift(2), gauge[3] @ gauge[2].shift(3)).real
elif MODE == "link_two":
    plaq[0] = numpy.vdot(link(gauge[0], gauge[1]).data, link(gauge[1], gauge[0]).data).real
    plaq[1] = numpy.vdot(link(gauge[0], gauge[2]).data, link(gauge[2], gauge[0]).data).real
    plaq[2] = numpy.vdot(link(gauge[1], gauge[2]).data, link(gauge[2], gauge[1]).data).real
    plaq[3] = numpy.vdot(link(gauge[0], gauge[3]).data, link(gauge[3], gauge[0]).data).real
    plaq[4] = numpy.vdot(link(gauge[1], gauge[3]).data, link(gauge[3], gauge[1]).data).real
    plaq[5] = numpy.vdot(link(gauge[2], gauge[3]).data, link(gauge[3], gauge[2]).data).real
elif MODE == "link_four":
    plaq[0] = numpy.einsum("tzyxaa->", link(gauge[0], gauge[1], gauge[4], gauge[5]).data).real
    plaq[1] = numpy.einsum("tzyxaa->", link(gauge[0], gauge[2], gauge[4], gauge[6]).data).real
    plaq[2] = numpy.einsum("tzyxaa->", link(gauge[1], gauge[2], gauge[5], gauge[6]).data).real
    plaq[3] = numpy.einsum("tzyxaa->", link(gauge[0], gauge[3], gauge[4], gauge[7]).data).real
    plaq[4] = numpy.einsum("tzyxaa->", link(gauge[1], gauge[3], gauge[5], gauge[7]).data).real
    plaq[5] = numpy.einsum("tzyxaa->", link(gauge[2], gauge[3], gauge[6], gauge[7]).data).real
plaq = MPI.COMM_WORLD.allreduce(plaq, op=MPI.SUM)
plaq /= math.prod(latt_size) * gauge.Nc
print(plaq.mean(), plaq[:3].mean(), plaq[3:].mean())
print(f"Compute: {perf_counter() - s:.3f} s")
