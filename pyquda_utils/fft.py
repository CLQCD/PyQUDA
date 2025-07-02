from time import perf_counter

import numpy
from numpy.typing import NDArray
from mpi4py.util import dtlib

from pyquda_comm import getMPIComm, getMPISize, getGridSize, getGridCoord, getCoordFromRank, getRankFromCoord


def fft4(field):
    buf: NDArray = field.lexico()

    size = getMPISize()
    grid_size = getGridSize()
    grid_coord = getGridCoord()
    Nd = len(grid_size)
    sublatt_size = list(buf.shape[:Nd][::-1])
    field_shape = list(buf.shape[Nd:])

    G = numpy.array(grid_size, dtype="<i4")
    g = numpy.array(grid_coord, dtype="<i4")
    gp = numpy.array([getCoordFromRank(i) for i in range(size)], dtype="<i4")
    L = numpy.array(sublatt_size, dtype="<i4")
    F = numpy.array(field_shape, dtype="<i4")

    send_offsets = (g * L + gp) % G
    send_subsizes = (L - (g * L + gp) % G - 1) // G + 1
    send_starts_cumsum = numpy.cumsum(numpy.insert(send_subsizes, 0, 0, axis=0), axis=0)
    send_starts = numpy.empty_like(send_subsizes)
    recv_subsizes = (L - (gp * L + g) % G - 1) // G + 1
    recv_starts_cumsum = numpy.cumsum(numpy.insert(recv_subsizes, 0, 0, axis=0), axis=0)
    recv_starts = numpy.empty_like(recv_subsizes)
    for i in range(size):
        send_starts[i] = numpy.array([send_starts_cumsum.T[ic, c] for ic, c in enumerate(gp[i])], "<i4")
        recv_starts[i] = numpy.array([recv_starts_cumsum.T[ic, c] for ic, c in enumerate(gp[i])], "<i4")

    sendtypes = [
        dtlib.from_numpy_dtype("<c16").Create_subarray(
            L.tolist() + F.tolist(),
            send_subsizes[i].tolist() + F.tolist(),
            send_starts[i].tolist() + numpy.zeros_like(F).tolist(),
        )
        for i in range(size)
    ]
    recvtypes = [
        dtlib.from_numpy_dtype("<c16").Create_subarray(
            L.tolist() + F.tolist(),
            recv_subsizes[i].tolist() + F.tolist(),
            recv_starts[i].tolist() + numpy.zeros_like(F).tolist(),
        )
        for i in range(size)
    ]

    s = perf_counter()
    sendbuf = numpy.empty_like(buf)
    recvbuf = numpy.empty_like(buf)
    for i in range(size):
        sendtypes[i].Commit()
        recvtypes[i].Commit()
        left_slices = tuple([slice(send_starts[i, d], send_starts[i, d] + send_subsizes[i, d]) for d in range(Nd)])
        right_slices = tuple([slice(send_offsets[i, d], None, G[d]) for d in range(Nd)])
        sendbuf[left_slices[::-1]] = buf[right_slices[::-1]]
    getMPIComm().Alltoallw((sendbuf, sendtypes), (recvbuf, recvtypes))
    for i in range(size):
        sendtypes[i].Free()
        recvtypes[i].Free()
    print(f"Alltoallw time: {perf_counter() - s:.6f} seconds")

    from pyfftw.interfaces import numpy_fft

    s = perf_counter()
    sendbuf[:] = numpy_fft.fftn(recvbuf, axes=list(range(0, Nd)))
    print(f"FFTW time: {perf_counter() - s:.6f} seconds")

    s = perf_counter()
    k = numpy.indices(L[::-1].tolist(), dtype="<i4")
    gk = g[0] * k[Nd - 1 - 0] / G[0] / L[0]
    for d in range(1, Nd):
        gk += g[d] * k[Nd - 1 - d] / G[d] / L[d]
    sendbuf *= numpy.exp(-2j * numpy.pi * gk).reshape(*L[::-1].tolist(), *numpy.ones_like(F).tolist())
    print(f"Phase time: {perf_counter() - s:.6f} seconds")

    s = perf_counter()
    for d in range(Nd):
        buf_hat = numpy.exp(-2j * numpy.pi * (g * g / G)[d]) * sendbuf
        for i in range(1, grid_size[d]):
            grid_coord[d] = (grid_coord[d] - i) % grid_size[d]
            dest = getRankFromCoord(grid_coord)
            grid_coord[d] = (grid_coord[d] + 2 * i) % grid_size[d]
            source = getRankFromCoord(grid_coord)
            grid_coord[d] = (grid_coord[d] - i) % grid_size[d]
            getMPIComm().Sendrecv(sendbuf, dest, i, recvbuf, source, i)
            buf_hat += numpy.exp(-2j * numpy.pi * (g * gp[source] / G)[d]) * recvbuf
        sendbuf = buf_hat
    # buf_hat = numpy.exp(-2j * numpy.pi * numpy.sum(g * g / G)) * sendbuf
    # for i in range(1, size):
    #     dest = (rank - i) % size
    #     source = (rank + i) % size
    #     getMPIComm().Sendrecv(sendbuf, dest, i, recvbuf, source, i)
    #     buf_hat += numpy.exp(-2j * numpy.pi * numpy.sum(g * gp[source] / G)) * recvbuf
    print(f"Sendrecv time: {perf_counter() - s:.6f} seconds")

    return buf_hat
