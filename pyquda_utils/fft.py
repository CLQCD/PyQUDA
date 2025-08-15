from math import prod
from time import perf_counter
from typing import List, Sequence, TypeVar, Union

import numpy
from numpy.typing import NDArray
from mpi4py import MPI
from mpi4py.util import dtlib

from pyquda_comm import getMPIComm, getMPISize, getMPIRank, getGridSize, getGridCoord, getCoordFromRank
from pyquda_comm.array import BackendType, arrayHostCopy, arrayDevice
from pyquda_comm.field import (
    LatticeInfo,
    LatticeComplex,
    LatticeLink,
    LatticeFermion,
    LatticePropagator,
    LatticeStaggeredFermion,
    LatticeStaggeredPropagator,
)

Field = TypeVar(
    "Field",
    LatticeComplex,
    LatticeLink,
    LatticeFermion,
    LatticePropagator,
    LatticeStaggeredFermion,
    LatticeStaggeredPropagator,
)


def arrayFFTN(data, axes: Union[int, Sequence[int]], backend: BackendType) -> NDArray:
    if isinstance(axes, int):
        axis = axes
        if backend == "numpy":
            return numpy.fft.fft(data, axis=axis)
        elif backend == "cupy":
            import cupy

            return cupy.fft.fft(data, axis=axis)
        elif backend == "torch":
            import torch

            return torch.fft.fft(data, dim=axis, norm="backward")
    else:
        if backend == "numpy":
            return numpy.fft.fftn(data, axes=axes)
        elif backend == "cupy":
            import cupy

            return cupy.fft.fftn(data, axes=axes)
        elif backend == "torch":
            import torch

            return torch.fft.fftn(data, dim=axes, norm="backward")


def arrayIFFTN(data, axes: Union[int, Sequence[int]], backend: BackendType) -> NDArray:
    if isinstance(axes, int):
        axis = axes
        if backend == "numpy":
            return numpy.fft.ifft(data, axis=axis)
        elif backend == "cupy":
            import cupy

            return cupy.fft.ifft(data, axis=axis)
        elif backend == "torch":
            import torch

            return torch.fft.ifft(data, dim=axis, norm="backward")
    else:
        if backend == "numpy":
            return numpy.fft.ifftn(data, axes=axes)
        elif backend == "cupy":
            import cupy

            return cupy.fft.ifftn(data, axes=axes)
        elif backend == "torch":
            import torch

            return torch.fft.ifftn(data, dim=axes, norm="backward")


def loadBalanceSubsize(GL: int, size: int, rank: int):
    return GL // size + (1 if rank < GL % size else 0)


def loadBalanceStart(GL: int, size: int, rank: int):
    return GL // size * rank + (rank if rank < GL % size else GL % size)


def transform(latt_info: LatticeInfo, field_shape: List[int], recvdim: int, senddim: int, buf: NDArray):
    latt_size = latt_info.global_size
    size = getMPISize()
    rank = getMPIRank()

    sendtypes = []
    recvtypes = []
    send_sizes = [(loadBalanceSubsize(GL, size, rank) if d == senddim else GL) for d, GL in enumerate(latt_size)]
    recv_sizes = [(loadBalanceSubsize(GL, size, rank) if d == recvdim else GL) for d, GL in enumerate(latt_size)]
    for i in range(size):
        send_subsizes = [(loadBalanceSubsize(s, size, i) if d == recvdim else s) for d, s in enumerate(send_sizes)]
        send_starts = [(loadBalanceStart(s, size, i) if d == recvdim else 0) for d, s in enumerate(send_sizes)]
        sendtypes.append(
            MPI.COMPLEX16.Create_subarray(
                send_sizes[::-1] + field_shape,
                send_subsizes[::-1] + field_shape,
                send_starts[::-1] + [0 for _ in field_shape],
            )
            if prod(send_sizes) > 0
            else MPI.COMPLEX16
        )

        recv_subsizes = [(loadBalanceSubsize(s, size, i) if d == senddim else s) for d, s in enumerate(recv_sizes)]
        recv_starts = [(loadBalanceStart(s, size, i) if d == senddim else 0) for d, s in enumerate(recv_sizes)]
        recvtypes.append(
            MPI.COMPLEX16.Create_subarray(
                recv_sizes[::-1] + field_shape,
                recv_subsizes[::-1] + field_shape,
                recv_starts[::-1] + [0 for _ in field_shape],
            )
            if prod(recv_sizes) > 0
            else MPI.COMPLEX16
        )

    sendbuf = buf.reshape(send_sizes[::-1] + field_shape)
    recvbuf = numpy.empty(recv_sizes[::-1] + field_shape, buf.dtype)
    for i in range(size):
        sendtypes[i].Commit()
        recvtypes[i].Commit()
    getMPIComm().Alltoallw((sendbuf, sendtypes), (recvbuf, recvtypes))
    for i in range(size):
        sendtypes[i].Free()
        recvtypes[i].Free()

    return recvbuf


def redistribute(latt_info: LatticeInfo, field_shape: List[int], dim: int, buf: NDArray):
    size = getMPISize()
    rank = getMPIRank()
    GLt = latt_info.global_size[dim]
    Lt = latt_info.size[dim]

    sendtypes = []
    recvtypes = []
    send_sizes = [L for L in latt_info.size]
    send_subsizes = [L for L in latt_info.size]
    recv_sizes = [GL for GL in latt_info.global_size]
    recv_sizes[dim] = loadBalanceSubsize(GLt, size, rank)
    recv_subsizes = [L for L in latt_info.size]

    start_r = loadBalanceStart(GLt, size, rank)
    subsize_r = loadBalanceSubsize(GLt, size, rank)
    gLt_r = getCoordFromRank(rank)[dim] * Lt
    for i in range(size):
        start_i = loadBalanceStart(GLt, size, i)
        subsize_i = loadBalanceSubsize(GLt, size, i)
        gLt_i = getCoordFromRank(i)[dim] * Lt

        start = max(start_i, gLt_r)
        stop = min(start_i + subsize_i, gLt_r + Lt)
        if start >= stop:
            sendtypes.append(MPI.COMPLEX16)
        else:
            send_subsizes[dim] = stop - start
            send_starts = [0 for _ in latt_info.size]
            send_starts[dim] = start - gLt_r
            sendtypes.append(
                MPI.COMPLEX16.Create_subarray(
                    send_sizes[::-1] + field_shape,
                    send_subsizes[::-1] + field_shape,
                    send_starts[::-1] + [0 for _ in field_shape],
                )
            )

        start = max(start_r, gLt_i)
        stop = min(start_r + subsize_r, gLt_i + Lt)
        if start >= stop:
            recvtypes.append(MPI.COMPLEX16)
        else:
            recv_subsizes[dim] = stop - start
            recv_starts = [g * L for g, L in zip(getCoordFromRank(i), latt_info.size)]
            recv_starts[dim] = start - start_r
            recvtypes.append(
                MPI.COMPLEX16.Create_subarray(
                    recv_sizes[::-1] + field_shape,
                    recv_subsizes[::-1] + field_shape,
                    recv_starts[::-1] + [0 for _ in field_shape],
                )
            )

    sendbuf = buf.reshape(send_sizes[::-1] + field_shape)
    recvbuf = numpy.empty_like(buf, shape=recv_sizes[::-1] + field_shape)
    for i in range(size):
        if sendtypes[i] != MPI.COMPLEX16:
            sendtypes[i].Commit()
        if recvtypes[i] != MPI.COMPLEX16:
            recvtypes[i].Commit()
    sendcounts = [0 if sendtype == MPI.COMPLEX16 else 1 for sendtype in sendtypes]
    senddispls = [0 for _ in sendtypes]
    recvcounts = [0 if recvtype == MPI.COMPLEX16 else 1 for recvtype in recvtypes]
    recvdispls = [0 for _ in recvtypes]
    getMPIComm().Alltoallw((sendbuf, sendcounts, senddispls, sendtypes), (recvbuf, recvcounts, recvdispls, recvtypes))
    for i in range(size):
        if sendtypes[i] != MPI.COMPLEX16:
            sendtypes[i].Free()
        if recvtypes[i] != MPI.COMPLEX16:
            recvtypes[i].Free()

    return recvbuf


def redistribute_reverse(latt_info: LatticeInfo, field_shape: List[int], dim: int, buf: NDArray):
    size = getMPISize()
    rank = getMPIRank()
    GLt = latt_info.global_size[dim]
    Lt = latt_info.size[dim]

    recvtypes = []
    sendtypes = []
    recv_sizes = [L for L in latt_info.size]
    recv_subsizes = [L for L in latt_info.size]
    send_sizes = [GL for GL in latt_info.global_size]
    send_sizes[dim] = loadBalanceSubsize(GLt, size, rank)
    send_subsizes = [L for L in latt_info.size]

    start_r = loadBalanceStart(GLt, size, rank)
    subsize_r = loadBalanceSubsize(GLt, size, rank)
    gLt_r = getCoordFromRank(rank)[dim] * Lt
    for i in range(size):
        start_i = loadBalanceStart(GLt, size, i)
        subsize_i = loadBalanceSubsize(GLt, size, i)
        gLt_i = getCoordFromRank(i)[dim] * Lt

        start = max(start_i, gLt_r)
        stop = min(start_i + subsize_i, gLt_r + Lt)
        if start >= stop:
            recvtypes.append(MPI.COMPLEX16)
        else:
            recv_subsizes[dim] = stop - start
            recv_starts = [0 for _ in latt_info.size]
            recv_starts[dim] = start - gLt_r
            recvtypes.append(
                MPI.COMPLEX16.Create_subarray(
                    recv_sizes[::-1] + field_shape,
                    recv_subsizes[::-1] + field_shape,
                    recv_starts[::-1] + [0 for _ in field_shape],
                )
            )

        start = max(start_r, gLt_i)
        stop = min(start_r + subsize_r, gLt_i + Lt)
        if start >= stop:
            sendtypes.append(MPI.COMPLEX16)
        else:
            send_subsizes[dim] = stop - start
            send_starts = [g * L for g, L in zip(getCoordFromRank(i), latt_info.size)]
            send_starts[dim] = start - start_r
            sendtypes.append(
                MPI.COMPLEX16.Create_subarray(
                    send_sizes[::-1] + field_shape,
                    send_subsizes[::-1] + field_shape,
                    send_starts[::-1] + [0 for _ in field_shape],
                )
            )

    sendbuf = buf.reshape(send_sizes[::-1] + field_shape)
    recvbuf = numpy.empty_like(buf, shape=recv_sizes[::-1] + field_shape)
    for i in range(size):
        if sendtypes[i] != MPI.COMPLEX16:
            sendtypes[i].Commit()
        if recvtypes[i] != MPI.COMPLEX16:
            recvtypes[i].Commit()
    sendcounts = [0 if sendtype == MPI.COMPLEX16 else 1 for sendtype in sendtypes]
    senddispls = [0 for _ in sendtypes]
    recvcounts = [0 if recvtype == MPI.COMPLEX16 else 1 for recvtype in recvtypes]
    recvdispls = [0 for _ in recvtypes]
    getMPIComm().Alltoallw((sendbuf, sendcounts, senddispls, sendtypes), (recvbuf, recvcounts, recvdispls, recvtypes))
    for i in range(size):
        if sendtypes[i] != MPI.COMPLEX16:
            sendtypes[i].Free()
        if recvtypes[i] != MPI.COMPLEX16:
            recvtypes[i].Free()

    return recvbuf


def fft(field: Field, fft3d: bool, backend: BackendType = "numpy") -> Field:
    latt_info = field.latt_info
    field_shape = field.field_shape
    Nd = latt_info.Nd
    buf = field.lexico()
    if fft3d:
        buf = redistribute(latt_info, field_shape, Nd - 1, buf)
        buf = arrayHostCopy(arrayFFTN(arrayDevice(buf, backend), (1, 2, 3), backend), backend)
        buf = redistribute_reverse(latt_info, field_shape, Nd - 1, buf)
    else:
        buf = redistribute(latt_info, field_shape, Nd - 1, buf)
        buf = arrayHostCopy(arrayFFTN(arrayDevice(buf, backend), (1, 2, 3), backend), backend)
        buf = transform(latt_info, field_shape, Nd - 2, Nd - 1, buf)
        buf = arrayHostCopy(arrayFFTN(arrayDevice(buf, backend), 0, backend), backend)
        buf = redistribute_reverse(latt_info, field_shape, Nd - 2, buf)
    return field.__class__(latt_info, arrayDevice(latt_info.evenodd(buf, False), field.location))


def ifft(field: Field, fft3d: bool, backend: BackendType = "numpy") -> Field:
    latt_info = field.latt_info
    field_shape = field.field_shape
    Nd = latt_info.Nd
    buf = field.lexico()
    if fft3d:
        buf = redistribute(latt_info, field_shape, Nd - 1, buf)
        buf = arrayHostCopy(arrayIFFTN(arrayDevice(buf, backend), (1, 2, 3), backend), backend)
        buf = redistribute_reverse(latt_info, field_shape, Nd - 1, buf)
    else:
        buf = redistribute(latt_info, field_shape, Nd - 1, buf)
        buf = arrayHostCopy(arrayIFFTN(arrayDevice(buf, backend), (1, 2, 3), backend), backend)
        buf = transform(latt_info, field_shape, Nd - 2, Nd - 1, buf)
        buf = arrayHostCopy(arrayIFFTN(arrayDevice(buf, backend), 0, backend), backend)
        buf = redistribute_reverse(latt_info, field_shape, Nd - 2, buf)
    return field.__class__(latt_info, arrayDevice(latt_info.evenodd(buf, False), field.location))


def fft4(field: LatticePropagator):
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

    print(recvbuf.shape)
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
    rank = getMPIRank()

    s = perf_counter()
    # for d in range(Nd):
    #     buf_hat = numpy.exp(-2j * numpy.pi * (g * g / G)[d]) * sendbuf
    #     for i in range(1, grid_size[d]):
    #         grid_coord[d] = (grid_coord[d] - i) % grid_size[d]
    #         dest = getRankFromCoord(grid_coord)
    #         grid_coord[d] = (grid_coord[d] + 2 * i) % grid_size[d]
    #         source = getRankFromCoord(grid_coord)
    #         grid_coord[d] = (grid_coord[d] - i) % grid_size[d]
    #         getMPIComm().Sendrecv(sendbuf, dest, i, recvbuf, source, i)
    #         buf_hat += numpy.exp(-2j * numpy.pi * (g * gp[source] / G)[d]) * recvbuf
    #     sendbuf = buf_hat
    buf_hat = numpy.exp(-2j * numpy.pi * numpy.sum(g * g / G)) * sendbuf
    for i in range(1, size):
        dest = (rank - i) % size
        source = (rank + i) % size
        getMPIComm().Sendrecv(sendbuf, dest, i, recvbuf, source, i)
        buf_hat += numpy.exp(-2j * numpy.pi * numpy.sum(g * gp[source] / G)) * recvbuf
    print(f"Sendrecv time: {perf_counter() - s:.6f} seconds")

    return buf_hat
