from os import environ
from typing import Sequence

import numpy
from mpi4py import MPI

_RANK = MPI.COMM_WORLD.Get_rank()
_DEVICE: int = -1


def initDevice(device: int = -1):
    from platform import node as gethostname
    import cupy

    rank = _RANK

    # quda/include/communicator_quda.h
    # determine which GPU this rank will use
    hostname = gethostname()
    hostname_recv_buf = MPI.COMM_WORLD.allgather(hostname)

    if device < 0:
        device_count = cupy.cuda.runtime.getDeviceCount()
        if device_count == 0:
            raise RuntimeError("No devices found")

        # We initialize gpuid if it's still negative.
        device = 0
        for i in range(rank):
            if hostname == hostname_recv_buf[i]:
                device += 1

        if device >= device_count:
            if "QUDA_ENABLE_MPS" in environ and environ["QUDA_ENABLE_MPS"] == "1":
                device %= device_count
                print(f"MPS enabled, rank={rank} -> gpu={device}")
            else:
                raise RuntimeError(f"Too few GPUs available on {hostname}")

    cupy.cuda.Device(device).use()
    print(f"Rank {rank} uses GPU {device}")

    global _DEVICE
    _DEVICE = device


def getSublatticeSize(latt_size: Sequence[int], grid_size: Sequence[int]):
    assert len(latt_size) == len(grid_size)
    for GL, G in zip(latt_size, grid_size):
        assert GL % G == 0
    return [GL // G for GL, G in zip(latt_size, grid_size)]


def getGridCoord(grid_size: Sequence[int]):
    rank = _RANK
    grid_coord = []
    for G in grid_size[::-1]:
        grid_coord.append(rank % G)
        rank //= G
    return grid_coord[::-1]


def getShiftedRank(grid_coord: Sequence[int], grid_size: Sequence[int], delta: Sequence[int]):
    Nd = len(grid_size)
    grid_coord = [(g + d) % G for g, G, d in zip(grid_coord, grid_size, delta)]
    rank = grid_coord[0]
    for mu in range(1, Nd):
        rank = rank * grid_size[mu] + grid_coord[mu]
    return rank


def gaugeSendRecv(extended, gauge, dest, source):
    rank = _RANK
    if rank == dest and rank == source:
        extended[:] = gauge
    else:
        buf = gauge.copy()
        MPI.COMM_WORLD.Sendrecv_replace(buf, dest=dest, source=source)
        extended[:] = buf


class LatticeLink:
    def __init__(
        self,
        latt_size: Sequence[int],
        grid_size: Sequence[int],
        color: int,
        value: numpy.ndarray = None,
        mu: int = None,
    ):
        assert len(latt_size) == len(grid_size)
        self.Nd = len(latt_size)
        self.Nc = color
        self.latt_size = tuple(latt_size)
        self.grid_size = tuple(grid_size)
        self.grid_coord = getGridCoord(grid_size)
        self.sublatt_size = getSublatticeSize(latt_size, grid_size)
        if value is None:
            self.data = numpy.empty((*self.sublatt_size[::-1], self.Nc, self.Nc), numpy.complex128)
            self.data[...] = numpy.identity(self.Nc, dtype=numpy.complex128)
        else:
            self.data = value.reshape(*self.sublatt_size[::-1], self.Nc, self.Nc)
        self.mu = mu

    def __getitem__(self, key):
        return self.data[key]

    def __matmul__(self, other: "LatticeLink"):
        return self.data @ other.data

    @property
    def backend(self):
        if type(self.data).__module__ == "numpy":
            return numpy
        elif type(self.data).__module__ == "cupy":
            import cupy

            return cupy
        else:
            raise RuntimeError(f"Unknown array type {type(self.data)}")

    def shift(self, mu: int, dagger: bool = False):
        assert 0 <= mu < 2 * self.Nd
        backend = self.backend
        Nd = self.Nd
        dir = 1 - 2 * (mu // self.Nd)
        mu = mu % self.Nd
        left_slice = [slice(None, None) for nu in range(self.Nd)]
        right_slice = [slice(None, None) for nu in range(self.Nd)]
        result = backend.empty_like(self.data)
        right = self.data
        rank = _RANK
        dest = getShiftedRank(self.grid_coord, self.grid_size, [0 if nu != mu else -dir for nu in range(Nd)])
        source = getShiftedRank(self.grid_coord, self.grid_size, [0 if nu != mu else dir for nu in range(Nd)])

        left_slice[mu] = slice(-1, None) if dir == 1 else slice(None, 1)
        right_slice[mu] = slice(None, 1) if dir == 1 else slice(-1, None)

        # gaugeSendRecv(shifted[*shift_slice[::-1]], matrix[*matrix_slice[::-1]], dest, source)
        sendbuf = right[*right_slice[::-1]] if not dagger else right[*right_slice[::-1]].swapaxes(-2, -1).conjugate()
        if rank == source and rank == dest:
            pass
        else:
            sendbuf = backend.ascontiguousarray(sendbuf)
            request = MPI.COMM_WORLD.Isend(sendbuf, dest)

        left_slice[mu] = slice(None, -1) if dir == 1 else slice(1, None)
        right_slice[mu] = slice(1, None) if dir == 1 else slice(None, -1)
        result[*left_slice[::-1]] = (
            right[*right_slice[::-1]] if not dagger else right[*right_slice[::-1]].swapaxes(-2, -1).conjugate()
        )
        left_slice[mu] = slice(-1, None) if dir == 1 else slice(None, 1)
        right_slice[mu] = slice(None, 1) if dir == 1 else slice(-1, None)

        if rank == source and rank == dest:
            recvbuf = sendbuf
        else:
            recvbuf = backend.empty_like(sendbuf)
            MPI.COMM_WORLD.Recv(recvbuf, source)
            request.Wait()
        result[*left_slice[::-1]] = recvbuf

        return LatticeLink(self.latt_size, self.grid_size, self.Nc, result, self.mu)

    def matmul_shift(self, right: "LatticeLink"):
        assert self.mu is not None, "Ambiguous dimension and direction"
        backend = self.backend
        Nd = self.Nd
        dir = 1 - 2 * (self.mu // self.Nd)
        mu = self.mu % self.Nd
        left_slice = [slice(None, None) for nu in range(self.Nd)]
        right_slice = [slice(None, None) for nu in range(self.Nd)]
        result = backend.empty_like(self.data)
        left = self.data
        right = right.data
        rank = _RANK
        dest = getShiftedRank(self.grid_coord, self.grid_size, [0 if nu != mu else -dir for nu in range(Nd)])
        source = getShiftedRank(self.grid_coord, self.grid_size, [0 if nu != mu else dir for nu in range(Nd)])

        left_slice[mu] = slice(-1, None) if dir == 1 else slice(None, 1)
        right_slice[mu] = slice(None, 1) if dir == 1 else slice(-1, None)

        sendbuf = right[*right_slice[::-1]]
        if rank == source and rank == dest:
            pass
        else:
            sendbuf = backend.ascontiguousarray(sendbuf)
            request = MPI.COMM_WORLD.Isend(sendbuf, dest)

        left_slice[mu] = slice(None, -1) if dir == 1 else slice(1, None)
        right_slice[mu] = slice(1, None) if dir == 1 else slice(None, -1)
        result[*left_slice[::-1]] = left[*left_slice[::-1]] @ right[*right_slice[::-1]]
        left_slice[mu] = slice(-1, None) if dir == 1 else slice(None, 1)
        right_slice[mu] = slice(None, 1) if dir == 1 else slice(-1, None)

        if rank == source and rank == dest:
            recvbuf = sendbuf
        else:
            recvbuf = backend.empty_like(sendbuf)
            MPI.COMM_WORLD.Recv(recvbuf, source)
            request.Wait()
        result[*left_slice[::-1]] = left[*left_slice[::-1]] @ recvbuf

        return LatticeLink(self.latt_size, self.grid_size, self.Nc, result)

    def dagger(self):
        return LatticeLink(
            self.latt_size,
            self.grid_size,
            self.Nc,
            self.data.swapaxes(-2, -1).conjugate(),
        )

    def toDevice(self):
        import cupy

        if _DEVICE < 0:
            initDevice()
        self.data = cupy.asarray(self.data)

    def toHost(self):
        self.data = self.data.get()


class LatticeGauge:
    def __init__(
        self,
        latt_size: Sequence[int],
        grid_size: Sequence[int],
        color: int,
        border: int = 0,
        value: numpy.ndarray = None,
        extended: numpy.ndarray = None,
    ):
        assert len(latt_size) == len(grid_size)
        self.Nd = len(latt_size)
        self.Nc = color
        self.latt_size = tuple(latt_size)
        self.grid_size = tuple(grid_size)
        self.grid_coord = getGridCoord(grid_size)
        self.sublatt_size = getSublatticeSize(latt_size, grid_size)
        shape = (self.Nd, *self.sublatt_size[::-1], self.Nc, self.Nc)
        if value is None:
            self.data = numpy.empty(shape, numpy.complex128)
            self.data[...] = numpy.identity(self.Nc, dtype=numpy.complex128)
        else:
            self.data = value.reshape(shape)
        self.extend(border, extended)

    def __getitem__(self, mu):
        assert 0 <= mu < 2 * self.Nd
        gauge_mu = LatticeLink(self.latt_size, self.grid_size, self.Nc, self.data[mu % self.Nd], mu)
        return gauge_mu if mu < self.Nd else gauge_mu.shift(mu, True)

    @property
    def backend(self):
        if type(self.data).__module__ == "numpy":
            return numpy
        elif type(self.data).__module__ == "cupy":
            import cupy

            return cupy
        else:
            raise RuntimeError(f"Unknown array type {type(self.data)}")

    def extend(self, border: int, extended: numpy.ndarray = None):
        if border <= 0:
            self.Lb = 0
            self.extlatt_size = self.sublatt_size
            self.extended = None
        else:
            self.Lb = border
            self.extlatt_size = [L + 2 * border for L in self.sublatt_size]
            shape = (self.Nd, *self.extlatt_size[::-1], self.Nc, self.Nc)
            if extended is None:
                self.extended = self.backend.empty(shape, self.data.dtype)
            else:
                self.extended = extended.reshape(shape)
            self.exchange()

    def exchange(self):
        assert self.extended is not None
        Nd = self.Nd
        Lb = self.Lb
        extended_slice = [slice(Lb, -Lb) for mu in range(Nd)]
        gauge_slice = [slice(None, None) for mu in range(Nd)]
        stride = [3 ** (Nd - 1 - mu) for mu in range(Nd)]
        for tag in range(3**Nd):
            delta = [(tag // stride[mu] % 3 - 1) for mu in range(Nd)]
            for mu in range(Nd):
                if delta[mu] == -1:
                    extended_slice[mu] = slice(-Lb, None)
                    gauge_slice[mu] = slice(None, Lb)
                elif delta[mu] == 1:
                    extended_slice[mu] = slice(None, Lb)
                    gauge_slice[mu] = slice(-Lb, None)
            gaugeSendRecv(
                self.extended[:, *extended_slice[::-1]],
                self.data[:, *gauge_slice[::-1]],
                getShiftedRank(self.grid_coord, self.grid_size, delta),
                getShiftedRank(self.grid_coord, self.grid_size, [-d for d in delta]),
            )
            for mu in range(Nd):
                if delta[mu] != 0:
                    extended_slice[mu] = slice(Lb, -Lb)
                    gauge_slice[mu] = slice(None, None)

    def shift(self, delta: Sequence[int]):
        assert numpy.abs(delta).max() <= self.Lb
        Lb = self.Lb
        extended_slice = [slice(Lb + d, None if Lb == d else -(Lb - d)) for d in delta[::-1]]
        return LatticeGauge(self.latt_size, self.grid_size, self.Nc, 0, self.extended[:, *extended_slice], None)

    def toDevice(self):
        import cupy

        if _DEVICE < 0:
            initDevice()
        self.data = cupy.asarray(self.data)
        if self.extended is not None:
            self.extended = cupy.asarray(self.extended)

    def toHost(self):
        self.data = self.data.get()
        if self.extended is not None:
            self.extended = self.extended.get()


def link(*color_matrices: LatticeLink):
    linked = color_matrices[-1]
    for color_matrix in color_matrices[::-1][1:]:
        linked = color_matrix.matmul_shift(linked)
    return linked
