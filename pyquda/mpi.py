from os import getenv
from platform import node as gethostname

from typing import List

comm = None
rank: int = 0
size: int = 1
grid: List[int] = [1, 1, 1, 1]
coord: List[int] = [0, 0, 0, 0]
gpuid: int = 0


def init(grid_size: List[int] = [1, 1, 1, 1]):
    global comm, rank, size, grid, coord, gpuid
    if comm is None:
        from mpi4py import MPI
        import cupy as cp
        from .pyquda import initCommsGridQuda

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        Gx, Gy, Gz, Gt = grid_size
        assert len(grid_size) == 4 and Gx * Gy * Gz * Gt == size
        grid = grid_size
        coord = [rank // Gt // Gz // Gy, rank // Gt // Gz % Gy, rank // Gt % Gz, rank % Gt]

        gpuid = 0
        hostname = gethostname()
        hostname_recv_buf = comm.allgather(hostname)
        for i in range(rank):
            if hostname == hostname_recv_buf[i]:
                gpuid += 1

        device_count = cp.cuda.runtime.getDeviceCount()
        if gpuid >= device_count:
            enable_mps_env = getenv("QUDA_ENABLE_MPS")
            if enable_mps_env is not None and enable_mps_env == "1":
                gpuid %= device_count

        cp.cuda.Device(gpuid).use()
        initCommsGridQuda(4, grid)


def gather(data, axes: List[int] = [-1, -1, -1, -1], mode: str = None, root: int = 0):
    import numpy as np
    global comm, rank, size, grid
    dtype = data.dtype
    Lt, Lz, Ly, Lx = [data.shape[axis] if axis != -1 else 1 for axis in axes]
    Gx, Gy, Gz, Gt = grid
    collect = tuple([axis for axis in axes if axis != -1])
    if collect == ():
        collect = (0, -1)
    process = tuple([collect[0] + d for d in range(4) if axes[d] == -1])
    prefix = data.shape[:collect[0]]
    suffix = data.shape[collect[-1] + 1:]
    Nroots = Lx * Ly * Lz * Lt
    Nprefix = int(np.prod(prefix))
    Nsuffix = int(np.prod(suffix))
    sendbuf = data.reshape(Nprefix * Nroots * Nsuffix).get()
    if rank == root:
        recvbuf = np.zeros((size, Nprefix * Nroots * Nsuffix), dtype)
    else:
        recvbuf = None
    if comm is not None:
        comm.Gatherv(sendbuf, recvbuf, root)
    else:
        recvbuf[0] = sendbuf
    if rank == root:
        data = np.zeros((Nprefix, Gt * Lt, Gz * Lz, Gy * Ly, Gx * Lx, Nsuffix), dtype)
        for i in range(size):
            gt = i % Gt
            gz = i // Gt % Gz
            gy = i // Gt // Gz % Gy
            gx = i // Gt // Gz // Gy
            data[:, gt * Lt:(gt + 1) * Lt, gz * Lz:(gz + 1) * Lz, gy * Ly:(gy + 1) * Ly,
                 gx * Lx:(gx + 1) * Lx] = recvbuf[i].reshape(Nprefix, Lt, Lz, Ly, Lx, Nsuffix)
        data = data.reshape(*prefix, Gt * Lt, Gz * Lz, Gy * Ly, Gx * Lx, *suffix)

        mode = "sum" if mode is None else mode
        if mode.lower() == "sum":
            data = data.sum(process)
        elif mode.lower() == "mean":
            data = data.mean(process)
        else:
            raise NotImplementedError(f"{mode} mode in mpi.gather not implemented yet.")
        return data
    else:
        return None
