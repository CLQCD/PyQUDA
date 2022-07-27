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
