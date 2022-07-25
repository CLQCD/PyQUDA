from typing import List

comm = None
rank: int = 0
size: int = 1
grid: List[int] = [1, 1, 1, 1]
coord: List[int] = [0, 0, 0, 0]


def init(grid_size: List[int] = [1, 1, 1, 1]):
    global comm, rank, size, grid, coord
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

        cp.cuda.Device(rank).use()
        initCommsGridQuda(4, grid)
