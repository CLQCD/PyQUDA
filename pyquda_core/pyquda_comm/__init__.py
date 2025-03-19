import logging
from sys import stdout
from typing import List, Literal, Tuple, Union

from mpi4py import MPI


class _MPILogger:
    def __init__(self, root: int = 0) -> None:
        self.root = root
        formatter = logging.Formatter(fmt="{name} {levelname}: {message}", style="{")
        stdout_handler = logging.StreamHandler(stdout)
        stdout_handler.setFormatter(formatter)
        stdout_handler.setLevel(logging.DEBUG)
        stdout_handler.addFilter(lambda record: record.levelno <= logging.INFO)
        stderr_handler = logging.StreamHandler()
        stderr_handler.setFormatter(formatter)
        stderr_handler.setLevel(logging.WARNING)
        self.logger = logging.getLogger("PyQUDA")
        self.logger.level = logging.DEBUG
        self.logger.handlers = [stdout_handler, stderr_handler]

    def debug(self, msg: str):
        if _MPI_RANK == self.root:
            self.logger.debug(msg)

    def info(self, msg: str):
        if _MPI_RANK == self.root:
            self.logger.info(msg)

    def warning(self, msg: str, category: Warning):
        if _MPI_RANK == self.root:
            self.logger.warning(msg, exc_info=category(msg), stack_info=True)

    def error(self, msg: str, category: Exception):
        if _MPI_RANK == self.root:
            self.logger.error(msg, exc_info=category(msg), stack_info=True)

    def critical(self, msg: str, category: Exception):
        if _MPI_RANK == self.root:
            self.logger.critical(msg, exc_info=category(msg), stack_info=True)
        raise category(msg)


_MPI_LOGGER: _MPILogger = _MPILogger()
_MPI_COMM: MPI.Comm = MPI.COMM_WORLD
_MPI_SIZE: int = _MPI_COMM.Get_size()
_MPI_RANK: int = _MPI_COMM.Get_rank()
_GRID_SIZE: Union[List[int], None] = None
_GRID_COORD: Union[List[int], None] = None
_GRID_MAP: Literal["XYZT_FASTEST", "TZYX_FASTEST"] = "XYZT_FASTEST"
"""For MPI, the default node mapping is lexicographical with t varying fastest."""


def getRankFromCoord(grid_coord: List[int], grid_size: List[int] = None) -> int:
    Gx, Gy, Gz, Gt = _GRID_SIZE if grid_size is None else grid_size
    gx, gy, gz, gt = grid_coord
    if _GRID_MAP == "XYZT_FASTEST":
        return ((gx * Gy + gy) * Gz + gz) * Gt + gt
    elif _GRID_MAP == "TZYX_FASTEST":
        return ((gt * Gz + gz) * Gy + gy) * Gx + gx
    else:
        _MPI_LOGGER.critical(f"Unsupported grid mapping {_GRID_MAP}", ValueError)


def getCoordFromRank(mpi_rank: int, grid_size: List[int] = None) -> List[int]:
    Gx, Gy, Gz, Gt = _GRID_SIZE if grid_size is None else grid_size
    if _GRID_MAP == "XYZT_FASTEST":
        return [mpi_rank // Gt // Gz // Gy, mpi_rank // Gt // Gz % Gy, mpi_rank // Gt % Gz, mpi_rank % Gt]
    elif _GRID_MAP == "TZYX_FASTEST":
        return [mpi_rank // Gx // Gy // Gz, mpi_rank // Gx // Gy % Gz, mpi_rank // Gx % Gy, mpi_rank % Gx]
    else:
        _MPI_LOGGER.critical(f"Unsupported grid mapping {_GRID_MAP}", ValueError)


def _composition4(n):
    """
    Writing n as the sum of 4 natural numbers
    """
    addend: List[Tuple[int, int, int, int]] = []
    for i in range(n + 1):
        for j in range(i, n + 1):
            for k in range(j, n + 1):
                x, y, z, t = i, j - i, k - j, n - k
                addend.append((x, y, z, t))
    return addend


def _factorization4(k: int):
    """
    Writing k as the product of 4 positive numbers
    """
    prime_factor: List[List[Tuple[int, int, int, int]]] = []
    for p in range(2, int(k**0.5) + 1):
        n = 0
        while k % p == 0:
            n += 1
            k //= p
        if n != 0:
            prime_factor.append([(p**x, p**y, p**z, p**t) for x, y, z, t in _composition4(n)])
    if k != 1:
        prime_factor.append([(k**x, k**y, k**z, k**t) for x, y, z, t in _composition4(1)])
    return prime_factor


def _partition(factor: List[List[Tuple[int, int, int, int]]], idx: int, sublatt_size: List[int], grid_size: List[int]):
    if idx == 0:
        factor = _factorization4(factor)
    if idx == len(factor):
        yield grid_size
    else:
        Lx, Ly, Lz, Lt = sublatt_size
        Gx, Gy, Gz, Gt = grid_size
        for x, y, z, t in factor[idx]:
            if Lx % x == 0 and Ly % y == 0 and Lz % z == 0 and Lt % t == 0:
                yield from _partition(
                    factor, idx + 1, [Lx // x, Ly // y, Lz // z, Lt // t], [Gx * x, Gy * y, Gz * z, Gt * t]
                )


def _getDefaultGrid(mpi_size: int, latt_size: List[int]):
    Lx, Ly, Lz, Lt = latt_size
    latt_vol = Lx * Ly * Lz * Lt
    latt_surf = [latt_vol // latt_size[dir] for dir in range(4)]
    min_comm, min_grid = latt_vol, []
    assert latt_vol % mpi_size == 0, "lattice volume must be divisible by MPI size"
    assert Lx % 2 == 0 and Ly % 2 == 0 and Lz % 2 == 0 and Lt % 2 == 0, "lattice size must be even in all directions"
    for grid_size in _partition(mpi_size, 0, [Lx // 2, Ly // 2, Lz // 2, Lt // 2], [1, 1, 1, 1]):
        comm = [latt_surf[dir] * grid_size[dir] for dir in range(4) if grid_size[dir] > 1]
        if sum(comm) < min_comm:
            min_comm, min_grid = sum(comm), [grid_size]
        elif sum(comm) == min_comm:
            min_grid.append(grid_size)
    if min_grid == []:
        getLogger().critical(
            f"Cannot get the proper GPU grid for lattice size {latt_size} with {mpi_size} MPI processes"
        )
    return min(min_grid)


def initGrid(grid_size: List[int], latt_size: List[int] = None):
    if grid_size is None and latt_size is not None:
        grid_size = _getDefaultGrid(_MPI_SIZE, latt_size)
    grid_size = grid_size if grid_size is not None else [1, 1, 1, 1]
    global _GRID_SIZE, _GRID_COORD
    if _GRID_SIZE is None:
        Gx, Gy, Gz, Gt = grid_size
        if _MPI_SIZE != Gx * Gy * Gz * Gt:
            _MPI_LOGGER.critical(f"The MPI size {_MPI_SIZE} does not match the grid size {grid_size}", ValueError)
        _GRID_SIZE = [Gx, Gy, Gz, Gt]
        _GRID_COORD = getCoordFromRank(_MPI_RANK, _GRID_SIZE)
        _MPI_LOGGER.info(f"Using the grid size {_GRID_SIZE}")
    else:
        _MPI_LOGGER.warning("Grid is already initialized", RuntimeWarning)


def isGridInitialized():
    return _GRID_SIZE is not None


def getLogger():
    return _MPI_LOGGER


def setLoggerLevel(level: Literal["debug", "info", "warning", "error", "critical"]):
    _MPI_LOGGER.logger.setLevel(level.upper())


def getMPIComm():
    return _MPI_COMM


def getMPISize():
    return _MPI_SIZE


def getMPIRank():
    return _MPI_RANK


def getGridSize(check=True):
    if check and _GRID_SIZE is None:
        _MPI_LOGGER.critical("PyQUDA is not initialized", RuntimeError)
    return _GRID_SIZE


def getGridCoord(check=True):
    if check and _GRID_COORD is None:
        _MPI_LOGGER.critical("PyQUDA is not initialized", RuntimeError)
    return _GRID_COORD


def getGridMap():
    return _GRID_MAP


def setGridMap(grid_map: Literal["XYZT_FASTEST", "TZYX_FASTEST"]):
    global _GRID_MAP
    _GRID_MAP = grid_map
