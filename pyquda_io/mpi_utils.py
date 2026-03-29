from typing import Optional, Sequence

from pyquda_comm import (  # noqa: F401
    MPI,
    initGrid,
    initDevice,
    getMPIComm,
    getMPIRank,
    getGridCoord,
    getNeighbourRank,
    getSublatticeSize,
    openReadHeader,
    openWriteHeader,
    readMPIFile,
    writeMPIFile,
)


def init(
    grid_size: Optional[Sequence[int]],
    latt_size: Optional[Sequence[int]] = None,
    mpi_comm: Optional[MPI.Intracomm] = None,
):
    initGrid(mpi_comm, "default", grid_size, latt_size, False)
    initDevice()
