from typing import Optional, Sequence

from pyquda_comm import (  # noqa: F401
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


def init(grid_size: Optional[Sequence[int]], latt_size: Optional[Sequence[int]] = None):
    initGrid("default", grid_size, latt_size, False)
    initDevice("numpy")
