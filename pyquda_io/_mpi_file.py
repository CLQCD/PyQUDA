from typing import Sequence

from pyquda_comm import (  # noqa: F401
    initGrid,
    isGridInitialized,
    getCoordFromRank,
    getRankFromCoord,
    getSublatticeSize,
    getMPIComm,
    getMPISize,
    getMPIRank,
    getGridSize,
    getGridCoord,
    readMPIFile,
    writeMPIFile,
)


def getNeighbourRank():
    Gx, Gy, Gz, Gt = getGridSize()
    gx, gy, gz, gt = getCoordFromRank(getMPIRank())
    return [
        getRankFromCoord([(gx + 1) % Gx, gy, gz, gt]),
        getRankFromCoord([gx, (gy + 1) % Gy, gz, gt]),
        getRankFromCoord([gx, gy, (gz + 1) % Gz, gt]),
        getRankFromCoord([gx, gy, gz, (gt + 1) % Gt]),
        getRankFromCoord([(gx - 1) % Gx, gy, gz, gt]),
        getRankFromCoord([gx, (gy - 1) % Gy, gz, gt]),
        getRankFromCoord([gx, gy, (gz - 1) % Gz, gt]),
        getRankFromCoord([gx, gy, gz, (gt - 1) % Gt]),
    ]
