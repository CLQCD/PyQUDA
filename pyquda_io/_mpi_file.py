from typing import Sequence

from pyquda_comm import (  # noqa: F401
    initGrid,
    isGridInitialized,
    getMPIComm,
    getMPISize,
    getMPIRank,
    getGridSize,
    getGridCoord,
    getCoordFromRank,
    getRankFromCoord,
    readMPIFile,
    writeMPIFile,
)


def getSublatticeSize(latt_size: Sequence[int], evenodd: bool = True):
    GLx, GLy, GLz, GLt = latt_size
    Gx, Gy, Gz, Gt = getGridSize()
    if evenodd:
        assert GLx % (2 * Gx) == 0 and GLy % (2 * Gy) == 0 and GLz % (2 * Gz) == 0 and GLt % (2 * Gt) == 0
    else:
        assert GLx % Gx == 0 and GLy % Gy == 0 and GLz % Gz == 0 and GLt % Gt == 0
    return [GLx // Gx, GLy // Gy, GLz // Gz, GLt // Gt]


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
