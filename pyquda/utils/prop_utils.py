from typing import List

import numpy as np
import cupy as cp

from .. import mpi
from ..core import Nc, Ns, LatticePropagator


def collect(propagator: LatticePropagator, root: int = 0):
    Lx, Ly, Lz, Lt = propagator.latt_size
    Gx, Gy, Gz, Gt = mpi.grid
    sendbuf = propagator.data.get()
    if mpi.rank == root:
        recvbuf = np.zeros((Gt * Gz * Gy * Gx, Lt * Lz * Ly * Lx * Ns * Ns * Nc * Nc), "<c16")
    else:
        recvbuf = None
    if mpi.comm is not None:
        mpi.comm.Gatherv(sendbuf, recvbuf, root)
    else:
        recvbuf[0] = sendbuf
    if mpi.rank == root:
        data = np.zeros((2, Lt * Gt, Lz * Gz, Ly * Gy, Lx * Gx // 2, Ns, Ns, Nc, Nc), "<c16")
        for i in range(Gx * Gy * Gz * Gt):
            gt = i % Gt
            gz = i // Gt % Gz
            gy = i // Gt // Gz % Gy
            gx = i // Gt // Gz // Gy
            data[:, gt * Lt:(gt + 1) * Lt, gz * Lz:(gz + 1) * Lz, gy * Ly:(gy + 1) * Ly,
                 gx * Lx // 2:(gx + 1) * Lx // 2] = recvbuf[i].reshape(2, Lt, Lz, Ly, Lx // 2, Ns, Ns, Nc, Nc)

        ret = LatticePropagator([Lx * Gx, Ly * Gy, Lz * Gz, Lt * Gt])
        ret.data.set(data.reshape(-1))
        return ret
    else:
        return None


# def collect(propagator: LatticePropagator, root: int = 0):
#     Lx, Ly, Lz, Lt = propagator.latt_size
#     Gx, Gy, Gz, Gt = mpi.grid
#     sendbuf = propagator.lexico()
#     if mpi.rank == root:
#         recvbuf = np.zeros((Gt * Gz * Gy * Gx, Lt * Lz * Ly * Lx * Ns * Ns * Nc * Nc), "<c16")
#     else:
#         recvbuf = None
#     mpi.comm.Gatherv(sendbuf, recvbuf, root)
#     if mpi.rank == root:
#         lexico = np.zeros((Lt * Gt, Lz * Gz, Ly * Gy, Lx * Gx, Ns, Ns, Nc, Nc), "<c16")
#         cb2 = np.zeros((2, Lt * Gt, Lz * Gz, Ly * Gy, Lx * Gx // 2, Ns, Ns, Nc, Nc), "<c16")
#         for i in range(Gx * Gy * Gz * Gt):
#             gt = i % Gt
#             gz = i // Gt % Gz
#             gy = i // Gt // Gz % Gy
#             gx = i // Gt // Gz // Gy
#             lexico[gt * Lt:(gt + 1) * Lt, gz * Lz:(gz + 1) * Lz, gy * Ly:(gy + 1) * Ly,
#                    gx * Lx:(gx + 1) * Lx] = recvbuf[i].reshape(Lt, Lz, Ly, Lx, Ns, Ns, Nc, Nc)

#         for t in range(Lt * Gt):
#             for z in range(Lz * Gz):
#                 for y in range(Ly * Gy):
#                     eo = (t + z + y) % 2
#                     if eo == 0:
#                         cb2[0, t, z, y, :] = lexico[t, z, y, 0::2]
#                         cb2[1, t, z, y, :] = lexico[t, z, y, 1::2]
#                     else:
#                         cb2[0, t, z, y, :] = lexico[t, z, y, 1::2]
#                         cb2[1, t, z, y, :] = lexico[t, z, y, 0::2]

#         ret = LatticePropagator([Lx * Gx, Ly * Gy, Lz * Gz, Lt * Gt])
#         ret.data = cp.array(cb2)
#         return ret
#     else:
#         return None
