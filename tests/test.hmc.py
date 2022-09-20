import os
import sys
import numpy as np
import cupy as cp

test_dir = os.path.dirname(os.path.abspath(__file__))
# sys.path.insert(0, os.path.join(test_dir, ".."))
import pyquda
from pyquda import core, mpi
from pyquda.core import Nc

os.environ["QUDA_RESOURCE_PATH"] = ".cache"

ensembles = {
    "A1a": ([16, 16, 16, 16], 5.789),
    "B0a": ([24, 24, 24, 24], 6),
    "C1d": ([32, 32, 32, 64], 6.136),
    "D1d": ([48, 48, 48, 48], 6.475)
}

tag = "B0a"

latt_size = ensembles[tag][0]
Lx, Ly, Lz, Lt = latt_size
Vol = Lx * Ly * Lz * Lt

beta = ensembles[tag][1]
t = 1
dt = 0.1
steps = round(t / dt)
dt = t / steps

dslash = core.getDslash(latt_size, 0, 0, 0, anti_periodic_t=False)
gauge_param = dslash.gauge_param
gauge_param.overwrite_gauge = 0
gauge_param.overwrite_mom = 0
gauge_param.use_resident_gauge = 1
gauge_param.use_resident_mom = 1
gauge_param.make_resident_gauge = 1
gauge_param.make_resident_mom = 1
gauge_param.return_result_gauge = 0
gauge_param.return_result_mom = 0

gauge = core.LatticeGauge(latt_size, None, True)
gauge.data[:] = cp.diag([1, 1, 1])

mpi.init()
pyquda.loadGauge(gauge, gauge_param)
pyquda.momResident(gauge, gauge_param)


def path_force(path, coeffs):
    num_paths = len(path)
    lengths = []
    force = [[], [], [], []]
    fcoeffs = [[], [], [], []]
    flengths = [[], [], [], []]
    for i in range(num_paths):
        lengths.append(len(path[i]))
        loop = np.array(path[i])
        loop_dag = np.flip(7 - loop)
        for j in range(lengths[i]):
            if loop[j] < 4:
                force[loop[j]].append(np.roll(loop, -j)[1:])
                fcoeffs[loop[j]].append(-coeffs[i])
                flengths[loop[j]].append(lengths[i] - 1)
            else:
                force[loop_dag[3 - j]].append(np.roll(loop_dag, j - 3)[1:])
                fcoeffs[loop_dag[3 - j]].append(-coeffs[i])
                flengths[loop_dag[3 - j]].append(lengths[i] - 1)
    max_length = max(lengths)
    return (
        num_paths, max_length, np.array(path, dtype="<i4"), np.array(lengths, dtype="<i4"), np.array(coeffs),
        np.array(force, dtype="<i4"), np.array(flengths[0], dtype="<i4"), np.array(fcoeffs[0])
    )


input_path = [
    [0, 1, 7, 6],
    [0, 2, 7, 5],
    [0, 3, 7, 4],
    [1, 2, 6, 5],
    [1, 3, 6, 4],
    [2, 3, 5, 4],
]
input_coeffs = [
    -beta / Nc,
    -beta / Nc,
    -beta / Nc,
    -beta / Nc,
    -beta / Nc,
    -beta / Nc,
]

num_paths, max_length, path, lengths, coeffs, force, flengths, fcoeffs = path_force(input_path, input_coeffs)

# input_path_buf = np.array(
#     [
#         [[1, 7, 6], [6, 7, 1], [2, 7, 5], [5, 7, 2], [3, 7, 4], [4, 7, 3]],
#         [[0, 6, 7], [7, 6, 0], [2, 6, 5], [5, 6, 2], [3, 6, 4], [4, 6, 3]],
#         [[0, 5, 7], [7, 5, 0], [1, 5, 6], [6, 5, 1], [3, 5, 4], [4, 5, 3]],
#         [[0, 4, 7], [7, 4, 0], [1, 4, 6], [6, 4, 1], [2, 4, 5], [5, 4, 2]]
#     ],
#     dtype="<i4"
# )

rho_ = 0.2539785108410595
theta_ = -0.03230286765269967
vartheta_ = 0.08398315262876693
lambda_ = 0.6822365335719091

warm = True
for i in range(100):
    pyquda.gaussMom(i)
    pyquda.saveGauge(gauge, gauge_param)

    kinetic = pyquda.momAction(gauge_param)
    potential = pyquda.computeGaugeLoopTrace(1, path, lengths, coeffs, num_paths, max_length)
    energy = kinetic + potential

    for step in range(steps):
        pyquda.computeGaugeForce(vartheta_ * dt, force, flengths, fcoeffs, num_paths, max_length - 1, gauge_param)
        pyquda.updateGaugeField(rho_ * dt, gauge_param)
        pyquda.computeGaugeForce(lambda_ * dt, force, flengths, fcoeffs, num_paths, max_length - 1, gauge_param)
        pyquda.updateGaugeField(theta_ * dt, gauge_param)
        pyquda.computeGaugeForce(
            (1 - 2 * (lambda_ + vartheta_)) * dt / 2, force, flengths, fcoeffs, num_paths, max_length - 1, gauge_param
        )
        pyquda.updateGaugeField((1 - 2 * (theta_ + rho_)) * dt, gauge_param)
        pyquda.computeGaugeForce(
            (1 - 2 * (lambda_ + vartheta_)) * dt / 2, force, flengths, fcoeffs, num_paths, max_length - 1, gauge_param
        )
        pyquda.updateGaugeField(theta_ * dt, gauge_param)
        pyquda.computeGaugeForce(lambda_ * dt, force, flengths, fcoeffs, num_paths, max_length - 1, gauge_param)
        pyquda.updateGaugeField(rho_ * dt, gauge_param)
        pyquda.computeGaugeForce(vartheta_ * dt, force, flengths, fcoeffs, num_paths, max_length - 1, gauge_param)

    kinetic1 = pyquda.momAction(gauge_param)
    potential1 = pyquda.computeGaugeLoopTrace(1, path, lengths, coeffs, num_paths, max_length)
    energy1 = kinetic1 + potential1

    accept = np.random.rand() < np.exp(energy - energy1) or warm
    if energy1 < energy:
        warm = False
    if not accept:
        pyquda.loadGauge(gauge, gauge_param)

    plaquette = pyquda.plaq()

    print(
        f'''
PE_old = {potential}, KE_old = {kinetic}
PE = {potential1}, KE = {kinetic1}
Delta_PE = {potential1 - potential}, Delta_KE = {kinetic1 - kinetic}
Delta_E = {energy1 - energy}
accept = {accept}
plaquette = {plaquette}
'''
    )

dslash.destroy()
