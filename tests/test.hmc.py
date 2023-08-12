import os
import sys
import numpy as np

test_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(test_dir, ".."))
import pyquda
from pyquda import field
from pyquda.hmc import HMC
from pyquda.field import Nc

# field.CUDA_BACKEND = "torch"

os.environ["QUDA_RESOURCE_PATH"] = ".cache"
pyquda.init()

ensembles = {
    "A1": ([16, 16, 16, 16], 5.789),
    "B0": ([24, 24, 24, 24], 6),
    "C2": ([32, 32, 32, 32], 6.179),
    "D1": ([48, 48, 48, 48], 6.475),
}

tag = "A1"

latt_size = ensembles[tag][0]
Lx, Ly, Lz, Lt = latt_size
Vol = Lx * Ly * Lz * Lt

beta = ensembles[tag][1]

gauge = field.LatticeGauge(latt_size, None, True)

hmc = HMC(latt_size, 0, 0, 0)
hmc.loadGauge(gauge)

gauge_param = hmc.gauge_param

hmc.loadMom(gauge)


def loop_ndarray(path, num_paths, max_length):
    ret = -np.ones((num_paths, max_length), "<i4")
    for i in range(num_paths):
        for j in range(len(path[i])):
            ret[i, j] = path[i][j]
    return ret


def path_ndarray(path, num_paths, max_length):
    ret = -np.ones((4, num_paths, max_length), "<i4")
    for d in range(4):
        for i in range(num_paths):
            for j in range(len(path[d][i])):
                ret[d, i, j] = path[d][i][j]
    return ret


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
                force[loop_dag[lengths[i] - 1 - j]].append(np.roll(loop_dag, j + 1 - lengths[i])[1:])
                fcoeffs[loop_dag[lengths[i] - 1 - j]].append(-coeffs[i])
                flengths[loop_dag[lengths[i] - 1 - j]].append(lengths[i] - 1)
    max_length = max(lengths)
    lengths = np.array(lengths, dtype="<i4")
    coeffs = np.array(coeffs, "<f8")
    path = loop_ndarray(path, num_paths, max_length)
    assert flengths[0] == flengths[1] == flengths[2] == flengths[3], "path is not symmetric"
    flengths = np.array(flengths[0], "<i4")
    assert fcoeffs[0] == fcoeffs[1] == fcoeffs[2] == fcoeffs[3], "path is not symmetric"
    fcoeffs = np.array(fcoeffs[0], "<f8")
    num_fpaths = len(flengths)
    force = path_ndarray(force, num_fpaths, max_length - 1)
    return num_paths, max_length, path, lengths, coeffs, force, num_fpaths, flengths, fcoeffs


input_path = [
    [0, 1, 7, 6],
    [0, 2, 7, 5],
    [0, 3, 7, 4],
    [1, 2, 6, 5],
    [1, 3, 6, 4],
    [2, 3, 5, 4],
    [0, 1, 1, 7, 6, 6],
    [0, 2, 2, 7, 5, 5],
    [0, 3, 3, 7, 4, 4],
    [1, 0, 0, 6, 7, 7],
    [1, 2, 2, 6, 5, 5],
    [1, 3, 3, 6, 4, 4],
    [2, 0, 0, 5, 7, 7],
    [2, 1, 1, 5, 6, 6],
    [2, 3, 3, 5, 4, 4],
    [3, 0, 0, 4, 7, 7],
    [3, 1, 1, 4, 6, 6],
    [3, 2, 2, 4, 5, 5],
]
input_coeffs = [
    -5 / 3,
    -5 / 3,
    -5 / 3,
    -5 / 3,
    -5 / 3,
    -5 / 3,
    1 / 12,
    1 / 12,
    1 / 12,
    1 / 12,
    1 / 12,
    1 / 12,
    1 / 12,
    1 / 12,
    1 / 12,
    1 / 12,
    1 / 12,
    1 / 12,
]

# input_path = [
#     [0, 1, 7, 6],
#     [0, 2, 7, 5],
#     [0, 3, 7, 4],
#     [1, 2, 6, 5],
#     [1, 3, 6, 4],
#     [2, 3, 5, 4],
# ]
# input_coeffs = [
#     -1,
#     -1,
#     -1,
#     -1,
#     -1,
#     -1,
# ]

num_paths, max_length, path, lengths, coeffs, force, num_fpaths, flengths, fcoeffs = path_force(
    input_path, input_coeffs
)
coeffs *= beta / Nc
fcoeffs *= beta / Nc

rho_ = 0.2539785108410595
theta_ = -0.03230286765269967
vartheta_ = 0.08398315262876693
lambda_ = 0.6822365335719091

plaquette = pyquda.plaq()
print(f"\nplaquette = {plaquette}\n")

t = 1.0
dt = 0.1
steps = round(t / dt)
dt = t / steps
warm = 20
for i in range(100):
    hmc.gaussMom(i)

    kinetic = hmc.actionMom()
    potential = hmc.actionGauge(path, lengths, coeffs, num_paths, max_length)
    energy = kinetic + potential

    for step in range(steps):
        hmc.computeGaugeForce(vartheta_ * dt, force, flengths, fcoeffs, num_fpaths, max_length - 1)
        hmc.updateGaugeField(rho_ * dt)
        hmc.computeGaugeForce(lambda_ * dt, force, flengths, fcoeffs, num_fpaths, max_length - 1)
        hmc.updateGaugeField(theta_ * dt)
        hmc.computeGaugeForce((0.5 - (lambda_ + vartheta_)) * dt, force, flengths, fcoeffs, num_fpaths, max_length - 1)
        hmc.updateGaugeField((1.0 - 2 * (theta_ + rho_)) * dt)
        hmc.computeGaugeForce((0.5 - (lambda_ + vartheta_)) * dt, force, flengths, fcoeffs, num_fpaths, max_length - 1)
        hmc.updateGaugeField(theta_ * dt)
        hmc.computeGaugeForce(lambda_ * dt, force, flengths, fcoeffs, num_fpaths, max_length - 1)
        hmc.updateGaugeField(rho_ * dt)
        hmc.computeGaugeForce(vartheta_ * dt, force, flengths, fcoeffs, num_fpaths, max_length - 1)

    hmc.reunitGaugeField(gauge, 1e-15)

    kinetic1 = hmc.actionMom()
    potential1 = hmc.actionGauge(path, lengths, coeffs, num_paths, max_length)
    energy1 = kinetic1 + potential1

    accept = np.random.rand() < np.exp(energy - energy1)
    if warm > 0:
        warm -= 1
    if accept or warm:
        hmc.saveGauge(gauge)
    else:
        hmc.loadGauge(gauge)

    plaquette = pyquda.plaq()

    print(
        f"Step {i}:\n"
        f"PE_old = {potential}, KE_old = {kinetic}\n"
        f"PE = {potential1}, KE = {kinetic1}\n"
        f"Delta_PE = {potential1 - potential}, Delta_KE = {kinetic1 - kinetic}\n"
        f"Delta_E = {energy1 - energy}\n"
        f"accept rate = {min(1, np.exp(energy - energy1))*100:.2f}%\n"
        f"accept? {accept or not not warm}\n"
        f"plaquette = {plaquette}\n"
    )
