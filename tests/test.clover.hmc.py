import os
import sys
import numpy as np
import cupy as cp

test_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(test_dir, ".."))

import pyquda
from pyquda import core, field, enum_quda, nullptr
from pyquda.field import Nc, Ns

os.environ["QUDA_RESOURCE_PATH"] = ".cache"
pyquda.init()

ensembles = {
    "A1": ([16, 16, 16, 16], 5.789),
    "B0": ([24, 24, 24, 24], 6),
    "C2": ([32, 32, 32, 32], 6.179),
    "D1": ([48, 48, 48, 48], 6.475)
}

tag = "A1"

latt_size = ensembles[tag][0]
Lx, Ly, Lz, Lt = latt_size
Vol = Lx * Ly * Lz * Lt

beta = ensembles[tag][1]

gauge = field.LatticeGauge(latt_size, None, True)
gauge.data[:] = cp.identity(Nc)

mass = 4
kappa = 1 / (2 * (mass + 4))
csw = 1.0
dslash = core.getDslash(latt_size, mass, 1e-9, 1000, clover_coeff_t=csw, anti_periodic_t=False)
dslash.loadGauge(gauge)

gauge_param = dslash.gauge_param
gauge_param.overwrite_gauge = 0
gauge_param.overwrite_mom = 0
gauge_param.use_resident_gauge = 1
gauge_param.use_resident_mom = 1
gauge_param.make_resident_gauge = 1
gauge_param.make_resident_mom = 1
gauge_param.return_result_gauge = 0
gauge_param.return_result_mom = 0
invert_param = dslash.invert_param
invert_param.inv_type = enum_quda.QudaInverterType.QUDA_BICGSTAB_INVERTER
invert_param.solve_type = enum_quda.QudaSolveType.QUDA_NORMOP_PC_SOLVE
invert_param.matpc_type = enum_quda.QudaMatPCType.QUDA_MATPC_EVEN_EVEN_ASYMMETRIC
invert_param.solution_type = enum_quda.QudaSolutionType.QUDA_MATPCDAG_MATPC_SOLUTION
invert_param.mass_normalization = enum_quda.QudaMassNormalization.QUDA_ASYMMETRIC_MASS_NORMALIZATION
invert_param.verbosity = enum_quda.QudaVerbosity.QUDA_SILENT
invert_param.compute_action = 1
invert_param.compute_clover_trlog = 1

# pyquda.loadGauge(gauge, gauge_param)
pyquda.momResident(gauge, gauge_param)


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
    -5 / 3, -5 / 3, -5 / 3, -5 / 3, -5 / 3, -5 / 3, 1 / 12, 1 / 12, 1 / 12, 1 / 12, 1 / 12, 1 / 12, 1 / 12, 1 / 12,
    1 / 12, 1 / 12, 1 / 12, 1 / 12
]

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

t = 1
dt = 0.02
steps = round(t / dt)
dt = t / steps
warm = 100
for i in range(100):
    pyquda.gaussMom(i)

    phi = 2 * cp.pi * cp.random.random((2, Lt, Lz, Ly, Lx // 2, Ns, Nc))
    r = cp.random.random((2, Lt, Lz, Ly, Lx // 2, Ns, Nc))
    noise = core.LatticeFermion(latt_size, cp.sqrt(-cp.log(r)) * (cp.cos(phi) + 1j * cp.sin(phi)))

    # invert_param.dagger = enum_quda.QudaDagType.QUDA_DAG_YES
    # pyquda.quda.MatQuda(noise.odd_ptr, noise.odd_ptr, invert_param)
    # invert_param.dagger = enum_quda.QudaDagType.QUDA_DAG_NO

    pyquda.quda.freeCloverQuda()
    pyquda.quda.loadCloverQuda(nullptr, nullptr, invert_param)
    pyquda.quda.invertQuda(noise.even_ptr, noise.odd_ptr, invert_param)

    kinetic = pyquda.momAction(gauge_param)
    potential = pyquda.computeGaugeLoopTrace(1, path, lengths, coeffs, num_paths,
                                             max_length) + invert_param.action[0] - invert_param.trlogA[1]
    energy = kinetic + potential

    pyquda.computeGaugeForce(0.5 * dt, force, flengths, fcoeffs, num_fpaths, max_length - 1, gauge_param)
    pyquda.computeCloverForce(0.5 * dt, noise, -kappa**2, -kappa * csw / 8, 2, gauge_param, invert_param)
    for step in range(steps - 1):
        pyquda.updateGaugeField(1.0 * dt, gauge_param)
        pyquda.computeGaugeForce(1.0 * dt, force, flengths, fcoeffs, num_fpaths, max_length - 1, gauge_param)
        pyquda.computeCloverForce(1.0 * dt, noise, -kappa**2, -kappa * csw / 8, 2, gauge_param, invert_param)
    pyquda.updateGaugeField(0.5 * dt, gauge_param)

    pyquda.projectSU3(1e-15, gauge_param)

    pyquda.quda.freeCloverQuda()
    pyquda.quda.loadCloverQuda(nullptr, nullptr, invert_param)
    pyquda.quda.invertQuda(noise.even_ptr, noise.odd_ptr, invert_param)

    kinetic1 = pyquda.momAction(gauge_param)
    potential1 = pyquda.computeGaugeLoopTrace(1, path, lengths, coeffs, num_paths,
                                              max_length) + invert_param.action[0] - invert_param.trlogA[1]
    energy1 = kinetic1 + potential1

    accept = np.random.rand() < np.exp(energy - energy1)
    if warm > 0:
        warm -= 1
    if accept or warm:
        pyquda.saveGauge(gauge, gauge_param)
    else:
        pyquda.loadGauge(gauge, gauge_param)

    plaquette = pyquda.plaq()

    print(
        f'Step {i}:\n'
        f'PE_old = {potential}, KE_old = {kinetic}\n'
        f'PE = {potential1}, KE = {kinetic1}\n'
        f'Delta_PE = {potential1 - potential}, Delta_KE = {kinetic1 - kinetic}\n'
        f'Delta_E = {energy1 - energy}\n'
        f'accept = {accept or not not warm}\n'
        f'plaquette = {plaquette}\n'
    )

dslash.destroy()
