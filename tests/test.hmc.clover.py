import numpy as np
import cupy as cp

from check_pyquda import test_dir

from pyquda import init
from pyquda.hmc_clover import HMC
from pyquda.field import Ns, Nc, LatticeInfo, LatticeFermion, LatticeGauge

ensembles = {
    "A1": ([16, 16, 16, 16], 5.789),
    "B0": ([24, 24, 24, 24], 6),
    "C2": ([32, 32, 32, 32], 6.179),
    "D1": ([48, 48, 48, 48], 6.475),
}

tag = "A1"

init(resource_path=".cache")
latt_info = LatticeInfo(ensembles[tag][0], -1, 1.0)
beta = ensembles[tag][1]
Lx, Ly, Lz, Lt = latt_info.size

gauge = LatticeGauge(latt_info, None)

mass = 4
kappa = 1 / (2 * (mass + 4))
csw = 1.0
const_fourth_root = 6.10610118771501
residue_fourth_root = [
    -5.90262826538435e-06,
    -2.63363387226834e-05,
    -8.62160355606352e-05,
    -0.000263984258286453,
    -0.000792810319715722,
    -0.00236581977385576,
    -0.00704746125114149,
    -0.0210131715847004,
    -0.0629242233443976,
    -0.190538104129215,
    -0.592816342814611,
    -1.96992441194278,
    -7.70705574740274,
    -46.55440910469,
    -1281.70053339288,
]
offset_fourth_root = [
    0.000109335909283339,
    0.000584211769074023,
    0.00181216713967916,
    0.00478464392272826,
    0.0119020708754186,
    0.0289155646996088,
    0.0695922442548162,
    0.166959610676697,
    0.400720136243831,
    0.965951931276981,
    2.35629923417205,
    5.92110728201649,
    16.0486180482883,
    53.7484938194392,
    402.99686403222,
]
residue_inv_square_root = [
    0.00943108618345698,
    0.0122499930158508,
    0.0187308029056777,
    0.0308130330025528,
    0.0521206555919226,
    0.0890870585774984,
    0.153090120000215,
    0.26493803350899,
    0.466760251501358,
    0.866223656646014,
    1.8819154073627,
    6.96033769739192,
]
offset_inv_square_root = [
    5.23045292201785e-05,
    0.000569214182255549,
    0.00226724207135389,
    0.00732861083302471,
    0.0222608882919378,
    0.0662886891030569,
    0.196319420401789,
    0.582378159903323,
    1.74664271771668,
    5.42569216297222,
    18.850085313508,
    99.6213166072174,
]

hmc = HMC(
    latt_info,
    mass,
    1e-9,
    1000,
    csw,
    1,
    const_fourth_root,
    residue_fourth_root,
    offset_fourth_root,
    residue_inv_square_root,
    offset_inv_square_root,
)
# hmc = HMC(latt_info, mass, 1e-9, 1000, csw, 2)
hmc.loadGauge(gauge)
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
    [0, 0, 1, 7, 7, 6],
    [0, 0, 2, 7, 7, 5],
    [0, 0, 3, 7, 7, 4],
    [1, 1, 0, 6, 6, 7],
    [1, 1, 2, 6, 6, 5],
    [1, 1, 3, 6, 6, 4],
    [2, 2, 0, 5, 5, 7],
    [2, 2, 1, 5, 5, 6],
    [2, 2, 3, 5, 5, 4],
    [3, 3, 0, 4, 4, 7],
    [3, 3, 1, 4, 4, 6],
    [3, 3, 2, 4, 4, 5],
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

num_paths, max_length, path, lengths, coeffs, force, num_fpaths, flengths, fcoeffs = path_force(
    input_path, input_coeffs
)
coeffs *= beta / Nc
fcoeffs *= beta / Nc

rho_ = 0.2539785108410595
theta_ = -0.03230286765269967
vartheta_ = 0.08398315262876693
lambda_ = 0.6822365335719091

plaquette = hmc.plaquette()
print(f"\nplaquette = {plaquette}\n")

t = 1.0
dt = 0.2
steps = round(t / dt)
dt = t / steps
warm = 20
for i in range(100):
    hmc.gaussMom(i)

    # np.random.seed(i)
    # phi = 2 * np.pi * np.random.random((2, Lt, Lz, Ly, Lx // 2, Ns, Nc))
    # r = np.random.random((2, Lt, Lz, Ly, Lx // 2, Ns, Nc))

    cp.random.seed(i)
    phi = 2 * cp.pi * cp.random.random((2, Lt, Lz, Ly, Lx // 2, Ns, Nc), "<f8")
    r = cp.random.random((2, Lt, Lz, Ly, Lx // 2, Ns, Nc), "<f8")

    # cp.random.manual_seed(i)
    # phi = 2 * cp.pi * cp.rand((2, Lt, Lz, Ly, Lx // 2, Ns, Nc), dtype=cp.float64)
    # r = cp.rand((2, Lt, Lz, Ly, Lx // 2, Ns, Nc), dtype=cp.float64)

    noise = LatticeFermion(latt_info, cp.sqrt(-cp.log(r)) * (cp.cos(phi) + 1j * cp.sin(phi)))

    hmc.initNoise(noise, i)

    kinetic = hmc.actionMom()
    potential = hmc.actionGauge(path, lengths, coeffs, num_paths, max_length)
    potential += hmc.actionFermion(noise)
    energy = kinetic + potential

    for step in range(steps):
        hmc.computeGaugeForce(vartheta_ * dt, force, flengths, fcoeffs, num_fpaths, max_length - 1)
        hmc.computeCloverForce(vartheta_ * dt, noise, -(kappa**2), -kappa * csw / 8)
        hmc.updateGaugeField(rho_ * dt)
        hmc.computeGaugeForce(lambda_ * dt, force, flengths, fcoeffs, num_fpaths, max_length - 1)
        hmc.computeCloverForce(lambda_ * dt, noise, -(kappa**2), -kappa * csw / 8)
        hmc.updateGaugeField(theta_ * dt)
        hmc.computeGaugeForce((0.5 - (lambda_ + vartheta_)) * dt, force, flengths, fcoeffs, num_fpaths, max_length - 1)
        hmc.computeCloverForce((0.5 - (lambda_ + vartheta_)) * dt, noise, -(kappa**2), -kappa * csw / 8)
        hmc.updateGaugeField((1.0 - 2 * (theta_ + rho_)) * dt)
        hmc.computeGaugeForce((0.5 - (lambda_ + vartheta_)) * dt, force, flengths, fcoeffs, num_fpaths, max_length - 1)
        hmc.computeCloverForce((0.5 - (lambda_ + vartheta_)) * dt, noise, -(kappa**2), -kappa * csw / 8)
        hmc.updateGaugeField(theta_ * dt)
        hmc.computeGaugeForce(lambda_ * dt, force, flengths, fcoeffs, num_fpaths, max_length - 1)
        hmc.computeCloverForce(lambda_ * dt, noise, -(kappa**2), -kappa * csw / 8)
        hmc.updateGaugeField(rho_ * dt)
        hmc.computeGaugeForce(vartheta_ * dt, force, flengths, fcoeffs, num_fpaths, max_length - 1)
        hmc.computeCloverForce(vartheta_ * dt, noise, -(kappa**2), -kappa * csw / 8)

    hmc.reunitGaugeField(1e-15)

    kinetic1 = hmc.actionMom()
    potential1 = hmc.actionGauge(path, lengths, coeffs, num_paths, max_length)
    potential1 += hmc.actionFermion(noise)
    energy1 = kinetic1 + potential1

    accept = np.random.rand() < np.exp(energy - energy1)
    if warm > 0:
        warm -= 1
    if accept or warm:
        hmc.saveGauge(gauge)
    else:
        hmc.loadGauge(gauge)

    plaquette = hmc.plaquette()

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
