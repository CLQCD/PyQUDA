from typing import Union
from time import perf_counter

import cupy as cp
from cupy.cuda.runtime import deviceSynchronize
from opt_einsum import contract

from check_pyquda import weak_field

from pyquda_utils import core, io, gamma
from pyquda_plugins import pycontract

core.init([1, 1, 1, 4], [24, 24, 24, 72], -1, 1.0, resource_path=".cache")
dirac = core.getDefaultDirac(-0.2400, 1e-8, 1000, 1.0, 1.160920226, 1.160920226)
latt_info = core.getDefaultLattice()

epsilon = cp.zeros((3, 3, 3), "<i4")
for i in range(3):
    j, k = (i + 1) % 3, (i + 2) % 3
    epsilon[i, j, k] = 1
    epsilon[i, k, j] = -1


def baryonTwoPoint(
    propag_i: core.LatticePropagator,
    propag_j: core.LatticePropagator,
    propag_m: core.LatticePropagator,
    contract_type,
    gamma_ij: gamma.Gamma,
    gamma_kl: gamma.Gamma,
    gamma_mn: Union[gamma.Gamma, gamma.Polarize],
):
    latt_info = propag_i.latt_info
    if contract_type == pycontract.BaryonContractType.IK_JL_MN:
        subscripts = "abc,def,ij,kl,mn,wtzyxikad,wtzyxjlbe,wtzyxmncf->wtzyx"
    elif contract_type == pycontract.BaryonContractType.IK_JN_ML:
        subscripts = "abc,def,ij,kl,mn,wtzyxikad,wtzyxjnbe,wtzyxmlcf->wtzyx"
    elif contract_type == pycontract.BaryonContractType.IL_JK_MN:
        subscripts = "abc,def,ij,kl,mn,wtzyxilad,wtzyxjkbe,wtzyxmncf->wtzyx"
    elif contract_type == pycontract.BaryonContractType.IL_JN_MK:
        subscripts = "abc,def,ij,kl,mn,wtzyxilad,wtzyxjnbe,wtzyxmkcf->wtzyx"
    elif contract_type == pycontract.BaryonContractType.IN_JK_ML:
        subscripts = "abc,def,ij,kl,mn,wtzyxinad,wtzyxjkbe,wtzyxmlcf->wtzyx"
    elif contract_type == pycontract.BaryonContractType.IN_JL_MK:
        subscripts = "abc,def,ij,kl,mn,wtzyxinad,wtzyxjlbe,wtzyxmkcf->wtzyx"
    return core.LatticeComplex(
        latt_info,
        contract(
            subscripts,
            epsilon,
            epsilon,
            gamma_ij.matrix,
            gamma_kl.matrix,
            gamma_mn.matrix,
            propag_i.data,
            propag_j.data,
            propag_m.data,
        ),
    )


gauge = io.readChromaQIOGauge("/public/ensemble/C24P29/beta6.20_mu-0.2770_ms-0.2400_L24x72_cfg_48000.lime")
gauge.stoutSmear(1, 0.125, 4)
dirac.loadGauge(gauge)
propag = core.invert(dirac, "wall", 0)
dirac.destroy()
# propag.save("C24P29.h5", "0")
# propag = LatticePropagator.load("C24P29.h5", "0")
# propag.toDevice()

propag_i = propag.copy()
propag_j = propag.copy()
propag_m = propag.copy()
propag_i.data -= 1
propag_m.data += 1

gamma_0 = gamma.Gamma(0)
gamma_1 = gamma.Gamma(1)
gamma_2 = gamma.Gamma(2)
gamma_3 = gamma.Gamma(4)
gamma_4 = gamma.Gamma(8)
gamma_5 = gamma.Gamma(15)

# a = gamma_1 @ propag_j @ gamma_3
# b = LatticePropagator(
#     propag_j.latt_info, contract("ij,wtzyxjkab,kl->wtzyxilab", gamma_1.matrix, propag_j.data, gamma_3.matrix)
# )
# print((a - b).norm2() ** 0.5 / a.norm2() ** 0.5)

C = gamma_2 @ gamma_4
CG_A = C @ gamma_2
CG_B = C @ gamma_4
Pp = (gamma_0 - gamma_4) / 2

for contract_type in [
    pycontract.BaryonContractType.IK_JL_MN,
    pycontract.BaryonContractType.IK_JN_ML,
    pycontract.BaryonContractType.IL_JK_MN,
    pycontract.BaryonContractType.IL_JN_MK,
    pycontract.BaryonContractType.IN_JK_ML,
    pycontract.BaryonContractType.IN_JL_MK,
]:
    deviceSynchronize()
    s = perf_counter()
    twopt = baryonTwoPoint(propag_i, propag_j, propag_m, contract_type, CG_A, CG_B, Pp)
    deviceSynchronize()
    core.getLogger().info(f"Time for einsum: {perf_counter() - s:.3f} sec")

    deviceSynchronize()
    s = perf_counter()
    twopt_ = pycontract.baryonTwoPoint(propag_i, propag_j, propag_m, contract_type, CG_A, CG_B, Pp)
    deviceSynchronize()
    core.getLogger().info(f"Time for pycontract: {perf_counter() - s:.3f} sec")

    core.getLogger().info(f"Relative error: {(twopt - twopt_).norm2() ** 0.5 / twopt.norm2() ** 0.5}")
