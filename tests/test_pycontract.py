from typing import Union
from time import perf_counter

import cupy as cp
from cupy.cuda.runtime import deviceSynchronize  # type: ignore
from opt_einsum import contract

from check_pyquda import data

from pyquda_utils import core, io, gamma
from pyquda_plugins import pycontract

core.init(resource_path=".cache/quda")
latt_info = core.LatticeInfo([4, 4, 4, 8])

epsilon = cp.zeros((3, 3, 3), "<i4")
for i in range(3):
    j, k = (i + 1) % 3, (i + 2) % 3
    epsilon[i, j, k] = 1
    epsilon[i, k, j] = -1


def mesonTwoPoint(
    propag_a: core.LatticePropagator,
    propag_b: core.LatticePropagator,
    gamma_ab: gamma.Gamma,
    gamma_de: gamma.Gamma,
):
    latt_info = propag_a.latt_info
    subscripts = "AB,DE,...ADab,...BEab->..."
    return core.LatticeComplex(
        latt_info,
        contract(
            subscripts,
            (gamma_ab.T @ gamma_5).matrix,
            (gamma_de @ gamma_5).matrix,
            propag_a.data,
            propag_b.data.conj(),
        ),
    )


def baryonTwoPoint(
    propag_i: core.LatticePropagator,
    propag_j: core.LatticePropagator,
    propag_n: core.LatticePropagator,
    contract_type,
    gamma_ij: gamma.Gamma,
    gamma_kl: gamma.Gamma,
    gamma_mn: Union[gamma.Gamma, gamma.Polarize],
):
    latt_info = propag_i.latt_info
    if contract_type == pycontract.BaryonContractType.IK_JL_NM:
        subscripts = "abc,def,ij,kl,mn,...ikad,...jlbe,...nmcf->..."
    elif contract_type == pycontract.BaryonContractType.IK_JM_NL:
        subscripts = "abc,def,ij,kl,mn,...ikad,...jmbe,...nlcf->..."
    elif contract_type == pycontract.BaryonContractType.IL_JK_NM:
        subscripts = "abc,def,ij,kl,mn,...ilad,...jkbe,...nmcf->..."
    elif contract_type == pycontract.BaryonContractType.IL_JM_NK:
        subscripts = "abc,def,ij,kl,mn,...ilad,...jmbe,...nkcf->..."
    elif contract_type == pycontract.BaryonContractType.IM_JK_NL:
        subscripts = "abc,def,ij,kl,mn,...imad,...jkbe,...nlcf->..."
    elif contract_type == pycontract.BaryonContractType.IM_JL_NK:
        subscripts = "abc,def,ij,kl,mn,...imad,...jlbe,...nkcf->..."
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
            propag_n.data,
        ),
    )


def mesonTwoPoint_v2(
    propag_fw: core.LatticePropagator,
    propag_bw: core.LatticePropagator,
    gamma_snk: gamma.Gamma,
    gamma_src: gamma.Gamma,
):
    latt_info = propag_fw.latt_info
    subscripts = "ij,...kjba,kl,...liba->..."
    return core.LatticeComplex(
        latt_info,
        contract(
            subscripts,
            (gamma_src @ gamma_5).matrix,
            propag_bw.data.conj(),
            (gamma_5 @ gamma_snk).matrix,
            propag_fw.data,
        ),
    )


def baryonDiquark(
    propag_i: core.LatticePropagator,
    propag_j: core.LatticePropagator,
    gamma_ij: gamma.Gamma,
    gamma_kl: gamma.Gamma,
):
    latt_info = propag_i.latt_info
    subscripts = "abc,def,ij,kl,...ikad,...jmbe->...mlcf"
    diquark = core.LatticePropagator(
        latt_info,
        contract(
            subscripts,
            epsilon,
            epsilon,
            gamma_ij.matrix,
            gamma_kl.matrix,
            propag_i.data,
            propag_j.data,
        ).conj(),
    )
    return gamma.gamma_mul_propagator_mul_gamma(gamma.Gamma(15), diquark, gamma.Gamma(15))


def baryonTwoPoint_v2(
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
        subscripts = "abc,def,ij,kl,mn,...ikad,...jlbe,...mncf->..."
    elif contract_type == pycontract.BaryonContractType.IK_JN_ML:
        subscripts = "abc,def,ij,kl,mn,...ikad,...jnbe,...mlcf->..."
    elif contract_type == pycontract.BaryonContractType.IL_JK_MN:
        subscripts = "abc,def,ij,kl,mn,...ilad,...jkbe,...mncf->..."
    elif contract_type == pycontract.BaryonContractType.IL_JN_MK:
        subscripts = "abc,def,ij,kl,mn,...ilad,...jnbe,...mkcf->..."
    elif contract_type == pycontract.BaryonContractType.IN_JK_ML:
        subscripts = "abc,def,ij,kl,mn,...inad,...jkbe,...mlcf->..."
    elif contract_type == pycontract.BaryonContractType.IN_JL_MK:
        subscripts = "abc,def,ij,kl,mn,...inad,...jlbe,...mkcf->..."
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


def baryonSequentialTwoPoint(
    propag_i: core.LatticePropagator,
    propag_j: core.LatticePropagator,
    propag_n: core.LatticePropagator,
    contract_type,
    sequential_type,
    gamma_ij: gamma.Gamma,
    gamma_kl: gamma.Gamma,
    gamma_mn: Union[gamma.Gamma, gamma.Polarize],
):
    latt_info = propag_i.latt_info
    if sequential_type == pycontract.BaryonSequentialType.SEQUENTIAL_I:
        if contract_type == pycontract.BaryonContractType.IK_JL_NM:
            subscripts = "abc,def,ij,kl,mn,...jlbe,...nmcf->...ikad"
        elif contract_type == pycontract.BaryonContractType.IK_JM_NL:
            subscripts = "abc,def,ij,kl,mn,...jmbe,...nlcf->...ikad"
        elif contract_type == pycontract.BaryonContractType.IL_JK_NM:
            subscripts = "abc,def,ij,kl,mn,...jkbe,...nmcf->...ilad"
        elif contract_type == pycontract.BaryonContractType.IL_JM_NK:
            subscripts = "abc,def,ij,kl,mn,...jmbe,...nkcf->...ilad"
        elif contract_type == pycontract.BaryonContractType.IM_JK_NL:
            subscripts = "abc,def,ij,kl,mn,...jkbe,...nlcf->...imad"
            # subscripts = "abc,def,ji,kl,mn,...ikbe,...nlcf->...jmad"
        elif contract_type == pycontract.BaryonContractType.IM_JL_NK:
            subscripts = "abc,def,ij,kl,mn,...jlbe,...nkcf->...imad"
            # subscripts = "abc,def,ji,lk,mn,...ikbe,...nlcf->...jmad"
        return core.LatticePropagator(
            latt_info,
            contract(
                subscripts,
                epsilon,
                epsilon,
                gamma_ij.matrix,
                gamma_kl.matrix,
                gamma_mn.matrix,
                propag_j.data,
                propag_n.data,
            ),
        )
    elif sequential_type == pycontract.BaryonSequentialType.SEQUENTIAL_J:
        if contract_type == pycontract.BaryonContractType.IK_JL_NM:
            subscripts = "abc,def,ij,kl,mn,...ikad,...nmcf->...jlbe"
        elif contract_type == pycontract.BaryonContractType.IK_JM_NL:
            subscripts = "abc,def,ij,kl,mn,...ikad,...nlcf->...jmbe"
        elif contract_type == pycontract.BaryonContractType.IL_JK_NM:
            subscripts = "abc,def,ij,kl,mn,...ilad,...nmcf->...jkbe"
        elif contract_type == pycontract.BaryonContractType.IL_JM_NK:
            subscripts = "abc,def,ij,kl,mn,...ilad,...nkcf->...jmbe"
        elif contract_type == pycontract.BaryonContractType.IM_JK_NL:
            subscripts = "abc,def,ij,kl,mn,...imad,...nlcf->...jkbe"
        elif contract_type == pycontract.BaryonContractType.IM_JL_NK:
            subscripts = "abc,def,ij,kl,mn,...imad,...nkcf->...jlbe"
        return core.LatticePropagator(
            latt_info,
            contract(
                subscripts,
                epsilon,
                epsilon,
                gamma_ij.matrix,
                gamma_kl.matrix,
                gamma_mn.matrix,
                propag_i.data,
                propag_n.data,
            ),
        )
    else:
        if contract_type == pycontract.BaryonContractType.IK_JL_NM:
            subscripts = "abc,def,ij,kl,mn,...ikad,...jlbe->...nmcf"
        elif contract_type == pycontract.BaryonContractType.IK_JM_NL:
            subscripts = "abc,def,ij,kl,mn,...ikad,...jmbe->...nlcf"
        elif contract_type == pycontract.BaryonContractType.IL_JK_NM:
            subscripts = "abc,def,ij,kl,mn,...ilad,...jkbe->...nmcf"
        elif contract_type == pycontract.BaryonContractType.IL_JM_NK:
            subscripts = "abc,def,ij,kl,mn,...ilad,...jmbe->...nkcf"
        elif contract_type == pycontract.BaryonContractType.IM_JK_NL:
            subscripts = "abc,def,ij,kl,mn,...imad,...jkbe->...nlcf"
        elif contract_type == pycontract.BaryonContractType.IM_JL_NK:
            subscripts = "abc,def,ij,kl,mn,...imad,...jlbe->...nkcf"
        return core.LatticePropagator(
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
            ),
        )


propag = io.readQIOPropagator(data("pt_prop_0"))
propag.toDevice()

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

C = gamma_2 @ gamma_4
CG_A = C @ gamma_4 @ gamma_5
CG_B = C @ gamma_5
Pp = (gamma_0 - gamma_3) / 2

deviceSynchronize()
s = perf_counter()
twopt = mesonTwoPoint_v2(propag_i, propag_j, CG_A, CG_B)
deviceSynchronize()
core.getLogger().info(f"Time for mesonTwoPoint: {perf_counter() - s:.3f} sec")

deviceSynchronize()
s = perf_counter()
twopt_ = pycontract.mesonTwoPoint(propag_i, propag_j, CG_A, CG_B)
deviceSynchronize()
core.getLogger().info(f"Time for pycontract.mesonTwoPoint: {perf_counter() - s:.3f} sec")

deviceSynchronize()
s = perf_counter()
twopts_ = pycontract.mesonAllSinkTwoPoint(propag_i, propag_j, CG_B)
deviceSynchronize()
core.getLogger().info(f"Time for pycontract.mesonAllSinkTwoPoint: {perf_counter() - s:.3f} sec")

core.getLogger().info(f"Relative error: {(twopt - twopt_).norm2() ** 0.5 / twopt.norm2() ** 0.5}")
core.getLogger().info(
    f"Relative error: {(twopt - CG_A.factor * twopts_[CG_A.index]).norm2() ** 0.5 / twopt.norm2() ** 0.5}"
)
core.getLogger().info(twopt.data[0, 0, 0, 0, 0])
core.getLogger().info(twopt_.data[0, 0, 0, 0, 0])
core.getLogger().info(twopts_[CG_A.index].data[0, 0, 0, 0, 0])

deviceSynchronize()
s = perf_counter()
diquark = baryonDiquark(propag_i, propag_j, CG_A, CG_B)
deviceSynchronize()
core.getLogger().info(f"Time for baryonDiquark: {perf_counter() - s:.3f} sec")

deviceSynchronize()
s = perf_counter()
diquark_ = pycontract.baryonDiquark(propag_i, propag_j, CG_A, CG_B)
deviceSynchronize()
core.getLogger().info(f"Time for pycontract.baryonDiquark: {perf_counter() - s:.3f} sec")

core.getLogger().info(f"Relative error: {(diquark - diquark_).norm2() ** 0.5 / diquark.norm2() ** 0.5}")
core.getLogger().info(diquark.data[0, 0, 0, 0, 0, 0, 0, 1, 0])
core.getLogger().info(diquark_.data[0, 0, 0, 0, 0, 0, 0, 1, 0])

for contract_type in [
    pycontract.BaryonContractType.IK_JL_NM,
    pycontract.BaryonContractType.IK_JM_NL,
    pycontract.BaryonContractType.IL_JK_NM,
    pycontract.BaryonContractType.IL_JM_NK,
    pycontract.BaryonContractType.IM_JK_NL,
    pycontract.BaryonContractType.IM_JL_NK,
]:
    deviceSynchronize()
    s = perf_counter()
    twopt = baryonTwoPoint(propag_i, propag_j, propag_m, contract_type, CG_A, CG_B, Pp)
    deviceSynchronize()
    core.getLogger().info(f"Time for baryonTwoPoint: {perf_counter() - s:.3f} sec")

    deviceSynchronize()
    s = perf_counter()
    twopt_ = pycontract.baryonTwoPoint(propag_i, propag_j, propag_m, contract_type, CG_A, CG_B, Pp)
    deviceSynchronize()
    core.getLogger().info(f"Time for pycontract.baryonTwoPoint: {perf_counter() - s:.3f} sec")

    # deviceSynchronize()
    # s = perf_counter()
    # twopt_ = pycontract.baryonGeneralTwoPoint(propag_i, propag_j, propag_m, contract_type, CG_A, CG_B, Pp.matrix.get())
    # deviceSynchronize()
    # core.getLogger().info(f"Time for pycontract.baryonTwoPoint: {perf_counter() - s:.3f} sec")

    core.getLogger().info(f"Relative error: {(twopt - twopt_).norm2() ** 0.5 / twopt.norm2() ** 0.5}")
    # core.getLogger().info(twopt.data[0, 0, 0, 0, 0])
    # core.getLogger().info(twopt_.data[0, 0, 0, 0, 0])

    deviceSynchronize()
    s = perf_counter()
    twopt2 = baryonTwoPoint_v2(propag_i, propag_j, propag_m, contract_type, CG_A, CG_B, Pp)
    deviceSynchronize()
    core.getLogger().info(f"Time for baryonTwoPoint_v2: {perf_counter() - s:.3f} sec")

    deviceSynchronize()
    s = perf_counter()
    twopt2_ = pycontract.baryonTwoPoint_v2(propag_i, propag_j, propag_m, contract_type, CG_A, CG_B, Pp)
    deviceSynchronize()
    core.getLogger().info(f"Time for pycontract.baryonTwoPoint_v2: {perf_counter() - s:.3f} sec")

    core.getLogger().info(f"Relative error: {(twopt2 - twopt2_).norm2() ** 0.5 / twopt2.norm2() ** 0.5}")
    # core.getLogger().info(twopt2.data[0, 0, 0, 0, 0])
    # core.getLogger().info(twopt2_.data[0, 0, 0, 0, 0])

for sequential_type in [
    pycontract.BaryonSequentialType.SEQUENTIAL_I,
    pycontract.BaryonSequentialType.SEQUENTIAL_J,
    pycontract.BaryonSequentialType.SEQUENTIAL_N,
]:
    for contract_type in [
        pycontract.BaryonContractType.IK_JL_NM,
        pycontract.BaryonContractType.IK_JM_NL,
        pycontract.BaryonContractType.IL_JK_NM,
        pycontract.BaryonContractType.IL_JM_NK,
        pycontract.BaryonContractType.IM_JK_NL,
        pycontract.BaryonContractType.IM_JL_NK,
    ]:
        deviceSynchronize()
        s = perf_counter()
        seq = baryonSequentialTwoPoint(propag_i, propag_j, propag_m, contract_type, sequential_type, CG_A, CG_B, Pp)
        deviceSynchronize()
        core.getLogger().info(f"Time for baryonTwoPoint: {perf_counter() - s:.3f} sec")

        deviceSynchronize()
        s = perf_counter()
        seq_ = pycontract.baryonSequentialTwoPoint(
            propag_i, propag_j, propag_m, contract_type, sequential_type, CG_A, CG_B, Pp
        )
        deviceSynchronize()
        core.getLogger().info(f"Time for pycontract.baryonTwoPoint: {perf_counter() - s:.3f} sec")

        # core.getLogger().info(f"sequential_type = {sequential_type}, contract_type = {contract_type}")
        core.getLogger().info(f"Relative error: {(seq - seq_).norm2() ** 0.5 / seq.norm2() ** 0.5}")
        # core.getLogger().info(seq.data[0, 0, 0, 0, 0, 0, 0, 1, 0])
        # core.getLogger().info(seq_.data[0, 0, 0, 0, 0, 0, 0, 1, 0])
