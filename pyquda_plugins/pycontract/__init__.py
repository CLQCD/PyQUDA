from typing import Union
import numpy
from numpy.typing import NDArray

from pyquda_comm import getLogger
from pyquda_comm.field import LatticeComplex, MultiLatticeComplex, LatticePropagator
from pyquda_utils.gamma import Gamma, Polarize

from . import _pycontract as contract
from ._pycontract import BaryonContractType, BaryonSequentialType


def mesonTwoPoint(
    propag_a: LatticePropagator,
    propag_b: LatticePropagator,
    gamma_ab: Gamma,
    gamma_dc: Gamma,
):
    latt_info = propag_a.latt_info
    assert latt_info.Nd == 4 and latt_info.Ns == 4 and latt_info.Nc == 3
    correl = LatticeComplex(latt_info)
    contract.meson_two_point(
        correl.data_ptr,
        propag_a.data_ptr,
        propag_b.data_ptr,
        latt_info.volume,
        gamma_ab.index,
        gamma_dc.index,
    )
    correl *= gamma_ab.factor * gamma_dc.factor
    return correl


def mesonAllSinkTwoPoint(
    propag_a: LatticePropagator,
    propag_b: LatticePropagator,
    gamma_dc: Gamma,
):
    latt_info = propag_a.latt_info
    assert latt_info.Nd == 4 and latt_info.Ns == 4 and latt_info.Nc == 3
    correl = MultiLatticeComplex(latt_info, 16)
    contract.meson_all_sink_two_point(
        correl.data_ptrs,
        propag_a.data_ptr,
        propag_b.data_ptr,
        latt_info.volume,
        gamma_dc.index,
    )
    correl *= gamma_dc.factor
    return correl


def mesonAllSourceTwoPoint(
    propag_a: LatticePropagator,
    propag_b: LatticePropagator,
    gamma_ab: Gamma,
):
    latt_info = propag_a.latt_info
    assert latt_info.Nd == 4 and latt_info.Ns == 4 and latt_info.Nc == 3
    correl = MultiLatticeComplex(latt_info, 16)
    contract.meson_all_source_two_point(
        correl.data_ptrs,
        propag_a.data_ptr,
        propag_b.data_ptr,
        latt_info.volume,
        gamma_ab.index,
    )
    correl *= gamma_ab.factor
    return correl


def baryonDiquark(
    propag_i: LatticePropagator,
    propag_j: LatticePropagator,
    gamma_ij: Gamma,
    gamma_kl: Gamma,
):
    latt_info = propag_i.latt_info
    assert latt_info.Nd == 4 and latt_info.Ns == 4 and latt_info.Nc == 3
    diquark = LatticePropagator(latt_info)
    contract.baryon_diquark(
        diquark.data_ptr,
        propag_i.data_ptr,
        propag_j.data_ptr,
        latt_info.volume,
        gamma_ij.index,
        gamma_kl.index,
    )
    diquark *= gamma_ij.factor * gamma_kl.factor
    return diquark


def baryonTwoPoint(
    propag_i: LatticePropagator,
    propag_j: LatticePropagator,
    propag_n: LatticePropagator,
    contract_type: BaryonContractType,
    gamma_ij: Gamma,
    gamma_kl: Gamma,
    gamma_mn: Union[Gamma, Polarize],
) -> LatticeComplex:
    if isinstance(gamma_mn, Gamma):
        latt_info = propag_i.latt_info
        assert latt_info.Nd == 4 and latt_info.Ns == 4 and latt_info.Nc == 3
        correl = LatticeComplex(latt_info)
        contract.baryon_two_point(
            correl.data_ptr,
            propag_i.data_ptr,
            propag_j.data_ptr,
            propag_n.data_ptr,
            contract_type,
            latt_info.volume,
            gamma_ij.index,
            gamma_kl.index,
            gamma_mn.index,
        )
        correl *= gamma_ij.factor * gamma_kl.factor * gamma_mn.factor
        return correl
    elif isinstance(gamma_mn, Polarize):
        correl_left = baryonTwoPoint(propag_i, propag_j, propag_n, contract_type, gamma_ij, gamma_kl, gamma_mn.left)
        correl_right = baryonTwoPoint(propag_i, propag_j, propag_n, contract_type, gamma_ij, gamma_kl, gamma_mn.right)
        return correl_left + correl_right
    else:
        raise getLogger().critical("gamma_mn should be Gamma or Polarize", ValueError)


def baryonGeneralTwoPoint(
    propag_i: LatticePropagator,
    propag_j: LatticePropagator,
    propag_n: LatticePropagator,
    contract_type: BaryonContractType,
    gamma_ij: Gamma,
    gamma_kl: Gamma,
    project_mn: NDArray[numpy.complex128],
) -> LatticeComplex:
    latt_info = propag_i.latt_info
    assert latt_info.Nd == 4 and latt_info.Ns == 4 and latt_info.Nc == 3
    correl = LatticeComplex(latt_info)
    contract.baryon_general_two_point(
        correl.data_ptr,
        propag_i.data_ptr,
        propag_j.data_ptr,
        propag_n.data_ptr,
        contract_type,
        latt_info.volume,
        gamma_ij.index,
        gamma_kl.index,
        project_mn.reshape(-1),
    )
    correl *= gamma_ij.factor * gamma_kl.factor
    return correl


def baryonSequentialTwoPoint(
    propag_i: LatticePropagator,
    propag_j: LatticePropagator,
    propag_n: LatticePropagator,
    contract_type: BaryonContractType,
    sequential_type: BaryonSequentialType,
    gamma_ij: Gamma,
    gamma_kl: Gamma,
    gamma_mn: Union[Gamma, Polarize],
) -> LatticePropagator:
    if isinstance(gamma_mn, Gamma):
        latt_info = propag_i.latt_info
        assert latt_info.Nd == 4 and latt_info.Ns == 4 and latt_info.Nc == 3
        sequential = LatticePropagator(latt_info)
        if sequential_type == BaryonSequentialType.SEQUENTIAL_I:
            propag_i = sequential
        elif sequential_type == BaryonSequentialType.SEQUENTIAL_J:
            propag_j = sequential
        elif sequential_type == BaryonSequentialType.SEQUENTIAL_N:
            propag_n = sequential
        contract.baryon_sequential_two_point(
            propag_i.data_ptr,
            propag_j.data_ptr,
            propag_n.data_ptr,
            contract_type,
            sequential_type,
            latt_info.volume,
            gamma_ij.index,
            gamma_kl.index,
            gamma_mn.index,
        )
        sequential *= gamma_ij.factor * gamma_kl.factor * gamma_mn.factor
        return sequential
    elif isinstance(gamma_mn, Polarize):
        sequential_left = baryonSequentialTwoPoint(
            propag_i, propag_j, propag_n, contract_type, sequential_type, gamma_ij, gamma_kl, gamma_mn.left
        )
        sequential_right = baryonSequentialTwoPoint(
            propag_i, propag_j, propag_n, contract_type, sequential_type, gamma_ij, gamma_kl, gamma_mn.right
        )
        return sequential_left + sequential_right
    else:
        raise getLogger().critical("gamma_mn should be Gamma or Polarize", ValueError)


def baryonTwoPoint_v2(
    propag_i: LatticePropagator,
    propag_j: LatticePropagator,
    propag_m: LatticePropagator,
    contract_type: BaryonContractType,
    gamma_ij: Gamma,
    gamma_kl: Gamma,
    gamma_mn: Union[Gamma, Polarize],
) -> LatticeComplex:
    if isinstance(gamma_mn, Gamma):
        latt_info = propag_i.latt_info
        assert latt_info.Nd == 4 and latt_info.Ns == 4 and latt_info.Nc == 3
        correl = LatticeComplex(latt_info)
        contract.baryon_two_point_v2(
            correl.data_ptr,
            propag_i.data_ptr,
            propag_j.data_ptr,
            propag_m.data_ptr,
            contract_type,
            latt_info.volume,
            gamma_ij.index,
            gamma_kl.index,
            gamma_mn.index,
        )
        correl.data *= gamma_ij.factor * gamma_kl.factor * gamma_mn.factor
        return correl
    elif isinstance(gamma_mn, Polarize):
        correl_left = baryonTwoPoint_v2(propag_i, propag_j, propag_m, contract_type, gamma_ij, gamma_kl, gamma_mn.left)
        correl_right = baryonTwoPoint_v2(
            propag_i, propag_j, propag_m, contract_type, gamma_ij, gamma_kl, gamma_mn.right
        )
        return correl_left + correl_right
    else:
        raise getLogger().critical("gamma_mn should be Gamma or Polarize", ValueError)
