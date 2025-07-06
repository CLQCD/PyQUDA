from typing import Union

from pyquda_comm import getLogger
from pyquda_comm.field import LatticeComplex, MultiLatticeComplex, LatticePropagator
from pyquda_utils.gamma import Gamma, Polarize

from . import _pycontract as contract
from ._pycontract import BaryonContractType


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


def baryonTwoPoint(
    propag_a: LatticePropagator,
    propag_b: LatticePropagator,
    propag_c: LatticePropagator,
    contract_type: BaryonContractType,
    gamma_ab: Gamma,
    gamma_de: Gamma,
    gamma_fc: Union[Gamma, Polarize],
) -> LatticeComplex:
    if isinstance(gamma_fc, Gamma):
        latt_info = propag_a.latt_info
        assert latt_info.Nd == 4 and latt_info.Ns == 4 and latt_info.Nc == 3
        correl = LatticeComplex(latt_info)
        contract.baryon_two_point(
            correl.data_ptr,
            propag_a.data_ptr,
            propag_b.data_ptr,
            propag_c.data_ptr,
            contract_type,
            latt_info.volume,
            gamma_ab.index,
            gamma_de.index,
            gamma_fc.index,
        )
        correl *= gamma_ab.factor * gamma_de.factor * gamma_fc.factor
        return correl
    elif isinstance(gamma_fc, Polarize):
        correl_left = baryonTwoPoint(propag_a, propag_b, propag_c, contract_type, gamma_ab, gamma_de, gamma_fc.left)
        correl_right = baryonTwoPoint(propag_a, propag_b, propag_c, contract_type, gamma_ab, gamma_de, gamma_fc.right)
        return correl_left + correl_right
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
