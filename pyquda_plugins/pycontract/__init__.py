from typing import Union

from pyquda_comm import getLogger
from pyquda_comm.field import LatticeComplex, LatticePropagator
from pyquda_utils.gamma import Gamma, Polarize

from . import _pycontract as contract
from ._pycontract import BaryonContractType


def baryonTwoPoint(
    propag_i: LatticePropagator,
    propag_j: LatticePropagator,
    propag_m: LatticePropagator,
    contract_type: BaryonContractType,
    gamma_ij: Gamma,
    gamma_kl: Gamma,
    gamma_mn: Union[Gamma, Polarize],
):
    latt_info = propag_i.latt_info
    if isinstance(gamma_mn, Gamma):
        correl = LatticeComplex(latt_info)
        contract.baryon_two_point(
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
        correl.data *= gamma_ij.sign * gamma_kl.sign * gamma_mn.sign
        return correl
    elif isinstance(gamma_mn, Polarize):
        correl_left = baryonTwoPoint(propag_i, propag_j, propag_m, contract_type, gamma_ij, gamma_kl, gamma_mn.left)
        correl_right = baryonTwoPoint(propag_i, propag_j, propag_m, contract_type, gamma_ij, gamma_kl, gamma_mn.right)
        return correl_left + correl_right
    else:
        raise getLogger().critical("gamma_mn should be Gamma or Polarize", ValueError)
