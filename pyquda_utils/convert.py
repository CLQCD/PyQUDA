from typing import Union

from pyquda import getLogger
from pyquda.field import (
    LatticeLink,
    LatticeFermion,
    LatticePropagator,
    LatticeStaggeredPropagator,
    MultiLatticeFermion,
    MultiLatticeStaggeredFermion,
)


def linkToFermion(link: LatticeLink):
    fermion = LatticeFermion(link.latt_info)
    for color in range(link.latt_info.Nc):
        fermion.data[:, :, :, :, :, color, :] = link.data[:, :, :, :, :, :, color]
    return fermion


def fermionToLink(fermion: LatticeFermion):
    link = LatticeLink(fermion.latt_info)
    for color in range(fermion.latt_info.Nc):
        link.data[:, :, :, :, :, :, color] = fermion.data[:, :, :, :, :, color, :]
    return link


def multiFermionToPropagator(multi_fermion: Union[MultiLatticeFermion, MultiLatticeStaggeredFermion]):
    latt_info = multi_fermion.latt_info
    if isinstance(multi_fermion, MultiLatticeFermion):
        assert multi_fermion.L5 == latt_info.Ns * latt_info.Nc
        return LatticePropagator(
            latt_info,
            multi_fermion.data.reshape(
                latt_info.Ns,
                latt_info.Nc,
                *multi_fermion.lattice_shape,
                latt_info.Ns,
                latt_info.Nc,
            ).transpose(2, 3, 4, 5, 6, 7, 0, 8, 1),
        )
    elif isinstance(multi_fermion, MultiLatticeStaggeredFermion):
        assert multi_fermion.L5 == latt_info.Nc
        return LatticeStaggeredPropagator(
            latt_info,
            multi_fermion.data.reshape(
                latt_info.Nc,
                *multi_fermion.lattice_shape,
                latt_info.Nc,
            ).transpose(1, 2, 3, 4, 5, 6, 0),
        )
    else:
        raise getLogger().critical(
            f"No multiFermionToPropagator implementation for {multi_fermion.__class__.__name__}", NotImplementedError
        )


def propagatorToMultiFermion(propagator: Union[LatticePropagator, LatticeStaggeredPropagator]):
    latt_info = propagator.latt_info
    if isinstance(propagator, LatticePropagator):
        return MultiLatticeFermion(
            latt_info,
            latt_info.Ns * latt_info.Nc,
            propagator.data.transpose(6, 8, 0, 1, 2, 3, 4, 5, 7).reshape(
                latt_info.Ns * latt_info.Nc,
                *propagator.lattice_shape,
                latt_info.Ns,
                latt_info.Nc,
            ),
        )
    elif isinstance(propagator, LatticeStaggeredPropagator):
        return MultiLatticeStaggeredFermion(
            latt_info,
            latt_info.Nc,
            propagator.data.transpose(6, 0, 1, 2, 3, 4, 5).reshape(
                latt_info.Nc,
                *propagator.lattice_shape,
                latt_info.Nc,
            ),
        )
    else:
        raise getLogger().critical(
            f"No propagatorToMultiFermion implementation for {propagator.__class__.__name__}", NotImplementedError
        )
