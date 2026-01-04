from typing import Sequence, Union, overload

from pyquda_comm.array import arrayDevice
from pyquda import getLogger
from pyquda.field import (
    LatticeInt,
    MultiLatticeInt,
    LatticeReal,
    MultiLatticeReal,
    LatticeComplex,
    MultiLatticeComplex,
    LatticeLink,
    LatticeFermion,
    MultiLatticeFermion,
    LatticePropagator,
    LatticeStaggeredFermion,
    MultiLatticeStaggeredFermion,
    LatticeStaggeredPropagator,
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


@overload
def multiFermionToPropagator(multi_fermion: MultiLatticeFermion) -> LatticePropagator: ...
@overload
def multiFermionToPropagator(multi_fermion: MultiLatticeStaggeredFermion) -> LatticeStaggeredPropagator: ...


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


@overload
def propagatorToMultiFermion(propagator: LatticePropagator) -> MultiLatticeFermion: ...
@overload
def propagatorToMultiFermion(propagator: LatticeStaggeredPropagator) -> MultiLatticeStaggeredFermion: ...


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


@overload
def multiField(fields: Sequence[LatticeInt]) -> MultiLatticeInt: ...
@overload
def multiField(fields: Sequence[LatticeReal]) -> MultiLatticeReal: ...
@overload
def multiField(fields: Sequence[LatticeComplex]) -> MultiLatticeComplex: ...
@overload
def multiField(fields: Sequence[LatticeFermion]) -> MultiLatticeFermion: ...
@overload
def multiField(fields: Sequence[LatticeStaggeredFermion]) -> MultiLatticeStaggeredFermion: ...


def multiField(
    fields: Sequence[Union[LatticeInt, LatticeReal, LatticeComplex, LatticeFermion, LatticeStaggeredFermion]],
) -> Union[MultiLatticeInt, MultiLatticeReal, MultiLatticeComplex, MultiLatticeFermion, MultiLatticeStaggeredFermion]:
    field_0 = fields[0]
    if isinstance(field_0, LatticeInt):
        return MultiLatticeInt(
            field_0.latt_info, len(fields), arrayDevice([field.data for field in fields], field_0.location)
        )
    elif isinstance(field_0, LatticeReal):
        return MultiLatticeReal(
            field_0.latt_info, len(fields), arrayDevice([field.data for field in fields], field_0.location)
        )
    elif isinstance(field_0, LatticeComplex):
        return MultiLatticeComplex(
            field_0.latt_info, len(fields), arrayDevice([field.data for field in fields], field_0.location)
        )
    elif isinstance(field_0, LatticeFermion):
        return MultiLatticeFermion(
            field_0.latt_info, len(fields), arrayDevice([field.data for field in fields], field_0.location)
        )
    elif isinstance(field_0, LatticeStaggeredFermion):
        return MultiLatticeStaggeredFermion(
            field_0.latt_info, len(fields), arrayDevice([field.data for field in fields], field_0.location)
        )
    else:
        getLogger().critical(f"Unknown field type {type(field_0)}", ValueError)
