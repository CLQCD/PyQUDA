from math import log
from typing import Dict, Tuple

from pyquda.field import X, Y, Z, T
from pyquda.action.abstract import LoopParam, RationalParam


def symanzik_1loop_hisq(u_0: float, n_flavor: int):
    beta_0 = 1
    beta_1 = -beta_0 / (20 * u_0 * u_0) * (1.00 - (0.6264 - 1.1746 * n_flavor) * log(u_0))
    beta_2 = beta_0 / (u_0 * u_0) * (0.0433 - 0.0156 * n_flavor) * log(u_0)
    return LoopParam(
        path=[
            [X, Y, -X, -Y],
            [X, Z, -X, -Z],
            [X, T, -X, -T],
            [Y, Z, -Y, -Z],
            [Y, T, -Y, -T],
            [Z, T, -Z, -T],
            [X, X, Y, -X, -X, -Y],
            [X, X, Z, -X, -X, -Z],
            [X, X, T, -X, -X, -T],
            [Y, Y, X, -Y, -Y, -X],
            [Y, Y, Z, -Y, -Y, -Z],
            [Y, Y, T, -Y, -Y, -T],
            [Z, Z, X, -Z, -Z, -X],
            [Z, Z, Y, -Z, -Z, -Y],
            [Z, Z, T, -Z, -Z, -T],
            [T, T, X, -T, -T, -X],
            [T, T, Y, -T, -T, -Y],
            [T, T, Z, -T, -T, -Z],
            [X, Y, Z, -X, -Y, -Z],
            [X, Y, -Z, -X, -Y, Z],
            [X, -Y, Z, -X, Y, -Z],
            [X, -Y, -Z, -X, Y, Z],
            [X, Y, T, -X, -Y, -T],
            [X, Y, -T, -X, -Y, T],
            [X, -Y, T, -X, Y, -T],
            [X, -Y, -T, -X, Y, T],
            [X, Z, T, -X, -Z, -T],
            [X, Z, -T, -X, -Z, T],
            [X, -Z, T, -X, Z, -T],
            [X, -Z, -T, -X, Z, T],
            [Y, Z, T, -Y, -Z, -T],
            [Y, Z, -T, -Y, -Z, T],
            [Y, -Z, T, -Y, Z, -T],
            [Y, -Z, -T, -Y, Z, T],
        ],
        coeff=[beta_0 for _ in range(6)] + [beta_1 for _ in range(12)] + [beta_2 for _ in range(16)],
    )


def poly4_rhmc_param(filename: str):
    rhmc_params: Dict[Tuple[Tuple[float, ...], Tuple[int, ...]], RationalParam] = {}
    with open(filename, "r") as f:
        lines = [line.strip().split(" ") for line in f.readlines() if not line.startswith("#") and not line == "\n"]
    n_pseudo = int(lines[0][1])
    index = 1
    for _ in range(n_pseudo):
        while not lines[index][0] == "y_MD":
            index += 1
        index += 3
        order = int(lines[index][1])
        index += 1
        norm_molecular_dynamics = float(lines[index][1])
        index += 1
        residue_molecular_dynamics = [float(lines[index + i][1]) for i in range(order)]
        index += order + 1
        offset_molecular_dynamics = [float(lines[index + i][1]) for i in range(order)]
        index += order

        n_flavor = tuple(int(lines[index][1 + i]) for i in range(4) if lines[index][1 + i] != "0")
        mass = tuple(float(lines[index + 2][1 + i]) for i in range(4) if lines[index][1 + i] != "0")
        index += 3
        order = int(lines[index][1])
        index += 1
        norm_pseudo_fermion = float(lines[index][1])
        index += 1
        residue_pseudo_fermion = [float(lines[index + i][1]) for i in range(order)]
        index += order + 1
        offset_pseudo_fermion = [float(lines[index + i][1]) for i in range(order)]
        index += order

        index += 3
        order = int(lines[index][1])
        index += 1
        norm_fermion_action = float(lines[index][1])
        index += 1
        residue_fermion_action = [float(lines[index + i][1]) for i in range(order)]
        index += order + 1
        offset_fermion_action = [float(lines[index + i][1]) for i in range(order)]
        index += order

        key = (mass, n_flavor)
        if key not in rhmc_params:
            rhmc_params[key] = RationalParam(
                norm_molecular_dynamics,
                residue_molecular_dynamics,
                offset_molecular_dynamics,
                norm_pseudo_fermion,
                residue_pseudo_fermion,
                offset_pseudo_fermion,
                norm_fermion_action,
                residue_fermion_action,
                offset_fermion_action,
            )
        else:
            assert rhmc_params[key] == RationalParam(
                norm_molecular_dynamics,
                residue_molecular_dynamics,
                offset_molecular_dynamics,
                norm_pseudo_fermion,
                residue_pseudo_fermion,
                offset_pseudo_fermion,
                norm_fermion_action,
                residue_fermion_action,
                offset_fermion_action,
            )

    return rhmc_params
