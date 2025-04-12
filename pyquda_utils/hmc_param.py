from math import log
import json
from os import environ, path
from typing import Literal, Sequence, Tuple

from gmpy2 import mpfr

from pyquda_comm import getMPIComm, getMPIRank
from pyquda.field import X, Y, Z, T
from pyquda.action.abstract import LoopParam, RationalParam
from .alg_remez import AlgRemez


class ScientificNotation(float):
    def __repr__(self):
        return f"{self:.16e}"


json.encoder.float = ScientificNotation


def wilsonGaugeLoopParam():
    beta_0 = 1
    return LoopParam(
        path=[
            [X, Y, -X, -Y],
            [X, Z, -X, -Z],
            [X, T, -X, -T],
            [Y, Z, -Y, -Z],
            [Y, T, -Y, -T],
            [Z, T, -Z, -T],
        ],
        coeff=[beta_0 for _ in range(6)],
    )


def iwasakiGaugeLoopParam():
    beta_0 = 1 - 8 * (-0.331)
    beta_1 = -0.331
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
        ],
        coeff=[beta_0 for _ in range(6)] + [beta_1 for _ in range(12)],
    )


def symanzikTreeGaugeLoopParam(u_0: float):
    beta_0 = 1
    beta_1 = -beta_0 / (20 * u_0**2)
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
        ],
        coeff=[beta_0 for _ in range(6)] + [beta_1 for _ in range(12)],
    )


def symanzikOneLoopGaugeLoopParam(
    u_0: float, fermion_type: Literal["quenched", "asqtad", "hisq"] = "quenched", n_flavor: int = None
):
    """
    Quenched: https://arxiv.org/pdf/hep-lat/9507010 Eqs. (6), (7), (8), (9)

    Asqtad: https://arxiv.org/pdf/0705.4660 Eq. (41)

    HISQ: https://arxiv.org/pdf/0812.0503 Eq. (36)
    """
    beta_0 = 1
    alpha_s = -4 * log(u_0) / 3.06839
    if fermion_type == "quenched":
        beta_1 = -beta_0 / (20 * u_0**2) * (1 + 0.4805 * alpha_s)
        beta_2 = -beta_0 / u_0**2 * 0.03325 * alpha_s
    elif fermion_type == "asqtad":
        beta_1 = -beta_0 / (20 * u_0**2) * (1 + (0.4805 - 0.3637 * n_flavor) * alpha_s)
        beta_2 = -beta_0 / u_0**2 * (0.03325 - 0.009 * n_flavor) * alpha_s
    elif fermion_type == "hisq":
        beta_1 = -beta_0 / (20 * u_0**2) * (1 + (0.4805 - 0.899 * n_flavor) * alpha_s)
        beta_2 = -beta_0 / u_0**2 * (0.03325 - 0.0121 * n_flavor) * alpha_s
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


def _loadCache():
    cache = {}
    if "QUDA_RESOURCE_PATH" in environ and environ["QUDA_RESOURCE_PATH"] is not None:
        cache_file = path.join(path.realpath(environ["QUDA_RESOURCE_PATH"]), "hmc_param.json")
        if path.exists(cache_file):
            with open(cache_file, "r") as fp:
                for key, val in json.load(fp).items():
                    cache[key] = RationalParam(*val)
    return cache


def _dumpCache(cache):
    if "QUDA_RESOURCE_PATH" in environ and environ["QUDA_RESOURCE_PATH"] is not None:
        cache_file = path.join(path.realpath(environ["QUDA_RESOURCE_PATH"]), "hmc_param.json")
        with open(cache_file, "w") as fp:
            json.dump(cache, fp)


class _WilsonMD:
    def __init__(self, nf):
        self.nf = nf

    def __str__(self):
        ret = f"x^({self.nf}/2)"
        return f"Approximating the function {ret}"

    def __call__(self, x):
        ret = x ** (mpfr(self.nf) / 2)
        return ret


class _WilsonFA:
    def __init__(self, nf):
        self.nf = nf

    def __str__(self):
        ret = f"x^({self.nf}/4)"
        return f"Approximating the function {ret}"

    def __call__(self, x):
        ret = x ** (mpfr(self.nf) / 4)
        return ret


def wilsonFermionRationalParam(
    num_flavor: int,
    molecular_dynamics_degree: int,
    fermion_action_degree: int,
    lower_bound: float,
    upper_bound: float,
    precision: int,
):
    if getMPIRank() == 0:
        key = str(
            (
                num_flavor,
                molecular_dynamics_degree,
                fermion_action_degree,
                lower_bound,
                upper_bound,
                precision,
            )
        )
        cache = _loadCache()
        if key in cache:
            rational_param = cache[key]
        else:
            if num_flavor == 2:
                rational_param = RationalParam()
            else:
                remez = AlgRemez(lower_bound, upper_bound, precision)
                remez.generateApprox(molecular_dynamics_degree, molecular_dynamics_degree, _WilsonMD(num_flavor))
                molecular_dynamics = remez.getIPFE()
                remez.generateApprox(fermion_action_degree, fermion_action_degree, _WilsonFA(num_flavor))
                pseudo_fermion = remez.getPFE()
                fermion_action = remez.getIPFE()
                rational_param = RationalParam(*molecular_dynamics, *pseudo_fermion, *fermion_action)
            cache[key] = rational_param
        _dumpCache(cache)
    else:
        rational_param = None
    return getMPIComm().bcast(rational_param)


class _StaggeredMD:
    def __init__(self, m: Tuple[int], nf: Tuple[int]):
        self.m = m
        self.nf = nf

    def __str__(self):
        ret = ""
        for m, nf in zip(self.m, self.nf):
            ret += f"(x+4*{m}^2+x)^({nf}/4)*"
        return f"Approximating the function {ret[:-1]}"

    def __call__(self, x):
        ret = mpfr(1)
        for m, nf in zip(self.m, self.nf):
            ret *= (4 * mpfr(m) ** 2 + x) ** (mpfr(nf) / 4)
        return ret


class _StaggeredFA:
    def __init__(self, m: Tuple[int], nf: Tuple[int]):
        self.m = m
        self.nf = nf

    def __str__(self):
        ret = ""
        for m, nf in zip(self.m, self.nf):
            ret += f"(x+4*{m}^2)^({nf}/8)*"
        return f"Approximating the function {ret[:-1]}"

    def __call__(self, x):
        ret = mpfr(1)
        for m, nf in zip(self.m, self.nf):
            ret *= (4 * mpfr(m) ** 2 + x) ** (mpfr(nf) / 8)
        return ret


def staggeredFermionRationalParam(
    mass: Sequence[float],
    num_flavor: Sequence[int],
    molecular_dynamics_degree: int,
    fermion_action_degree: int,
    lower_bound: float,
    upper_bound: float,
    precision: int,
):
    if getMPIRank() == 0:
        mass = tuple(mass) if isinstance(mass, Sequence) else (mass,)
        num_flavor = tuple(num_flavor) if isinstance(num_flavor, Sequence) else (num_flavor,)
        key = str(
            (
                mass,
                num_flavor,
                molecular_dynamics_degree,
                fermion_action_degree,
                lower_bound,
                upper_bound,
                precision,
            )
        )
        cache = _loadCache()
        if key in cache:
            rational_param = cache[key]
        else:
            remez = AlgRemez(lower_bound, upper_bound, precision)
            remez.generateApprox(molecular_dynamics_degree, molecular_dynamics_degree, _StaggeredMD(mass, num_flavor))
            molecular_dynamics = remez.getIPFE()
            remez.generateApprox(fermion_action_degree, fermion_action_degree, _StaggeredFA(mass, num_flavor))
            pseudo_fermion = remez.getPFE()
            fermion_action = remez.getIPFE()
            rational_param = RationalParam(*molecular_dynamics, *pseudo_fermion, *fermion_action)
            cache[key] = rational_param
            _dumpCache(cache)
    else:
        rational_param = None
    return getMPIComm().bcast(rational_param)
