from math import prod
from typing import Any, List, Literal, Union

import numpy
from numpy.typing import NDArray

from pyquda_comm import getLogger
from pyquda_comm.field import (  # noqa: F401
    LatticeInfo,
    LatticeInt,
    MultiLatticeInt,
    LatticeReal,
    MultiLatticeReal,
    LatticeComplex,
    MultiLatticeComplex,
    LatticeLink,
    LatticeRotation as _LatticeRotation,
    LatticeGauge as _LatticeGauge,
    LatticeMom as _LatticeMom,
    LatticeClover,
    HalfLatticeFermion,
    LatticeFermion,
    MultiHalfLatticeFermion,
    MultiLatticeFermion,
    LatticePropagator,
    HalfLatticeStaggeredFermion,
    LatticeStaggeredFermion,
    MultiHalfLatticeStaggeredFermion,
    MultiLatticeStaggeredFermion,
    LatticeStaggeredPropagator,
)

Nd, Ns, Nc = 4, 4, 3


class _Direction(int):
    def __new__(cls, x: int):
        return int.__new__(cls, x)

    def __neg__(self):
        return _Direction((self + 4) % 8)


X = _Direction(0)
Y = _Direction(1)
Z = _Direction(2)
T = _Direction(3)


def lexico(data: NDArray, axes: List[int], dtype=None):
    assert len(axes) == 5
    shape = data.shape
    Np, Lt, Lz, Ly, Lx = [shape[axis] for axis in axes]
    assert Np == 2
    Lx *= 2
    Npre = prod(shape[: axes[0]])
    Nsuf = prod(shape[axes[-1] + 1 :])
    dtype = data.dtype if dtype is None else dtype
    data_evenodd = data.reshape(Npre, 2, Lt, Lz, Ly, Lx // 2, Nsuf)
    data_lexico = numpy.zeros((Npre, Lt, Lz, Ly, Lx, Nsuf), dtype)
    for t in range(Lt):
        for z in range(Lz):
            for y in range(Ly):
                eo = (t + z + y) % 2
                if eo == 0:
                    data_lexico[:, t, z, y, 0::2] = data_evenodd[:, 0, t, z, y, :]
                    data_lexico[:, t, z, y, 1::2] = data_evenodd[:, 1, t, z, y, :]
                else:
                    data_lexico[:, t, z, y, 1::2] = data_evenodd[:, 0, t, z, y, :]
                    data_lexico[:, t, z, y, 0::2] = data_evenodd[:, 1, t, z, y, :]
    return data_lexico.reshape(*shape[: axes[0]], Lt, Lz, Ly, Lx, *shape[axes[-1] + 1 :])


def evenodd(data: NDArray, axes: List[int], dtype=None):
    assert len(axes) == 4
    shape = data.shape
    Lt, Lz, Ly, Lx = [shape[axis] for axis in axes]
    Npre = prod(shape[: axes[0]])
    Nsuf = prod(shape[axes[-1] + 1 :])
    dtype = data.dtype if dtype is None else dtype
    data_lexico = data.reshape(Npre, Lt, Lz, Ly, Lx, Nsuf)
    data_evenodd = numpy.zeros((Npre, 2, Lt, Lz, Ly, Lx // 2, Nsuf), dtype)
    for t in range(Lt):
        for z in range(Lz):
            for y in range(Ly):
                eo = (t + z + y) % 2
                if eo == 0:
                    data_evenodd[:, 0, t, z, y, :] = data_lexico[:, t, z, y, 0::2]
                    data_evenodd[:, 1, t, z, y, :] = data_lexico[:, t, z, y, 1::2]
                else:
                    data_evenodd[:, 0, t, z, y, :] = data_lexico[:, t, z, y, 1::2]
                    data_evenodd[:, 1, t, z, y, :] = data_lexico[:, t, z, y, 0::2]
    return data_evenodd.reshape(*shape[: axes[0]], 2, Lt, Lz, Ly, Lx // 2, *shape[axes[-1] + 1 :])


def cb2(data: NDArray, axes: List[int], dtype=None):
    getLogger().warning("cb2 is deprecated, use evenodd instead", DeprecationWarning)
    return evenodd(data, axes, dtype)


class LatticeRotation(_LatticeRotation):
    @classmethod
    def loadH5(cls, filename: str, *, check: bool = True) -> "LatticeRotation":
        return super().loadH5(filename, ["R"], check=check)

    def saveH5(self, filename: str, *, annotation: str = "", check: bool = True):
        super().saveH5(filename, ["R"], annotation=annotation, check=check)

    def appendH5(self, filename: str, *, annotation: str = "", check: bool = True, use_fp32: bool = False):
        super().appendH5(filename, ["R"], annotation=annotation, check=check, use_fp32=use_fp32)

    def updateH5(self, filename: str, *, annotation: str = "", check: bool = True):
        super().updateH5(filename, ["R"], annotation=annotation, check=check)

    @property
    def gauge_dirac(self):
        if not hasattr(self, "_gauge_dirac"):
            from pyquda.dirac import GaugeDirac

            self._gauge_dirac = GaugeDirac(self.latt_info)
        return self._gauge_dirac

    def pack(self, x: "LatticeFermion"):
        for color in range(self.latt_info.Nc):
            x.data[:, :, :, :, :, color, :] = self[0].data[:, :, :, :, :, :, color]

    def unpack(self, x: "LatticeFermion"):
        for color in range(self.latt_info.Nc):
            self[0].data[:, :, :, :, :, :, color] = x.data[:, :, :, :, :, color, :]


class LatticeGauge(_LatticeGauge):
    def __init__(self, latt_info: LatticeInfo, L5: Union[int, Any] = Nd, value=None) -> None:
        """`L5` can be `value` here"""
        if not isinstance(L5, int):
            value = L5
            L5 = latt_info.Nd
        super().__init__(latt_info, L5, value)

    @classmethod
    def loadH5(cls, filename: str, *, check: bool = True) -> "LatticeGauge":
        return super().loadH5(filename, ["X", "Y", "Z", "T"], check=check)

    def saveH5(self, filename: str, *, annotation: str = "", check: bool = True, use_fp32: bool = False):
        super().saveH5(filename, ["X", "Y", "Z", "T"], annotation=annotation, check=check, use_fp32=use_fp32)

    def appendH5(self, filename: str, *, annotation: str = "", check: bool = True, use_fp32: bool = False):
        super().appendH5(filename, ["X", "Y", "Z", "T"], annotation=annotation, check=check, use_fp32=use_fp32)

    def updateH5(self, filename: str, *, annotation: str = "", check: bool = True):
        super().updateH5(filename, ["X", "Y", "Z", "T"], annotation=annotation, check=check)

    @property
    def gauge_dirac(self):
        if not hasattr(self, "_gauge_dirac"):
            from pyquda.dirac import GaugeDirac

            self._gauge_dirac = GaugeDirac(self.latt_info)
        return self._gauge_dirac

    @property
    def pure_gauge(self):
        return self.gauge_dirac

    def ensurePureGauge(self):
        pass

    def pack(self, index: int, x: "LatticeFermion"):
        for color in range(self.latt_info.Nc):
            x.data[:, :, :, :, :, color, :] = self[index].data[:, :, :, :, :, :, color]

    def unpack(self, index: int, x: "LatticeFermion"):
        for color in range(self.latt_info.Nc):
            self[index].data[:, :, :, :, :, :, color] = x.data[:, :, :, :, :, color, :]

    def use(self):
        return self.gauge_dirac.useGauge(self)

    def covDev(self, x: "LatticeFermion", covdev_mu: int):
        with self.use() as dirac:
            b = dirac.covDev(x, covdev_mu)
        return b

    def laplace(self, x: "LatticeStaggeredFermion", laplace3D: int):
        with self.use() as dirac:
            b = dirac.laplace(x, laplace3D)
        return b

    def wuppertalSmear(self, x: Union["LatticeFermion", "LatticeStaggeredFermion"], n_steps: int, alpha: float):
        with self.use() as dirac:
            b = dirac.wuppertalSmear(x, n_steps, alpha)
        return b

    def staggeredPhase(self, applied: bool):
        self.gauge_dirac.staggeredPhase(self, applied)

    def projectSU3(self, tol: float):
        self.gauge_dirac.projectSU3(self, tol)

    def path(self, paths: List[List[int]]):
        return self.gauge_dirac.path(self, paths)

    def loop(self, loops: List[List[List[int]]], coeff: List[float]):
        return self.gauge_dirac.loop(self, loops, coeff)

    def loopTrace(self, loops: List[List[int]]):
        with self.use() as dirac:
            traces = dirac.loopTrace(loops)
        return traces

    def apeSmear(
        self,
        n_steps: int,
        alpha: float,
        dir_ignore: int,
        compute_plaquette: bool = False,
        compute_qcharge: bool = False,
    ):
        with self.use() as dirac:
            dirac.apeSmear(n_steps, alpha, dir_ignore, compute_plaquette, compute_qcharge)
            dirac.saveSmearedGauge(self)
            dirac.freeSmearedGauge()

    def stoutSmear(
        self,
        n_steps: int,
        rho: float,
        dir_ignore: int,
        compute_plaquette: bool = False,
        compute_qcharge: bool = False,
    ):
        with self.use() as dirac:
            dirac.stoutSmear(n_steps, rho, dir_ignore, compute_plaquette, compute_qcharge)
            dirac.saveSmearedGauge(self)
            dirac.freeSmearedGauge()

    def hypSmear(
        self,
        n_steps: int,
        alpha1: float,
        alpha2: float,
        alpha3: float,
        dir_ignore: int,
        compute_plaquette: bool = False,
        compute_qcharge: bool = False,
    ):
        with self.use() as dirac:
            dirac.hypSmear(n_steps, alpha1, alpha2, alpha3, dir_ignore, compute_plaquette, compute_qcharge)
            dirac.saveSmearedGauge(self)
            dirac.freeSmearedGauge()

    def wilsonFlow(
        self,
        n_steps: int,
        epsilon: float,
        compute_plaquette: bool = False,
        compute_qcharge: bool = True,
    ):
        with self.use() as dirac:
            dirac.wilsonFlow(1, epsilon, 0, False, compute_plaquette, compute_qcharge)
            energy = [dirac.obs_param.energy]
            for step in range(1, n_steps):
                dirac.wilsonFlow(1, epsilon, step * epsilon, True, compute_plaquette, compute_qcharge)
                energy.append(dirac.obs_param.energy)
            energy.append(dirac.energy())
            dirac.saveSmearedGauge(self)
            dirac.freeSmearedGauge()
        return energy

    def wilsonFlowScale(
        self,
        max_steps: int,
        epsilon: float,
        compute_plaquette: bool = False,
        compute_qcharge: bool = True,
    ):
        with self.use() as dirac:
            dirac.wilsonFlow(1, epsilon, 0, False, compute_plaquette, compute_qcharge)
            t2E, tdt2E = 0, 0
            t0, w0 = 0, 0
            for step in range(1, max_steps + 1):
                if t2E >= 0.3 and tdt2E >= 0.3:
                    break
                dirac.wilsonFlow(1, epsilon, step * epsilon, True, compute_plaquette, compute_qcharge)
                t2E_old, t2E = t2E, (step * epsilon) ** 2 * dirac.obs_param.energy[0]
                tdt2E_old, tdt2E = tdt2E, (step - 0.5) * (t2E - t2E_old)
                if t0 == 0 and t2E >= 0.3:
                    t0 = (step - (t2E - 0.3) / (t2E - t2E_old)) * epsilon
                if w0 == 0 and tdt2E >= 0.3:
                    w0 = ((step - 0.5 - (tdt2E - 0.3) / (tdt2E - tdt2E_old)) * epsilon) ** 0.5
                getLogger().info(f"t2E({step * epsilon:.3f})={t2E}, tdt2E({(step - 0.5) * epsilon:.3f})={tdt2E}")
            else:
                getLogger().error(
                    f"Wilson flow scale doesn't exceed 0.3 at max_steps*epsilon={max_steps * epsilon}", RuntimeError
                )
            dirac.freeSmearedGauge()
        return t0, w0

    def symanzikFlow(
        self,
        n_steps: int,
        epsilon: float,
        compute_plaquette: bool = False,
        compute_qcharge: bool = True,
    ):
        with self.use() as dirac:
            dirac.symanzikFlow(1, epsilon, 0, False, compute_plaquette, compute_qcharge)
            energy = [dirac.obs_param.energy]
            for step in range(1, n_steps):
                dirac.symanzikFlow(1, epsilon, step * epsilon, True, compute_plaquette, compute_qcharge)
                energy.append(dirac.obs_param.energy)
            energy.append(dirac.energy())
            dirac.saveSmearedGauge(self)
            dirac.freeSmearedGauge()
        return energy

    def symanzikFlowScale(
        self,
        max_steps: int,
        epsilon: float,
        compute_plaquette: bool = False,
        compute_qcharge: bool = True,
    ):
        with self.use() as dirac:
            dirac.symanzikFlow(1, epsilon, 0, False, compute_plaquette, compute_qcharge)
            t2E, tdt2E = 0, 0
            t0, w0 = 0, 0
            for step in range(1, max_steps + 1):
                if t2E >= 0.3 and tdt2E >= 0.3:
                    break
                dirac.symanzikFlow(1, epsilon, step * epsilon, True, compute_plaquette, compute_qcharge)
                t2E_old, t2E = t2E, (step * epsilon) ** 2 * dirac.obs_param.energy[0]
                tdt2E_old, tdt2E = tdt2E, (step - 0.5) * (t2E - t2E_old)
                if t0 == 0 and t2E >= 0.3:
                    t0 = (step - (t2E - 0.3) / (t2E - t2E_old)) * epsilon
                if w0 == 0 and tdt2E >= 0.3:
                    w0 = ((step - 0.5 - (tdt2E - 0.3) / (tdt2E - tdt2E_old)) * epsilon) ** 0.5
                getLogger().info(f"t2E({step * epsilon:.3f})={t2E}, tdt2E({(step - 0.5) * epsilon:.3f})={tdt2E}")
            else:
                getLogger().error(
                    f"Symanzik flow scale doesn't exceed 0.3 at max_steps*epsilon={max_steps * epsilon}", RuntimeError
                )
            dirac.freeSmearedGauge()
        return t0, w0

    def apeSmearChroma(
        self,
        n_steps: int,
        factor: float,
        dir_ignore: int,
        compute_plaquette: bool = False,
        compute_qcharge: bool = False,
    ):
        """A variant of apeSmear() to match Chroma"""
        dimAPE = 3 if dir_ignore >= 0 and dir_ignore <= 3 else 4
        self.apeSmear(n_steps, (dimAPE - 1) / (dimAPE - 1 + factor / 2), dir_ignore, compute_plaquette, compute_qcharge)

    def wilsonFlowChroma(
        self, n_steps: int, time: float, compute_plaquette: bool = False, compute_qcharge: bool = True
    ):
        """Use total time instead of epsilon"""
        return self.wilsonFlow(n_steps, time / n_steps, compute_plaquette, compute_qcharge)

    def symanzikFlowChroma(
        self, n_steps: int, time: float, compute_plaquette: bool = False, compute_qcharge: bool = True
    ):
        """Use total time instead of epsilon"""
        return self.symanzikFlow(n_steps, time / n_steps, compute_plaquette, compute_qcharge)

    def plaquette(self):
        with self.use() as dirac:
            plaquette = dirac.plaquette()
        return plaquette

    def polyakovLoop(self):
        with self.use() as dirac:
            polyakovLoop = dirac.polyakovLoop()
        return polyakovLoop

    def energy(self):
        with self.use() as dirac:
            energy = dirac.energy()
        return energy

    def qcharge(self):
        with self.use() as dirac:
            qcharge = dirac.qcharge()
        return qcharge

    def qchargeDensity(self):
        with self.use() as dirac:
            qcharge_density = dirac.qchargeDensity()
        return qcharge_density

    def gauss(self, seed: int, sigma: float):
        """
        Generate Gaussian distributed fields and store in the
        resident gauge field.  We create a Gaussian-distributed su(n)
        field and exponentiate it, e.g., U = exp(sigma * H), where H is
        the distributed su(n) field and sigma is the width of the
        distribution (sigma = 0 results in a free field, and sigma = 1 has
        maximum disorder).

        Parameters
        ----------
        seed: int
            The seed used for the RNG
        sigma: float
            Width of Gaussian distrubution
        """
        with self.use() as dirac:
            dirac.gaussGauge(seed, sigma)
            dirac.saveGauge(self)

    def fixingOVR(
        self,
        gauge_dir: Literal[3, 4],
        Nsteps: int,
        verbose_interval: int,
        relax_boost: float,
        tolerance: float,
        reunit_interval: int,
        stopWtheta: int,
    ):
        """
        Gauge fixing with overrelaxation with support for single and multi GPU.

        Parameters
        ----------
        gauge_dir: {3, 4}
            3 for Coulomb gauge fixing, 4 for Landau gauge fixing
        Nsteps: int
            maximum number of steps to perform gauge fixing
        verbose_interval: int
            print gauge fixing info when iteration count is a multiple of this
        relax_boost: float
            gauge fixing parameter of the overrelaxation method, most common value is 1.5 or 1.7.
        tolerance: float
            torelance value to stop the method, if this value is zero then the method stops when
            iteration reachs the maximum number of steps defined by Nsteps
        reunit_interval: int
            reunitarize gauge field when iteration count is a multiple of this
        stopWtheta: int
            0 for MILC criterion and 1 to use the theta value
        """
        self.gauge_dirac.fixingOVR(
            self, gauge_dir, Nsteps, verbose_interval, relax_boost, tolerance, reunit_interval, stopWtheta
        )

    def fixingFFT(
        self,
        gauge_dir: Literal[3, 4],
        Nsteps: int,
        verbose_interval: int,
        alpha: float,
        autotune: int,
        tolerance: float,
        stopWtheta: int,
    ):
        """
        Gauge fixing with Steepest descent method with FFTs with support for single GPU only.

        Parameters
        ----------
        gauge_dir: {3, 4}
            3 for Coulomb gauge fixing, 4 for Landau gauge fixing
        Nsteps: int
            maximum number of steps to perform gauge fixing
        verbose_interval: int
            print gauge fixing info when iteration count is a multiple of this
        alpha: float
            gauge fixing parameter of the method, most common value is 0.08
        autotune: int
            1 to autotune the method, i.e., if the Fg inverts its tendency we decrease the alpha value
        tolerance: float
            torelance value to stop the method, if this value is zero then the method stops when
            iteration reachs the maximum number of steps defined by Nsteps
        stopWtheta: int
            0 for MILC criterion and 1 to use the theta value
        """
        self.gauge_dirac.fixingFFT(self, gauge_dir, Nsteps, verbose_interval, alpha, autotune, tolerance, stopWtheta)


class LatticeMom(_LatticeMom):
    def __init__(self, latt_info: LatticeInfo, L5: Union[int, Any] = 4, value=None) -> None:
        """`L5` can be `value` here"""
        if not isinstance(L5, int):
            value = L5
            L5 = latt_info.Nd
        super().__init__(latt_info, L5, value)

    @classmethod
    def loadH5(cls, filename: str, *, check: bool = True) -> "LatticeMom":
        return super().loadH5(filename, ["X", "Y", "Z", "T"], check=check)

    def saveH5(self, filename: str, *, annotation: str = "", check: bool = True, use_fp32: bool = False):
        super().saveH5(filename, ["X", "Y", "Z", "T"], annotation=annotation, check=check, use_fp32=use_fp32)

    def appendH5(self, filename: str, *, annotation: str = "", check: bool = True, use_fp32: bool = False):
        super().appendH5(filename, ["X", "Y", "Z", "T"], annotation=annotation, check=check, use_fp32=use_fp32)

    def updateH5(self, filename: str, *, annotation: str = "", check: bool = True):
        super().updateH5(filename, ["X", "Y", "Z", "T"], annotation=annotation, check=check)

    @property
    def gauge_dirac(self):
        if not hasattr(self, "_gauge_dirac"):
            from pyquda.dirac import GaugeDirac

            self._gauge_dirac = GaugeDirac(self.latt_info)
        return self._gauge_dirac

    def gauss(self, seed: int, sigma: float):
        self.gauge_dirac.loadMom(self)
        self._gauge_dirac.gaussMom(seed, sigma)
        self._gauge_dirac.saveFreeMom(self)
