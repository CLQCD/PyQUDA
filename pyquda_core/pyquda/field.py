from typing import Any, List, Literal, Union

import numpy

from pyquda_comm import getLogger
from pyquda_comm.field import (  # noqa: F401
    lexico,
    evenodd,
    LatticeInfo,
    ParityField,
    FullField,
    MultiField,
    LatticeInt,
    LatticeReal,
    LatticeComplex,
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


def cb2(data: numpy.ndarray, axes: List[int], dtype=None):
    getLogger().warning("cb2 is deprecated, use evenodd instead", DeprecationWarning)
    return evenodd(data, axes, dtype)


class LatticeRotation(_LatticeRotation):
    @classmethod
    def load(cls, filename: str, *, check: bool = True) -> "LatticeRotation":
        return super().load(filename, ["R"], check=check)

    def save(self, filename: str, *, annotation: str = "", check: bool = True):
        super().save(filename, ["R"], annotation=annotation, check=check)

    def append(self, filename: str, *, annotation: str = "", check: bool = True, use_fp32: bool = False):
        super().append(filename, ["R"], annotation=annotation, check=check, use_fp32=use_fp32)

    def update(self, filename: str, *, annotation: str = "", check: bool = True):
        super().update(filename, ["R"], annotation=annotation, check=check)

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
    def load(cls, filename: str, *, check: bool = True) -> "LatticeGauge":
        return super().load(filename, ["X", "Y", "Z", "T"], check=check)

    def save(self, filename: str, *, annotation: str = "", check: bool = True, use_fp32: bool = False):
        super().save(filename, ["X", "Y", "Z", "T"], annotation=annotation, check=check, use_fp32=use_fp32)

    def append(self, filename: str, *, annotation: str = "", check: bool = True, use_fp32: bool = False):
        super().append(filename, ["X", "Y", "Z", "T"], annotation=annotation, check=check, use_fp32=use_fp32)

    def update(self, filename: str, *, annotation: str = "", check: bool = True):
        super().update(filename, ["X", "Y", "Z", "T"], annotation=annotation, check=check)

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

    def covDev(self, x: "LatticeFermion", covdev_mu: int):
        self.gauge_dirac.loadGauge(self)
        b = self._gauge_dirac.covDev(x, covdev_mu)
        self._gauge_dirac.freeGauge()
        return b

    def laplace(self, x: "LatticeStaggeredFermion", laplace3D: int):
        self.gauge_dirac.loadGauge(self)
        b = self._gauge_dirac.laplace(x, laplace3D)
        self._gauge_dirac.freeGauge()
        return b

    def wuppertalSmear(self, x: Union["LatticeFermion", "LatticeStaggeredFermion"], n_steps: int, alpha: float):
        self.gauge_dirac.loadGauge(self)
        b = self._gauge_dirac.wuppertalSmear(x, n_steps, alpha)
        self._gauge_dirac.freeGauge()
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
        self.gauge_dirac.loadGauge(self)
        traces = self._gauge_dirac.loopTrace(loops)
        self._gauge_dirac.freeGauge()
        return traces

    def apeSmear(
        self,
        n_steps: int,
        alpha: float,
        dir_ignore: int,
        compute_plaquette: bool = False,
        compute_qcharge: bool = False,
    ):
        self.gauge_dirac.loadGauge(self)
        self._gauge_dirac.apeSmear(n_steps, alpha, dir_ignore, compute_plaquette, compute_qcharge)
        self._gauge_dirac.saveSmearedGauge(self)
        self._gauge_dirac.freeGauge()
        self._gauge_dirac.freeSmearedGauge()

    def stoutSmear(
        self,
        n_steps: int,
        rho: float,
        dir_ignore: int,
        compute_plaquette: bool = False,
        compute_qcharge: bool = False,
    ):
        self.gauge_dirac.loadGauge(self)
        self._gauge_dirac.stoutSmear(n_steps, rho, dir_ignore, compute_plaquette, compute_qcharge)
        self._gauge_dirac.saveSmearedGauge(self)
        self._gauge_dirac.freeGauge()
        self._gauge_dirac.freeSmearedGauge()

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
        self.gauge_dirac.loadGauge(self)
        self._gauge_dirac.hypSmear(n_steps, alpha1, alpha2, alpha3, dir_ignore, compute_plaquette, compute_qcharge)
        self._gauge_dirac.saveSmearedGauge(self)
        self._gauge_dirac.freeGauge()
        self._gauge_dirac.freeSmearedGauge()

    def wilsonFlow(
        self,
        n_steps: int,
        epsilon: float,
        compute_plaquette: bool = False,
        compute_qcharge: bool = True,
    ):
        self.gauge_dirac.loadGauge(self)
        self._gauge_dirac.wilsonFlow(1, epsilon, 0, False, compute_plaquette, compute_qcharge)
        energy = [self._gauge_dirac.obs_param.energy]
        for step in range(1, n_steps):
            self._gauge_dirac.wilsonFlow(1, epsilon, step * epsilon, True, compute_plaquette, compute_qcharge)
            energy.append(self._gauge_dirac.obs_param.energy)
        self._gauge_dirac.saveSmearedGauge(self)  # Save before the last step
        self._gauge_dirac.wilsonFlow(1, epsilon, n_steps * epsilon, True, compute_plaquette, compute_qcharge)
        energy.append(self._gauge_dirac.obs_param.energy)
        self._gauge_dirac.freeGauge()
        self._gauge_dirac.freeSmearedGauge()
        return energy

    def wilsonFlowScale(
        self,
        max_steps: int,
        epsilon: float,
        compute_plaquette: bool = False,
        compute_qcharge: bool = True,
    ):
        self.gauge_dirac.loadGauge(self)
        self._gauge_dirac.wilsonFlow(1, epsilon, 0, False, compute_plaquette, compute_qcharge)
        t2E, tdt2E = 0, 0
        t0, w0 = 0, 0
        for step in range(1, max_steps + 1):
            if t2E >= 0.3 and tdt2E >= 0.3:
                break
            self._gauge_dirac.wilsonFlow(1, epsilon, step * epsilon, True, compute_plaquette, compute_qcharge)
            t2E_old, t2E = t2E, (step * epsilon) ** 2 * self._gauge_dirac.obs_param.energy[0]
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
        self._gauge_dirac.freeGauge()
        self._gauge_dirac.freeSmearedGauge()
        return t0, w0

    def symanzikFlow(
        self,
        n_steps: int,
        epsilon: float,
        compute_plaquette: bool = False,
        compute_qcharge: bool = True,
    ):
        self.gauge_dirac.loadGauge(self)
        self._gauge_dirac.symanzikFlow(1, epsilon, 0, False, compute_plaquette, compute_qcharge)
        energy = [self._gauge_dirac.obs_param.energy]
        for step in range(1, n_steps):
            self._gauge_dirac.symanzikFlow(1, epsilon, step * epsilon, True, compute_plaquette, compute_qcharge)
            energy.append(self._gauge_dirac.obs_param.energy)
        self._gauge_dirac.saveSmearedGauge(self)  # Save before the last step
        self._gauge_dirac.symanzikFlow(1, epsilon, n_steps * epsilon, True, compute_plaquette, compute_qcharge)
        energy.append(self._gauge_dirac.obs_param.energy)
        self._gauge_dirac.freeGauge()
        self._gauge_dirac.freeSmearedGauge()
        return energy

    def symanzikFlowScale(
        self,
        max_steps: int,
        epsilon: float,
        compute_plaquette: bool = False,
        compute_qcharge: bool = True,
    ):
        self.gauge_dirac.loadGauge(self)
        self._gauge_dirac.symanzikFlow(1, epsilon, 0, False, compute_plaquette, compute_qcharge)
        t2E, tdt2E = 0, 0
        t0, w0 = 0, 0
        for step in range(1, max_steps + 1):
            if t2E >= 0.3 and tdt2E >= 0.3:
                break
            self._gauge_dirac.symanzikFlow(1, epsilon, step * epsilon, True, compute_plaquette, compute_qcharge)
            t2E_old, t2E = t2E, (step * epsilon) ** 2 * self._gauge_dirac.obs_param.energy[0]
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
        self._gauge_dirac.freeGauge()
        self._gauge_dirac.freeSmearedGauge()
        return t0, w0

    def smearAPE(
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

    def smearSTOUT(
        self,
        n_steps: int,
        rho: float,
        dir_ignore: int,
        compute_plaquette: bool = False,
        compute_qcharge: bool = False,
    ):
        self.stoutSmear(n_steps, rho, dir_ignore, compute_plaquette, compute_qcharge)

    def smearHYP(
        self,
        n_steps: int,
        alpha1: float,
        alpha2: float,
        alpha3: float,
        dir_ignore: int,
        compute_plaquette: bool = False,
        compute_qcharge: bool = False,
    ):
        self.hypSmear(n_steps, alpha1, alpha2, alpha3, dir_ignore, compute_plaquette, compute_qcharge)

    def flowWilson(self, n_steps: int, time: float, compute_plaquette: bool = False, compute_qcharge: bool = True):
        return self.wilsonFlow(n_steps, time / n_steps, compute_plaquette, compute_qcharge)

    def flowWilsonScale(self, epsilon: float, compute_plaquette: bool = False, compute_qcharge: bool = True):
        return self.wilsonFlowScale(100000, epsilon, compute_plaquette, compute_qcharge)

    def flowSymanzik(self, n_steps: int, time: float, compute_plaquette: bool = False, compute_qcharge: bool = True):
        return self.symanzikFlow(n_steps, time / n_steps, compute_plaquette, compute_qcharge)

    def flowSymanzikScale(self, epsilon: float, compute_plaquette: bool = False, compute_qcharge: bool = True):
        return self.symanzikFlowScale(100000, epsilon, compute_plaquette, compute_qcharge)

    def plaquette(self):
        self.gauge_dirac.loadGauge(self)
        plaquette = self._gauge_dirac.plaquette()
        self._gauge_dirac.freeGauge()
        return plaquette

    def polyakovLoop(self):
        self.gauge_dirac.loadGauge(self)
        polyakovLoop = self._gauge_dirac.polyakovLoop()
        self._gauge_dirac.freeGauge()
        return polyakovLoop

    def energy(self):
        self.gauge_dirac.loadGauge(self)
        energy = self._gauge_dirac.energy()
        self._gauge_dirac.freeGauge()
        return energy

    def qcharge(self):
        self.gauge_dirac.loadGauge(self)
        qcharge = self._gauge_dirac.qcharge()
        self._gauge_dirac.freeGauge()
        return qcharge

    def qchargeDensity(self):
        self.gauge_dirac.loadGauge(self)
        qcharge_density = self._gauge_dirac.qchargeDensity()
        self._gauge_dirac.freeGauge()
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
        self.gauge_dirac.loadGauge(self)
        self._gauge_dirac.gaussGauge(seed, sigma)
        self._gauge_dirac.saveGauge(self)
        self._gauge_dirac.freeGauge()

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
    def load(cls, filename: str, *, check: bool = True) -> "LatticeMom":
        return super().load(filename, ["X", "Y", "Z", "T"], check=check)

    def save(self, filename: str, *, annotation: str = "", check: bool = True, use_fp32: bool = False):
        super().save(filename, ["X", "Y", "Z", "T"], annotation=annotation, check=check, use_fp32=use_fp32)

    def append(self, filename: str, *, annotation: str = "", check: bool = True, use_fp32: bool = False):
        super().append(filename, ["X", "Y", "Z", "T"], annotation=annotation, check=check, use_fp32=use_fp32)

    def update(self, filename: str, *, annotation: str = "", check: bool = True):
        super().update(filename, ["X", "Y", "Z", "T"], annotation=annotation, check=check)

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
