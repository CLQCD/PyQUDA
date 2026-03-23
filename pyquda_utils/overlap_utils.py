from os import path
from time import perf_counter
import numpy as np
from mpi4py import MPI
from pyquda import quda
from pyquda import enum_quda
from pyquda_utils import core
from pyquda_comm import  getLogger, readMPIFile
from pyquda_comm.field import LatticeInfo,  MultiLatticeFermion

comm = MPI.COMM_WORLD
mpi_size = comm.Get_size()
mpi_rank = comm.Get_rank()

def negMultiFermionFromDiracPauli(dirac_pauli: MultiLatticeFermion):
    """Convert to the negative DeGrand-Rossi basis"""
    data = dirac_pauli.data / 2**0.5
    degrand_rossi = MultiLatticeFermion(dirac_pauli.latt_info, dirac_pauli.L5)
    degrand_rossi.data[:, :, :, :, :, :, 0, :] = -data[:, :, :, :, :, :, 1] + data[:, :, :, :, :, :, 3]
    degrand_rossi.data[:, :, :, :, :, :, 1, :] = data[:, :, :, :, :, :, 0] - data[:, :, :, :, :, :, 2]
    degrand_rossi.data[:, :, :, :, :, :, 2, :] = data[:, :, :, :, :, :, 1] + data[:, :, :, :, :, :, 3]
    degrand_rossi.data[:, :, :, :, :, :, 3, :] = -data[:, :, :, :, :, :, 0] - data[:, :, :, :, :, :, 2]
    return degrand_rossi

def readEigenValue(file: str):
    eigvals = []
    with open(file, "r") as f:
        for line in f.readlines():
            if line.startswith("EIGV"):
                tag, real, imag, res = line.strip().split()
                eigvals.append(float(real) + 1j * float(imag))
    return eigvals


def readEigenSystem(latt_info: LatticeInfo, eignum: int, file: str, use_fp32: bool):
    """Convert to the negative DeGrand-Rossi"""
    s = perf_counter()
    file = path.expanduser(path.expandvars(file))
    eigvals = readEigenValue(f"{file}.eigvals")
    assert eignum <= len(eigvals)
    Lx, Ly, Lz, Lt = latt_info.size
    Ns, Nc = latt_info.Ns, latt_info.Nc
    if use_fp32:
        eigvecs_raw = readMPIFile(f"{file}.s", "<c8", 0, (eignum, Lt, Lz, Ly, Lx, Ns, Nc), (4, 3, 2, 1)).astype("<c16")
        eigvecs = MultiLatticeFermion(latt_info, eignum, latt_info.evenodd(eigvecs_raw, True))
    else:
        eigvecs_raw = (
            readMPIFile(file, ">f8", 0, (eignum, 2, Ns, Nc, Lt, Lz, Ly, Lx), (7, 6, 5, 4))
            .astype("<f8")
            .transpose(0, 4, 5, 6, 7, 2, 3, 1)
            .reshape(eignum, Lt, Lz, Ly, Lx, Ns, Nc * 2)
            .copy()
            .view("<c16")
        )
        eigvecs = negMultiFermionFromDiracPauli(
            MultiLatticeFermion(latt_info, eignum, latt_info.evenodd(eigvecs_raw, True))
        )
    getLogger().info(f"Read {eignum} eigen system in {perf_counter() - s:.3} secs")
    return eigvals[:eignum], eigvecs


def reconstruct_eigensystem(
    evecs_half, 
    evals_half, 
    invert_param, 
    latt_info, 
    kappa, 
    chirality, 
    normalize=True,
    tol=1e-12
):
    """
    Reconstruct the full overlap eigensystem from half-chirality results.

    Equations:
        1. lambda = a + i * sqrt(2*rho*a - a^2)
        2. v_missing = 1/(ib_i) * (1+-gamma_5)/2 * D * v_known
    """
    
    n_ev = len(evals_half)
    
    lambda_sq = evals_half.real
    evals_final = np.zeros_like(evals_half, dtype=np.complex128)
    b_sq = lambda_sq - lambda_sq**2
    b = np.sqrt(np.maximum(0, b_sq))
    evals_final = lambda_sq + 1j * b

    reconstructed_evecs = core.MultiLatticeFermion(latt_info, n_ev)
    full_evec_in = core.LatticeFermion(latt_info)
    full_evec_out = core.LatticeFermion(latt_info)

    for i in range(n_ev):
        b_val = evals_final[i].imag
        
        is_zero_mode = False
        if abs(lambda_sq[i]) < 1e-12 or abs(b_val) < tol:
            is_zero_mode = True
            factor = 0.0
        else:
            is_zero_mode = False
            factor = (2 * kappa) / (1j * b_val) # multiply by 2*kappa to bridge the gap between D_quda and D_phys

        full_evec_in.data[:] = 0
        if chirality == enum_quda.QudaChirality.QUDA_LEFT_CHIRALITY:
            full_evec_in.data[..., 2:, :] = evecs_half[i].data
        elif chirality == enum_quda.QudaChirality.QUDA_RIGHT_CHIRALITY:
            full_evec_in.data[..., :2, :] = evecs_half[i].data
        
        if not is_zero_mode:
            quda.MatQuda(full_evec_out.data_ptr, full_evec_in.data_ptr, invert_param)
        else:
            pass

        if chirality == enum_quda.QudaChirality.QUDA_LEFT_CHIRALITY:
            reconstructed_evecs[i, ..., :2, :] = full_evec_out.data[..., :2, :] * factor
            reconstructed_evecs[i, ..., 2:, :] = evecs_half[i].data
        elif chirality == enum_quda.QudaChirality.QUDA_RIGHT_CHIRALITY:
            reconstructed_evecs[i, ..., :2, :] = evecs_half[i].data
            reconstructed_evecs[i, ..., 2:, :] = full_evec_out.data[..., 2:, :] * factor

        if normalize:
            local_norm_sq = np.sum(np.abs(reconstructed_evecs[i].data)**2)
            
            if comm is not None and mpi_size > 1:
                sendbuf = np.array(local_norm_sq, dtype=np.float64)
                recvbuf = np.array(0.0, dtype=np.float64)
                comm.Allreduce(sendbuf, recvbuf, op=MPI.SUM)
                global_norm = np.sqrt(recvbuf)
            else:
                global_norm = np.sqrt(local_norm_sq)

            if global_norm > 1e-15:
                reconstructed_evecs[i].data /= global_norm

    return reconstructed_evecs, evals_final


def apply_gamma5(vec):
    """Apply gamma5 matrix to the fermion vector."""
    # Gamma5 definition: +1 for upper(L), -1 for lower(R)
    g5 = np.array([1, 1, -1, -1], dtype=vec.data.dtype)
    return vec.data * g5[None, None, None, None, None, :, None]


def deflation(evecs_ov, b):
    """
    Perform deflation (remove eigenmode components) on the source vector b.
    """
    num_eigen = evecs_ov.data.shape[0]
    b_defl = b.copy()
    
    # Iterate over each source vector in the MultiLatticeFermion b
    for i in range(b.L5):
        b_defl[i].data[:] = b[i].data
        for j in range(num_eigen):
            # 1. Orthogonal projection
            local_alpha = np.vdot(evecs_ov.data[j].ravel(), b_defl[i].data.ravel())
            
            if comm is not None and mpi_size > 1:
                alpha = comm.allreduce(local_alpha, op=MPI.SUM)
            else:
                alpha = local_alpha
                
            b_defl[i].data -= alpha * evecs_ov.data[j]
            
            # 2. Gamma5 projection
            gamma5_evec = apply_gamma5(evecs_ov[j])
            local_alpha_g5 = np.vdot(gamma5_evec.ravel(), b_defl[i].data.ravel())
            
            if comm is not None and mpi_size > 1:
                alpha_g5 = comm.allreduce(local_alpha_g5, op=MPI.SUM)
            else:
                alpha_g5 = local_alpha_g5
                
            b_defl[i].data -= alpha_g5 * gamma5_evec
            
    return b_defl


def compute_low_mode(b, evecs_ov, evals_ov, mass, rho):
    """
    Compute the low-mode part of the overlap propagator for a SINGLE mass.
    """
    x_low = b.copy()
    x_low.data[:] = 0

    num_eigen = len(evals_ov)
    
    for s_idx in range(b.L5):
        src_data = b[s_idx].data
        out_data = x_low[s_idx].data
        
        src_L = src_data[..., 0:2, :]
        src_R = src_data[..., 2:4, :]

        for i in range(num_eigen):
            vec_data = evecs_ov[i].data
            vec_L = vec_data[..., 0:2, :]
            vec_R = vec_data[..., 2:4, :]
            
            # Projection coefficients
            local_left = np.vdot(vec_L, src_L)
            local_right = np.vdot(vec_R, src_R)
            
            if comm is not None and mpi_size > 1:
                leftCoef = comm.allreduce(local_left, op=MPI.SUM)
                rightCoef = comm.allreduce(local_right, op=MPI.SUM)
            else:
                leftCoef = local_left
                rightCoef = local_right
            
            val = evals_ov[i] * 2 

            # Zero mode check
            is_zero_mode = np.sqrt(np.abs(val.real)) > 100 * np.abs(val.imag)
            
            if is_zero_mode:
                inv_m = (1.0 / mass) * (2 * rho)
                coeff = (leftCoef + rightCoef) * inv_m
                out_data += coeff * vec_data
            else:
                half = 0.5 + 0j
                denom = rho * val + mass * (1.0 - val * half)
                inv_m_val = (1.0 - val * half) / denom
                inv_m_val *= (2 * rho)
                
                left_prj = inv_m_val.real * leftCoef + inv_m_val.imag * rightCoef * 1j
                right_prj = inv_m_val.real * rightCoef + inv_m_val.imag * leftCoef * 1j
                
                out_data[..., 0:2, :] += 2.0 * left_prj * vec_L
                out_data[..., 2:4, :] += 2.0 * right_prj * vec_R
                
    return x_low


def compute_low_mode_multi_mass(b_src, evecs_ov, evals_ov, mass_list, rho):
    """
    Compute low-mode propagator for MULTIPLE masses.
    Returns a list of MultiLatticeFermion.
    """
    x_low_list = []
    
    for _ in mass_list:
        tmp = b_src.copy()
        tmp.data[:] = 0
        x_low_list.append(tmp)

    num_eigen = len(evals_ov)

    for s_idx in range(b_src.L5):
        src_data = b_src[s_idx].data
        src_L = src_data[..., 0:2, :]
        src_R = src_data[..., 2:4, :]

        projections = []
        for i in range(num_eigen):
            vec_data = evecs_ov[i].data
            vec_L = vec_data[..., 0:2, :]
            vec_R = vec_data[..., 2:4, :]
            
            local_left = np.vdot(vec_L, src_L)
            local_right = np.vdot(vec_R, src_R)
            
            if comm is not None and mpi_size > 1:
                leftCoef = comm.allreduce(local_left, op=MPI.SUM)
                rightCoef = comm.allreduce(local_right, op=MPI.SUM)
            else:
                leftCoef = local_left
                rightCoef = local_right
            
            projections.append((leftCoef, rightCoef))

        for i in range(num_eigen):
            vec_data = evecs_ov[i].data
            vec_L = vec_data[..., 0:2, :]
            vec_R = vec_data[..., 2:4, :]
            
            leftCoef, rightCoef = projections[i]
            val = evals_ov[i] * 2
            is_zero_mode = np.sqrt(np.abs(val.real)) > 100 * np.abs(val.imag)

            for m_idx, m_phys in enumerate(mass_list):
                out_data = x_low_list[m_idx][s_idx].data
                
                if is_zero_mode:
                    inv_m = (1.0 / m_phys) * (2 * rho)
                    coeff = (leftCoef + rightCoef) * inv_m
                    out_data += coeff * vec_data
                else:
                    half = 0.5 + 0j
                    denom = rho * val + m_phys * (1.0 - val * half)
                    inv_m_val = (1.0 - val * half) / denom
                    inv_m_val *= (2 * rho)
                    
                    left_prj = inv_m_val.real * leftCoef + inv_m_val.imag * rightCoef * 1j
                    right_prj = inv_m_val.real * rightCoef + inv_m_val.imag * leftCoef * 1j
                    
                    out_data[..., 0:2, :] += 2.0 * left_prj * vec_L
                    out_data[..., 2:4, :] += 2.0 * right_prj * vec_R
                    
    return x_low_list