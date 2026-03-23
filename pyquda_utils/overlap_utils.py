from os import path
from time import perf_counter
import numpy as np
from mpi4py import MPI
from pyquda import quda, enum_quda
from pyquda_utils import core
from pyquda_comm import getLogger, readMPIFile, field

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    cp = None
    HAS_CUPY = False

comm = MPI.COMM_WORLD
mpi_size = comm.Get_size()

def _resolve_backend(backend: str = "auto") -> bool:
    """Resolve backend mode and return True when this call should use CuPy."""
    mode = str(backend).lower()
    if mode not in ("auto", "numpy", "cupy"):
        raise ValueError("backend must be one of: 'auto', 'numpy', 'cupy'")
    if mode == "numpy":
        return False
    if mode == "cupy":
        if not HAS_CUPY:
            raise RuntimeError("backend='cupy' requested but CuPy is not available")
        return True
    return HAS_CUPY

def _ensure_gpu(arr, use_cupy=None):
    """Move an array to GPU when use_cupy is enabled."""
    if use_cupy is None:
        use_cupy = HAS_CUPY
    if not use_cupy:
        return arr
    if isinstance(arr, cp.ndarray):
        return arr
    return cp.asarray(arr)

def _ensure_cpu(arr):
    """Move an array to CPU when it exposes a device-to-host getter."""
    if hasattr(arr, 'get'):
        return arr.get()
    return arr

def apply_gamma5(vec_or_data, backend: str = "auto"):
    """Apply gamma5 to a fermion object or raw data array."""
    use_cupy = _resolve_backend(backend)
    xp = cp if use_cupy else np

    # ndarray.data is a low-level buffer; only peel .data from fermion-like wrappers.
    is_np_array = isinstance(vec_or_data, np.ndarray)
    is_cp_array = HAS_CUPY and isinstance(vec_or_data, cp.ndarray)
    if is_np_array or is_cp_array:
        data = vec_or_data
    elif hasattr(vec_or_data, "data"):
        data = vec_or_data.data
    else:
        data = vec_or_data

    if use_cupy:
        data = _ensure_gpu(data, use_cupy)
    else:
        if HAS_CUPY and isinstance(data, cp.ndarray):
            data = cp.asnumpy(data)
    g5 = xp.array([1, 1, -1, -1], dtype=data.dtype)
    return data * g5[None, None, None, None, None, :, None]

def negMultiFermionFromDiracPauli(dirac_pauli: field.MultiLatticeFermion, backend: str = "auto"):
    """Convert a fermion field to the negative DeGrand-Rossi basis."""
    use_cupy = _resolve_backend(backend)
    if use_cupy:
        data_gpu = _ensure_gpu(dirac_pauli.data, use_cupy)
        data_gpu = data_gpu / (2**0.5)

        out_data_gpu = cp.empty_like(data_gpu)

        out_data_gpu[..., 0, :] = -data_gpu[..., 1] + data_gpu[..., 3]
        out_data_gpu[..., 1, :] =  data_gpu[..., 0] - data_gpu[..., 2]
        out_data_gpu[..., 2, :] =  data_gpu[..., 1] + data_gpu[..., 3]
        out_data_gpu[..., 3, :] = -data_gpu[..., 0] - data_gpu[..., 2]

        degrand_rossi = field.MultiLatticeFermion(dirac_pauli.latt_info, dirac_pauli.L5)
        degrand_rossi.data[:] = _ensure_cpu(out_data_gpu)
        return degrand_rossi
    else:
        data = dirac_pauli.data / 2**0.5
        degrand_rossi = field.MultiLatticeFermion(dirac_pauli.latt_info, dirac_pauli.L5)
        degrand_rossi.data[:, :, :, :, :, :, 0, :] = -data[:, :, :, :, :, :, 1] + data[:, :, :, :, :, :, 3]
        degrand_rossi.data[:, :, :, :, :, :, 1, :] = data[:, :, :, :, :, :, 0] - data[:, :, :, :, :, :, 2]
        degrand_rossi.data[:, :, :, :, :, :, 2, :] = data[:, :, :, :, :, :, 1] + data[:, :, :, :, :, :, 3]
        degrand_rossi.data[:, :, :, :, :, :, 3, :] = -data[:, :, :, :, :, :, 0] - data[:, :, :, :, :, :, 2]
        return degrand_rossi

def readEigenValue(file: str):
    """Read complex eigenvalues from a text file."""
    eigvals = []
    with open(file, "r") as f:
        for line in f.readlines():
            if line.startswith("EIGV"):
                tag, real, imag, res = line.strip().split()
                eigvals.append(float(real) + 1j * float(imag))
    return eigvals

def readEigenSystem(latt_info: field.LatticeInfo, eignum: int, file: str, use_fp32: bool, backend: str = "auto"):
    """Read eigenvalues/eigenvectors; backend is used for non-fp32 basis conversion."""
    s = perf_counter()
    file = path.expanduser(path.expandvars(file))
    eigvals = readEigenValue(f"{file}.eigvals")
    if eignum > len(eigvals):
        raise ValueError(f"Requested eignum={eignum}, but only {len(eigvals)} eigenvalues are available")
    Lx, Ly, Lz, Lt = latt_info.size
    Ns, Nc = latt_info.Ns, latt_info.Nc
    if use_fp32:
        eigvecs_raw = readMPIFile(f"{file}.s", "<c8", 0, (eignum, Lt, Lz, Ly, Lx, Ns, Nc), (4, 3, 2, 1)).astype("<c16")
        eigvecs = field.MultiLatticeFermion(latt_info, eignum, latt_info.evenodd(eigvecs_raw, True))
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
            field.MultiLatticeFermion(latt_info, eignum, latt_info.evenodd(eigvecs_raw, True)),
            backend=backend,
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
    tol=1e-12,
    backend: str = "auto",
):
    """Reconstruct full overlap eigensystem from half-chirality inputs."""
    n_ev = len(evals_half)
    lambda_sq = evals_half.real

    use_cupy = _resolve_backend(backend)
    xp = cp if use_cupy else np
    lambda_sq_gpu = xp.asarray(lambda_sq)
    b_sq = lambda_sq_gpu - lambda_sq_gpu**2
    b = xp.sqrt(xp.maximum(0, b_sq))
    evals_final = lambda_sq_gpu + 1j * b
    evals_final_cpu = _ensure_cpu(evals_final)

    reconstructed_evecs = core.MultiLatticeFermion(latt_info, n_ev)
    full_evec_in = core.LatticeFermion(latt_info)
    full_evec_out = core.LatticeFermion(latt_info)

    for i in range(n_ev):
        b_val = evals_final_cpu[i].imag
        is_zero_mode = False
        if abs(lambda_sq[i]) < 1e-12 or abs(b_val) < tol:
            is_zero_mode = True
            factor = 0.0
        else:
            is_zero_mode = False
            factor = (2 * kappa) / (1j * b_val)

        full_evec_in.data[:] = 0
        if chirality == enum_quda.QudaChirality.QUDA_LEFT_CHIRALITY:
            full_evec_in.data[..., 2:, :] = evecs_half[i].data
        elif chirality == enum_quda.QudaChirality.QUDA_RIGHT_CHIRALITY:
            full_evec_in.data[..., :2, :] = evecs_half[i].data
        
        if not is_zero_mode:
            quda.MatQuda(full_evec_out.data_ptr, full_evec_in.data_ptr, invert_param)
        
        if use_cupy:
            recon_gpu = _ensure_gpu(reconstructed_evecs[i].data, use_cupy)
            half_gpu = _ensure_gpu(evecs_half[i].data, use_cupy)
            full_out_gpu = _ensure_gpu(full_evec_out.data, use_cupy)
            
            if chirality == enum_quda.QudaChirality.QUDA_LEFT_CHIRALITY:
                recon_gpu[..., :2, :] = full_out_gpu[..., :2, :] * factor
                recon_gpu[..., 2:, :] = half_gpu
            elif chirality == enum_quda.QudaChirality.QUDA_RIGHT_CHIRALITY:
                recon_gpu[..., :2, :] = half_gpu
                recon_gpu[..., 2:, :] = full_out_gpu[..., 2:, :] * factor
            
            if normalize:
                local_norm_sq = cp.sum(cp.abs(recon_gpu)**2)
                local_norm_sq_cpu = float(local_norm_sq)
                
                if comm is not None and mpi_size > 1:
                    sendbuf = np.array(local_norm_sq_cpu, dtype=np.float64)
                    recvbuf = np.array(0.0, dtype=np.float64)
                    comm.Allreduce(sendbuf, recvbuf, op=MPI.SUM)
                    global_norm = np.sqrt(recvbuf)
                else:
                    global_norm = np.sqrt(local_norm_sq_cpu)

                if global_norm > 1e-15:
                    recon_gpu /= global_norm
            
            reconstructed_evecs[i].data[:] = _ensure_cpu(recon_gpu)

        else:
            if chirality == enum_quda.QudaChirality.QUDA_LEFT_CHIRALITY:
                reconstructed_evecs[i].data[..., :2, :] = full_evec_out.data[..., :2, :] * factor
                reconstructed_evecs[i].data[..., 2:, :] = evecs_half[i].data
            elif chirality == enum_quda.QudaChirality.QUDA_RIGHT_CHIRALITY:
                reconstructed_evecs[i].data[..., :2, :] = evecs_half[i].data
                reconstructed_evecs[i].data[..., 2:, :] = full_evec_out.data[..., 2:, :] * factor
            
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

    return reconstructed_evecs, evals_final_cpu

def deflation(evecs_ov, b, backend: str = "auto"):
    """Perform sequential deflation on b using overlap eigenvectors."""
    use_cupy = _resolve_backend(backend)
    if not use_cupy:
        num_eigen = evecs_ov.data.shape[0]
        b_defl = b.copy()
        for i in range(b.L5):
            b_defl[i].data[:] = b[i].data
            for j in range(num_eigen):
                local_alpha = np.vdot(evecs_ov.data[j].ravel(), b_defl[i].data.ravel())
                if comm is not None and mpi_size > 1:
                    alpha = comm.allreduce(local_alpha, op=MPI.SUM)
                else:
                    alpha = local_alpha

                b_defl[i].data -= alpha * evecs_ov.data[j]

                gamma5_evec = apply_gamma5(evecs_ov[j], backend="numpy")
                local_alpha_g5 = np.vdot(gamma5_evec.ravel(), b_defl[i].data.ravel())
                if comm is not None and mpi_size > 1:
                    alpha_g5 = comm.allreduce(local_alpha_g5, op=MPI.SUM)
                else:
                    alpha_g5 = local_alpha_g5

                b_defl[i].data -= alpha_g5 * gamma5_evec
        return b_defl

    num_eigen = evecs_ov.data.shape[0]
    b_defl = b.copy()

    evecs_gpu = _ensure_gpu(evecs_ov.data, use_cupy)
    b_gpu = _ensure_gpu(b.data, use_cupy)
    b_defl_gpu = _ensure_gpu(b_defl.data, use_cupy)
    gamma5_evecs_gpu = apply_gamma5(evecs_gpu, backend="cupy")

    for i in range(b.L5):
        b_defl_gpu[i] = b_gpu[i]
        for j in range(num_eigen):
            local_alpha = cp.vdot(evecs_gpu[j].ravel(), b_defl_gpu[i].ravel()).item()
            if comm is not None and mpi_size > 1:
                alpha = comm.allreduce(local_alpha, op=MPI.SUM)
            else:
                alpha = local_alpha

            b_defl_gpu[i] -= alpha * evecs_gpu[j]

            local_alpha_g5 = cp.vdot(gamma5_evecs_gpu[j].ravel(), b_defl_gpu[i].ravel()).item()
            if comm is not None and mpi_size > 1:
                alpha_g5 = comm.allreduce(local_alpha_g5, op=MPI.SUM)
            else:
                alpha_g5 = local_alpha_g5

            b_defl_gpu[i] -= alpha_g5 * gamma5_evecs_gpu[j]

    b_defl.data[:] = _ensure_cpu(b_defl_gpu)
    return b_defl

def compute_low_mode_multi_mass(b_src, evecs_ov, evals_ov, mass_list, rho, backend: str = "auto"):
    """Compute low-mode propagators for all masses in mass_list."""
    use_cupy = _resolve_backend(backend)
    if not use_cupy:
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

    x_low_list = []
    for _ in mass_list:
        tmp = b_src.copy()
        tmp.data[:] = 0
        x_low_list.append(tmp)

    num_eigen = len(evals_ov)
    evals_gpu = cp.asarray(evals_ov)
    evecs_gpu = _ensure_gpu(evecs_ov.data, use_cupy)
    b_src_gpu = _ensure_gpu(b_src.data, use_cupy)
    x_low_gpu_list = [_ensure_gpu(x.data, use_cupy) for x in x_low_list]

    for s_idx in range(b_src.L5):
        src_data = b_src_gpu[s_idx]
        src_L = src_data[..., 0:2, :]
        src_R = src_data[..., 2:4, :]

        projections = []
        for i in range(num_eigen):
            vec_data = evecs_gpu[i]
            vec_L = vec_data[..., 0:2, :]
            vec_R = vec_data[..., 2:4, :]

            local_left = cp.vdot(vec_L, src_L).item()
            local_right = cp.vdot(vec_R, src_R).item()

            if comm is not None and mpi_size > 1:
                leftCoef = comm.allreduce(local_left, op=MPI.SUM)
                rightCoef = comm.allreduce(local_right, op=MPI.SUM)
            else:
                leftCoef = local_left
                rightCoef = local_right

            projections.append((leftCoef, rightCoef))

        for i in range(num_eigen):
            vec_data = evecs_gpu[i]
            vec_L = vec_data[..., 0:2, :]
            vec_R = vec_data[..., 2:4, :]

            leftCoef, rightCoef = projections[i]
            val = evals_gpu[i] * 2
            is_zero_mode = cp.sqrt(cp.abs(val.real)) > 100 * cp.abs(val.imag)

            for m_idx, m_phys in enumerate(mass_list):
                out_data = x_low_gpu_list[m_idx][s_idx]

                if bool(is_zero_mode.item()):
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

    for out_obj, out_gpu in zip(x_low_list, x_low_gpu_list):
        out_obj.data[:] = _ensure_cpu(out_gpu)

    return x_low_list

def compute_low_mode(b, evecs_ov, evals_ov, mass, rho, backend: str = "auto"):
    """Compute the low-mode propagator for a single mass."""
    return compute_low_mode_multi_mass(b, evecs_ov, evals_ov, [mass], rho, backend=backend)[0]