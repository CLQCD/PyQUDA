#include <cupy/complex.cuh>

#define Nc 3
#define Nd 4
#define get_x(_coord, _X) \
  (((_coord[3] * _X[2] + _coord[2]) * _X[1] + _coord[1]) * _X[0] + _coord[0])

template <typename T>
class Matrix {
private:
  T data[Nc][Nc]{};

public:
  __device__ __host__ Matrix() = default;

  __device__ __host__ Matrix(const T *source)
  {
    for (int i = 0; i < Nc; ++i) {
      for (int j = 0; j < Nc; ++j) {
        data[i][j] = source[i * Nc + j];
      }
    }
  }

  __device__ __host__ const T *operator[](const int i) const
  {
    return data[i];
  }

  __device__ __host__ T *operator[](const int i)
  {
    return data[i];
  }

  __device__ __host__ Matrix<T> operator+(const T &rhs)
  {
    Matrix<T> result;
    for (int i = 0; i < Nc; ++i) {
      for (int j = 0; j < Nc; ++j) {
        result[i][j] = data[i][j];
      }
      result[i][i] += rhs;
    }
    return result;
  }

  __device__ __host__ Matrix<T> operator+(const Matrix<T> &rhs)
  {
    Matrix<T> result;
    for (int i = 0; i < Nc; ++i) {
      for (int j = 0; j < Nc; ++j) {
        result[i][j] = data[i][j] + rhs[i][j];
      }
    }
    return result;
  }

  __device__ __host__ Matrix<T> &operator+=(const T &rhs)
  {
    for (int i = 0; i < Nc; ++i) {
      data[i][i] += rhs;
    }
  }

  __device__ __host__ Matrix<T> &operator+=(const Matrix<T> &rhs)
  {
    for (int i = 0; i < Nc; ++i) {
      for (int j = 0; j < Nc; ++j) {
        data[i][j] += rhs[i][j];
      }
    }
  }

  __device__ __host__ Matrix<T> operator-(const T &rhs)
  {
    Matrix<T> result;
    for (int i = 0; i < Nc; ++i) {
      for (int j = 0; j < Nc; ++j) {
        result[i][j] = data[i][j];
      }
      result[i][i] -= rhs;
    }
    return result;
  }

  __device__ __host__ Matrix<T> operator-(const Matrix<T> &rhs)
  {
    Matrix<T> result;
    for (int i = 0; i < Nc; ++i) {
      for (int j = 0; j < Nc; ++j) {
        result[i][j] = data[i][j] - rhs[i][j];
      }
    }
    return result;
  }

  __device__ __host__ Matrix<T> &operator-=(const T &rhs)
  {
    for (int i = 0; i < Nc; ++i) {
      data[i][i] -= rhs;
    }
  }

  __device__ __host__ Matrix<T> &operator-=(const Matrix<T> &rhs)
  {
    for (int i = 0; i < Nc; ++i) {
      for (int j = 0; j < Nc; ++j) {
        data[i][j] -= rhs[i][j];
      }
    }
  }

  __device__ __host__ Matrix<T> operator*(const T &rhs)
  {
    Matrix<T> result;
    for (int i = 0; i < Nc; ++i) {
      for (int j = 0; j < Nc; ++j) {
        result[i][j] = data[i][j] * rhs;
      }
    }
    return result;
  }

  friend __device__ __host__ Matrix<T> operator*(const T &lhs, const Matrix<T> &rhs)
  {
    Matrix<T> result;
    for (int i = 0; i < Nc; ++i) {
      for (int j = 0; j < Nc; ++j) {
        result[i][j] = lhs * rhs[i][j];
      }
    }
    return result;
  }

  __device__ __host__ Matrix<T> operator*(const Matrix<T> &rhs)
  {
    Matrix<T> result;
    for (int i = 0; i < Nc; ++i) {
      for (int j = 0; j < Nc; ++j) {
        result[i][j] = 0;
        for (int k = 0; k < Nc; ++k) {
          result[i][j] += data[i][k] * rhs[k][j];
        }
      }
    }
    return result;
  }

  __device__ __host__ Matrix<T> &operator*=(const Matrix<T> &rhs)
  {
    Matrix<T> result;
    for (int i = 0; i < Nc; ++i) {
      for (int j = 0; j < Nc; ++j) {
        result[i][j] = 0;
        for (int k = 0; k < Nc; ++k) {
          result[i][j] += data[i][k] * rhs[k][j];
        }
      }
    }
    *this = result;
    return *this;
  }
};

template <typename T>
__device__ __host__ T trace(const Matrix<T> &matrix)
{
  T result = 0;
  for (int i = 0; i < Nc; ++i) {
    result += matrix[i][i];
  }
  return result;
}

template <typename T>
__device__ __host__ Matrix<T> adjoint(const Matrix<T> &matrix)
{
  Matrix<T> result;
  for (int i = 0; i < Nc; ++i) {
    for (int j = 0; j < Nc; ++j) {
      result[i][j] = conj(matrix[j][i]);
    }
  }
  return result;
}

template <typename T>
__device__ __host__ Matrix<T> i_herm(const Matrix<T> &matrix)
{
  Matrix<T> result;
  for (int i = 0; i < Nc; ++i) {
    for (int j = 0; j < Nc; ++j) {
      result[i][j].real((matrix[j][i].imag() + matrix[i][j].imag()) / 2.);
      result[i][j].imag((matrix[j][i].real() - matrix[i][j].real()) / 2.);
    }
  }
  result -= trace(result) / T(Nc);
  return result;
}

template <typename T>
__device__ __host__ Matrix<T> herm(const Matrix<T> &matrix)
{
  Matrix<T> result;
  for (int i = 0; i < Nc; ++i) {
    for (int j = 0; j < Nc; ++j) {
      result[i][j].real((matrix[i][j].real() + matrix[j][i].real()) / 2.);
      result[i][j].imag((matrix[i][j].imag() - matrix[j][i].imag()) / 2.);
    }
  }
  result -= trace(result) / T(Nc);
  return result;
}

__device__ __host__ int index_from_pcoord(const int p_X[5], const int p_coord[5], int nu, int volume)
{
  return ((nu * 2 + p_coord[4]) * volume + get_x(p_coord, p_X)) * Nc * Nc;
}

__device__ __host__ int index_from_coord(const int p_X[5], const int coord[4], int nu, int volume)
{
  const int parity = (coord[0] + coord[1] + coord[2] + coord[3]) % 2;

  int p_coord[Nd + 1];
  p_coord[0] = coord[0] / 2;
  p_coord[1] = coord[1];
  p_coord[2] = coord[2];
  p_coord[3] = coord[3];
  p_coord[4] = parity;

  // return ((nu * 2 + parity) * volume + get_x(p_coord, p_X)) * Nc * Nc;
  return index_from_pcoord(p_X, p_coord, nu, volume);
}

template <typename T, int DIM>
__global__ void stout_smear(
    complex<T> *U_out,
    const complex<T> *U_in,
    const T rho,
    const int Lx,
    const int Ly,
    const int Lz,
    const int Lt)
{
  const int volume = Lx * Ly * Lz * Lt / 2;
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int mu = blockIdx.y * blockDim.y + threadIdx.y;
  const int parity = blockIdx.z * blockDim.z + threadIdx.z;
  if (x >= volume || mu >= DIM || parity >= 2) {
    return;
  }
  const int p_X[Nd + 1] = {Lx / 2, Ly, Lz, Lt, 2};
  const int X[Nd] = {Lx, Ly, Lz, Lt};
  const int p_coord[Nd + 1] = {x % p_X[0], x / p_X[0] % p_X[1], x / (p_X[0] * p_X[1]) % p_X[2], x / (p_X[0] * p_X[1] * p_X[2]) % p_X[3], parity};
  int coord[Nd] = {(p_coord[0] << 1) + (parity ^ ((p_coord[1] + p_coord[2] + p_coord[3]) & 1)), p_coord[1], p_coord[2], p_coord[3]};
  typedef Matrix<complex<T>> ColorMatrix;
  ColorMatrix U(U_in + index_from_pcoord(p_X, p_coord, mu, volume));
  ColorMatrix Q;
  for (int nu = 0; nu < DIM; ++nu) {
    if (nu != mu) {
      ColorMatrix staple1(U_in + index_from_pcoord(p_X, p_coord, nu, volume));
      coord[nu] = (coord[nu] + 1 + X[nu]) % X[nu];
      staple1 = staple1 * ColorMatrix(U_in + index_from_coord(p_X, coord, mu, volume));
      coord[mu] = (coord[mu] + 1 + X[mu]) % X[mu];
      coord[nu] = (coord[nu] - 1 + X[nu]) % X[nu];
      staple1 = staple1 * adjoint(ColorMatrix(U_in + index_from_coord(p_X, coord, nu, volume)));
      coord[mu] = (coord[mu] - 1 + X[mu]) % X[mu];

      coord[nu] = (coord[nu] - 1 + X[nu]) % X[nu];
      ColorMatrix staple2(U_in + index_from_coord(p_X, coord, nu, volume));
      staple2 = adjoint(staple2) * ColorMatrix(U_in + index_from_coord(p_X, coord, mu, volume));
      coord[mu] = (coord[mu] + 1 + X[mu]) % X[mu];
      staple2 = staple2 * ColorMatrix(U_in + index_from_coord(p_X, coord, nu, volume));
      coord[nu] = (coord[nu] + 1 + X[nu]) % X[nu];
      coord[mu] = (coord[mu] - 1 + X[mu]) % X[mu];

      Q += staple1 + staple2;
    }
  }
  Q = i_herm((Q * rho) * adjoint(U));

  ColorMatrix Q_sq = Q * Q;
  double c0 = trace(Q_sq * Q).real() / 3;
  double c1 = trace(Q_sq).real() / 2;
  double c0_max = 2 * c1 / 3 * sqrt(c1 / 3);
  int sig_parity = 0;
  if (c0 < 0) {
    sig_parity = 1;
    c0 *= -1;
  }
  c0_max = fmax(c0_max, 1e-15);
  c1 = fmax(c1, 1e-15);
  double theta = acos(c0 / c0_max);
  double u = sqrt(c1 / 3) * cos(theta / 3);
  double w = sqrt(c1) * sin(theta / 3);
  double u_sq = u * u;
  double w_sq = w * w;
  double e_iu_real = cos(u);
  double e_iu_imag = sin(u);
  double e_2iu_real = cos(2 * u);
  double e_2iu_imag = sin(2 * u);
  double cos_w = cos(w);
  double sinc_w = 1 - w_sq / 6 * (1 - w_sq / 20 * (1 - w_sq / 42 * (1 - w_sq / 72)));
  if (fabs(w) > 0.05) {
    sinc_w = sin(w) / w;
  }
  double f_denom = 1 / (9 * u_sq - w_sq);
  double f0_real = ((u_sq - w_sq) * e_2iu_real + e_iu_real * 8 * u_sq * cos_w + e_iu_imag * 2 * u * (3 * u_sq + w_sq) * sinc_w) * f_denom;
  double f0_imag = ((u_sq - w_sq) * e_2iu_imag - e_iu_imag * 8 * u_sq * cos_w + e_iu_real * 2 * u * (3 * u_sq + w_sq) * sinc_w) * f_denom;
  double f1_real = (2 * u * e_2iu_real - e_iu_real * 2 * u * cos_w + e_iu_imag * (3 * u_sq - w_sq) * sinc_w) * f_denom;
  double f1_imag = (2 * u * e_2iu_imag + e_iu_imag * 2 * u * cos_w + e_iu_real * (3 * u_sq - w_sq) * sinc_w) * f_denom;
  double f2_real = (e_2iu_real - e_iu_real * cos_w - e_iu_imag * 3 * u * sinc_w) * f_denom;
  double f2_imag = (e_2iu_imag + e_iu_imag * cos_w - e_iu_real * 3 * u * sinc_w) * f_denom;
  if (sig_parity) {
    f0_imag *= -1;
    f1_real *= -1;
    f2_imag *= -1;
  }
  complex<T> f[3];
  f[0] = {T(f0_real), T(f0_imag)};
  f[1] = {T(f1_real), T(f1_imag)};
  f[2] = {T(f2_real), T(f2_imag)};
  ColorMatrix e_iQ = Q_sq * f[2] + Q * f[1] + f[0];
  U = e_iQ * U;
  for (int i = 0; i < Nc; ++i) {
    for (int j = 0; j < Nc; ++j) {
      U_out[index_from_pcoord(p_X, p_coord, mu, volume) + i * Nc + j] = U[i][j];
    }
  }
}

template <typename T, int DIM>
__global__ void compute_lambda_kernel(
    // const T *sigma,
    complex<T> *sigma,
    complex<T> *lambda,
    const complex<T> *U_in,
    const T rho,
    const int Lx,
    const int Ly,
    const int Lz,
    const int Lt)
{
  const int volume = Lx * Ly * Lz * Lt / 2;
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int mu = blockIdx.y * blockDim.y + threadIdx.y;
  const int parity = blockIdx.z * blockDim.z + threadIdx.z;
  if (x >= volume || mu >= DIM || parity >= 2) {
    return;
  }
  const int p_X[Nd + 1] = {Lx / 2, Ly, Lz, Lt, 2};
  const int X[Nd] = {Lx, Ly, Lz, Lt};
  const int p_coord[Nd + 1] = {x % p_X[0], x / p_X[0] % p_X[1], x / (p_X[0] * p_X[1]) % p_X[2], x / (p_X[0] * p_X[1] * p_X[2]) % p_X[3], parity};
  int coord[Nd] = {(p_coord[0] << 1) + (parity ^ ((p_coord[1] + p_coord[2] + p_coord[3]) & 1)), p_coord[1], p_coord[2], p_coord[3]};
  typedef Matrix<complex<T>> ColorMatrix;
  ColorMatrix U(U_in + index_from_pcoord(p_X, p_coord, mu, volume));
  ColorMatrix Q;
  for (int nu = 0; nu < DIM; ++nu) {
    if (nu != mu) {
      ColorMatrix staple1(U_in + index_from_pcoord(p_X, p_coord, nu, volume));
      coord[nu] = (coord[nu] + 1 + X[nu]) % X[nu];
      staple1 = staple1 * ColorMatrix(U_in + index_from_coord(p_X, coord, mu, volume));
      coord[mu] = (coord[mu] + 1 + X[mu]) % X[mu];
      coord[nu] = (coord[nu] - 1 + X[nu]) % X[nu];
      staple1 = staple1 * adjoint(ColorMatrix(U_in + index_from_coord(p_X, coord, nu, volume)));
      coord[mu] = (coord[mu] - 1 + X[mu]) % X[mu];

      coord[nu] = (coord[nu] - 1 + X[nu]) % X[nu];
      ColorMatrix staple2(U_in + index_from_coord(p_X, coord, nu, volume));
      staple2 = adjoint(staple2) * ColorMatrix(U_in + index_from_coord(p_X, coord, mu, volume));
      coord[mu] = (coord[mu] + 1 + X[mu]) % X[mu];
      staple2 = staple2 * ColorMatrix(U_in + index_from_coord(p_X, coord, nu, volume));
      coord[nu] = (coord[nu] + 1 + X[nu]) % X[nu];
      coord[mu] = (coord[mu] - 1 + X[mu]) % X[mu];

      Q += staple1 + staple2;
    }
  }
  Q = i_herm((Q * rho) * adjoint(U));

  ColorMatrix Q_sq = Q * Q;
  double c0 = trace(Q_sq * Q).real() / 3;
  double c1 = trace(Q_sq).real() / 2;
  double c0_max = 2 * c1 / 3 * sqrt(c1 / 3);
  int sig_parity = 0;
  if (c0 < 0) {
    sig_parity = 1;
    c0 *= -1;
  }
  c0_max = fmax(c0_max, 1e-15);
  c1 = fmax(c1, 1e-15);
  double theta = acos(c0 / c0_max);
  double u = sqrt(c1 / 3) * cos(theta / 3);
  double w = sqrt(c1) * sin(theta / 3);
  double u_sq = u * u;
  double w_sq = w * w;
  complex<T> I = {T(0), T(1)};
  double e_iu_real = cos(u);
  double e_iu_imag = sin(u);
  double e_2iu_real = cos(2 * u);
  double e_2iu_imag = sin(2 * u);
  double sin_w = sin(w);
  double cos_w = cos(w);
  double sinc_w = 1 - w_sq / 6 * (1 - w_sq / 20 * (1 - w_sq / 42 * (1 - w_sq / 72)));
  if (fabs(w) > 0.05) {
    sinc_w = sin_w / w;
  }
  double f_denom = 1 / (9 * u_sq - w_sq);
  double f0_real = ((u_sq - w_sq) * e_2iu_real + e_iu_real * 8 * u_sq * cos_w + e_iu_imag * 2 * u * (3 * u_sq + w_sq) * sinc_w) * f_denom;
  double f0_imag = ((u_sq - w_sq) * e_2iu_imag - e_iu_imag * 8 * u_sq * cos_w + e_iu_real * 2 * u * (3 * u_sq + w_sq) * sinc_w) * f_denom;
  double f1_real = (2 * u * e_2iu_real - e_iu_real * 2 * u * cos_w + e_iu_imag * (3 * u_sq - w_sq) * sinc_w) * f_denom;
  double f1_imag = (2 * u * e_2iu_imag + e_iu_imag * 2 * u * cos_w + e_iu_real * (3 * u_sq - w_sq) * sinc_w) * f_denom;
  double f2_real = (e_2iu_real - e_iu_real * cos_w - e_iu_imag * 3 * u * sinc_w) * f_denom;
  double f2_imag = (e_2iu_imag + e_iu_imag * cos_w - e_iu_real * 3 * u * sinc_w) * f_denom;
  complex<T> f[3];
  f[0] = {T(f0_real), T(f0_imag)};
  f[1] = {T(f1_real), T(f1_imag)};
  f[2] = {T(f2_real), T(f2_imag)};

  double sinc1_w = -1 / 3 + w_sq / 30 * (1 - w_sq / 260 * (1 - w_sq / 54 * (1 - w_sq / 88)));
  if (fabs(w) > 0.05) {
    sinc1_w = cos_w / w_sq - sin_w / (w_sq * w);
  }
  complex<T> r[3][3];
  complex<T> e_2iu = cos(2. * u) + I * sin(2. * u);
  complex<T> e__iu = cos(u) - I * sin(u);
  r[0][1] = 2. * (u + I * (u_sq - w_sq)) * e_2iu + 2. * e__iu * (4. * u * (2. - I * u) * cos(w) + I * (9. * u_sq + w_sq - I * u * (3. * u_sq + w_sq)) * sinc_w);
  r[0][2] = -2. * e_2iu + 2. * I * u * e__iu * (cos(w) + (1. + 4. * I * u) * sinc_w + 3. * u_sq * sinc1_w);
  r[1][1] = 2. * (1. + 2. * I * u) * e_2iu + e__iu * (-2. * (1. - I * u) * cos(w) + I * (6. * u + I * (w_sq - 3. * u_sq)) * sinc_w);
  r[1][2] = -I * e__iu * (cos(w) + (1. + 2. * I * u) * sinc_w - 3. * u_sq * sinc1_w);
  r[2][1] = 2. * I * e_2iu + I * e__iu * (cos(w) - 3. * (1. - I * u) * sinc_w);
  r[2][2] = e__iu * (sinc_w - 3. * I * u * sinc1_w);
  complex<T> b[3][3];
  double b_denom = 1 / (2 * (9 * u_sq - w_sq) * (9 * u_sq - w_sq));
  for (int j = 0; j < 3; ++j) {
    b[1][j] = (2 * u * r[j][1] + (3 * u_sq - w_sq) * r[j][2] - 2 * (15 * u_sq + w_sq) * f[j]) * b_denom;
    b[2][j] = (r[j][1] - 3 * u * r[j][2] - 24 * u * f[j]) * b_denom;
  }
  if (sig_parity) {
    f0_imag *= -1;
    f1_real *= -1;
    f2_imag *= -1;
  }
  f[0] = {T(f0_real), T(f0_imag)};
  f[1] = {T(f1_real), T(f1_imag)};
  f[2] = {T(f2_real), T(f2_imag)};
  if (sig_parity) {
    for (int i = 1; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        b[i][j].real(b[i][j].real() * (1 - 2 * (((i + j + 1) & 1))));
        b[i][j].imag(b[i][j].imag() * (2 * (((i + j + 1) & 1)) - 1));
      }
    }
  }
  ColorMatrix B1, B2;

  B1 = Q_sq * b[1][2] + Q * b[1][1] + b[1][0];
  B2 = Q_sq * b[2][2] + Q * b[2][1] + b[2][0];
  ColorMatrix S(sigma + index_from_pcoord(p_X, p_coord, mu, volume));
  ColorMatrix gamma = (Q * trace(S * B1 * U)) + (Q_sq * trace(S * B2 * U)) + (U * S * f[1]) + (Q * U * S * f[2]) + (U * S * Q * f[2]);
  ColorMatrix L = herm(gamma);
  for (int i = 0; i < Nc; ++i) {
    for (int j = 0; j < Nc; ++j) {
      lambda[index_from_pcoord(p_X, p_coord, mu, volume) + i * Nc + j] = L[i][j];
    }
  }
}

template <typename T, int DIM>
__global__ void stout_smear_reverse(
    // T *sigma,
    complex<T> *sigma,
    const complex<T> *lambda,
    const complex<T> *U_in,
    const T rho,
    const int Lx,
    const int Ly,
    const int Lz,
    const int Lt)
{
  const int volume = Lx * Ly * Lz * Lt / 2;
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int mu = blockIdx.y * blockDim.y + threadIdx.y;
  const int parity = blockIdx.z * blockDim.z + threadIdx.z;
  if (x >= volume || mu >= DIM || parity >= 2) {
    return;
  }
  const int p_X[Nd + 1] = {Lx / 2, Ly, Lz, Lt, 2};
  const int X[Nd] = {Lx, Ly, Lz, Lt};
  int p_coord[Nd + 1] = {x % p_X[0], x / p_X[0] % p_X[1], x / (p_X[0] * p_X[1]) % p_X[2], x / (p_X[0] * p_X[1] * p_X[2]) % p_X[3], parity};
  int coord[Nd] = {(p_coord[0] << 1) + (parity ^ ((p_coord[1] + p_coord[2] + p_coord[3]) & 1)), p_coord[1], p_coord[2], p_coord[3]};
  typedef Matrix<complex<T>> ColorMatrix;
  ColorMatrix U(U_in + index_from_pcoord(p_X, p_coord, mu, volume));
  ColorMatrix Q;
  for (int nu = 0; nu < DIM; ++nu) {
    if (nu != mu) {
      ColorMatrix staple1(U_in + index_from_pcoord(p_X, p_coord, nu, volume));
      coord[nu] = (coord[nu] + 1 + X[nu]) % X[nu];
      staple1 = staple1 * ColorMatrix(U_in + index_from_coord(p_X, coord, mu, volume));
      coord[mu] = (coord[mu] + 1 + X[mu]) % X[mu];
      coord[nu] = (coord[nu] - 1 + X[nu]) % X[nu];
      staple1 = staple1 * adjoint(ColorMatrix(U_in + index_from_coord(p_X, coord, nu, volume)));
      coord[mu] = (coord[mu] - 1 + X[mu]) % X[mu];

      coord[nu] = (coord[nu] - 1 + X[nu]) % X[nu];
      ColorMatrix staple2(U_in + index_from_coord(p_X, coord, nu, volume));
      staple2 = adjoint(staple2) * ColorMatrix(U_in + index_from_coord(p_X, coord, mu, volume));
      coord[mu] = (coord[mu] + 1 + X[mu]) % X[mu];
      staple2 = staple2 * ColorMatrix(U_in + index_from_coord(p_X, coord, nu, volume));
      coord[nu] = (coord[nu] + 1 + X[nu]) % X[nu];
      coord[mu] = (coord[mu] - 1 + X[mu]) % X[mu];

      Q += staple1 + staple2;
    }
  }
  ColorMatrix C = Q * rho;
  Q = i_herm(C * adjoint(U));

  ColorMatrix Q_sq = Q * Q;
  double c0 = trace(Q_sq * Q).real() / 3;
  double c1 = trace(Q_sq).real() / 2;
  double c0_max = 2 * c1 / 3 * sqrt(c1 / 3);
  int sig_parity = 0;
  if (c0 < 0) {
    sig_parity = 1;
    c0 *= -1;
  }
  c0_max = fmax(c0_max, 1e-15);
  c1 = fmax(c1, 1e-15);
  double theta = acos(c0 / c0_max);
  double u = sqrt(c1 / 3) * cos(theta / 3);
  double w = sqrt(c1) * sin(theta / 3);
  complex<T> I = {T(0), T(1)};
  double u_sq = u * u;
  double w_sq = w * w;
  double e_iu_real = cos(u);
  double e_iu_imag = sin(u);
  double e_2iu_real = cos(2 * u);
  double e_2iu_imag = sin(2 * u);
  double cos_w = cos(w);
  double sin_w = sin(w);
  double sinc_w = 1 - w_sq / 6 * (1 - w_sq / 20 * (1 - w_sq / 42 * (1 - w_sq / 72)));
  if (fabs(w) > 0.05) {
    sinc_w = sin(w) / w;
  }
  double f_denom = 1 / (9 * u_sq - w_sq);
  double f0_real = ((u_sq - w_sq) * e_2iu_real + e_iu_real * 8 * u_sq * cos_w + e_iu_imag * 2 * u * (3 * u_sq + w_sq) * sinc_w) * f_denom;
  double f0_imag = ((u_sq - w_sq) * e_2iu_imag - e_iu_imag * 8 * u_sq * cos_w + e_iu_real * 2 * u * (3 * u_sq + w_sq) * sinc_w) * f_denom;
  double f1_real = (2 * u * e_2iu_real - e_iu_real * 2 * u * cos_w + e_iu_imag * (3 * u_sq - w_sq) * sinc_w) * f_denom;
  double f1_imag = (2 * u * e_2iu_imag + e_iu_imag * 2 * u * cos_w + e_iu_real * (3 * u_sq - w_sq) * sinc_w) * f_denom;
  double f2_real = (e_2iu_real - e_iu_real * cos_w - e_iu_imag * 3 * u * sinc_w) * f_denom;
  double f2_imag = (e_2iu_imag + e_iu_imag * cos_w - e_iu_real * 3 * u * sinc_w) * f_denom;
  if (sig_parity) {
    f0_imag *= -1;
    f1_real *= -1;
    f2_imag *= -1;
  }
  complex<T> f[3];
  f[0] = {T(f0_real), T(f0_imag)};
  f[1] = {T(f1_real), T(f1_imag)};
  f[2] = {T(f2_real), T(f2_imag)};
  ColorMatrix S(sigma + index_from_pcoord(p_X, p_coord, mu, volume));
  ColorMatrix S_new = S * (Q_sq * f[2] + Q * f[1] + f[0]);

  S_new = S_new + adjoint(C) * ColorMatrix(lambda + index_from_pcoord(p_X, p_coord, mu, volume)) * I;
  ColorMatrix dU;
  for (int nu = 0; nu < DIM; ++nu) {
    if (nu != mu) {

      // dU = dU + dU1 + dU2 + dU3 - dU4 - dU5 + dU6;

      //  Staple 1
      //                                ( [ U_nu(x+mu) * U^+_mu(x+nu) ] U^+_nu(x) ) * Lambda_nu(x)
      //  Staple 5
      //            - Lambda_nu(x+mu) * ( [ U_nu(x+mu) * U^+_mu(x+nu) ] U^+_nu(x) )
      //  Staple 6
      //                                  [ U_nu(x+mu) * U^+_mu(x+nu) ] Lambda_mu(x + nu) * U^+_nu(x)

      coord[mu] = (coord[mu] + 1 + X[mu]) % X[mu];
      ColorMatrix tM1(U_in + index_from_coord(p_X, coord, nu, volume));
      ColorMatrix tL2(lambda + index_from_coord(p_X, coord, nu, volume)); // Lambda_nu(x + mu)
      coord[mu] = (coord[mu] - 1 + X[mu]) % X[mu];
      coord[nu] = (coord[nu] + 1 + X[nu]) % X[nu];
      tM1 *= adjoint(ColorMatrix(U_in + index_from_coord(p_X, coord, mu, volume))); // U_nu(x+mu) * U^+_mu(x+nu)
      ColorMatrix tL3(lambda + index_from_coord(p_X, coord, mu, volume));           // Lambda_mu(x + nu)
      coord[nu] = (coord[nu] - 1 + X[nu]) % X[nu];
      ColorMatrix tM2 = adjoint(ColorMatrix(U_in + index_from_coord(p_X, coord, nu, volume))); // U^+_nu(x)
      ColorMatrix tL1(lambda + index_from_coord(p_X, coord, nu, volume));                      // Lambda_nu(x)
      ColorMatrix dU156 = (tM1 * tM2 * tL1) - (tL2 * tM1 * tM2) + (tM1 * tL3 * tM2);

      // // Staple 2
      // //                U^+_nu(x-nu+mu) * [ U^+_mu(x-nu) * Lambda_mu(x-nu)    ] * U_nu(x-nu)
      // // Staple 3
      // //                U^+_nu(x-nu+mu) * [ Lambda_nu(x-nu+mu) * U^+_mu(x-nu) ] * U_nu(x-nu)
      // // Staple 4
      // //              - U^+_nu(x-nu+mu) * [ U^+_mu(x-nu) * Lambda_nu(x-nu)    ] * U_nu(x-nu)

      coord[nu] = (coord[nu] - 1 + X[nu]) % X[nu];
      tM1 = ColorMatrix(U_in + index_from_coord(p_X, coord, nu, volume));          // U_nu(x-nu)
      tM2 = adjoint(ColorMatrix(U_in + index_from_coord(p_X, coord, mu, volume))); // U^+_mu(x-nu)
      tL1 = ColorMatrix(lambda + index_from_coord(p_X, coord, mu, volume));        // Lambda_mu(x-nu)
      tL3 = ColorMatrix(lambda + index_from_coord(p_X, coord, nu, volume));        // Lambda_nu(x-nu)
      coord[mu] = (coord[mu] + 1 + X[mu]) % X[mu];
      ColorMatrix tM3 = adjoint(ColorMatrix(U_in + index_from_coord(p_X, coord, nu, volume))); // U^+_nu(x-nu+mu)
      tL2 = ColorMatrix(lambda + index_from_coord(p_X, coord, nu, volume));                    // Lambda_nu(x-nu+mu)
      coord[mu] = (coord[mu] - 1 + X[mu]) % X[mu];
      coord[nu] = (coord[nu] + 1 + X[nu]) % X[nu];
      ColorMatrix dU234 = tM3 * (tM2 * tL1 + tL2 * tM2 - tM2 * tL3) * tM1;

      dU += dU156 + dU234;
    }
  }
  dU = dU * rho;
  S_new = S_new - dU * I;
  for (int i = 0; i < Nc; ++i) {
    for (int j = 0; j < Nc; ++j) {
      sigma[index_from_pcoord(p_X, p_coord, mu, volume) + i * Nc + j] = S_new[i][j];
    }
  }
}
