#include <cupy/complex.cuh>

#define get_x(_coord, _X) \
  (((_coord[3] * _X[2] + _coord[2]) * _X[1] + _coord[1]) * _X[0] + _coord[0])

template <typename T, int Nc>
class Matrix {
private:
  T data[Nc][Nc];

public:
  __device__ __host__ Matrix()
  {
    for (int i = 0; i < Nc; ++i) {
      for (int j = 0; j < Nc; ++j) {
        data[i][j] = 0;
      }
    }
  }

  __device__ __host__ Matrix(const T *source)
  {
    for (int i = 0; i < Nc; ++i) {
      for (int j = 0; j < Nc; ++j) {
        data[i][j] = source[i * Nc + j];
      }
    }
  }

  __device__ __host__ Matrix(const Matrix<T, Nc> &matrix)
  {
    for (int i = 0; i < Nc; ++i) {
      for (int j = 0; j < Nc; ++j) {
        data[i][j] = matrix[i][j];
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

  __device__ __host__ void operator=(const Matrix<T, Nc> &rhs)
  {
    for (int i = 0; i < Nc; ++i) {
      for (int j = 0; j < Nc; ++j) {
        data[i][j] = rhs[i][j];
      }
    }
  }

  __device__ __host__ Matrix<T, Nc> operator+(const T &rhs)
  {
    Matrix<T, Nc> result;
    for (int i = 0; i < Nc; ++i) {
      for (int j = 0; j < Nc; ++j) {
        result[i][j] = data[i][j];
      }
      result[i][i] += rhs;
    }
    return result;
  }

  __device__ __host__ Matrix<T, Nc> operator+(const Matrix<T, Nc> &rhs)
  {
    Matrix<T, Nc> result;
    for (int i = 0; i < Nc; ++i) {
      for (int j = 0; j < Nc; ++j) {
        result[i][j] = data[i][j] + rhs[i][j];
      }
    }
    return result;
  }

  __device__ __host__ Matrix<T, Nc> operator-(const Matrix<T, Nc> &rhs)
  {
    Matrix<T, Nc> result;
    for (int i = 0; i < Nc; ++i) {
      for (int j = 0; j < Nc; ++j) {
        result[i][j] = data[i][j] - rhs[i][j];
      }
    }
    return result;
  }

  __device__ __host__ void operator+=(const Matrix<T, Nc> &rhs)
  {
    for (int i = 0; i < Nc; ++i) {
      for (int j = 0; j < Nc; ++j) {
        data[i][j] += rhs[i][j];
      }
    }
  }

  __device__ __host__ void operator-=(const T &rhs)
  {
    for (int i = 0; i < Nc; ++i) {
      data[i][i] -= rhs;
    }
  }

  __device__ __host__ Matrix<T, Nc> operator*(const T &rhs)
  {
    Matrix<T, Nc> result;
    for (int i = 0; i < Nc; ++i) {
      for (int j = 0; j < Nc; ++j) {
        result[i][j] = data[i][j] * rhs;
      }
    }
    return result;
  }

  __device__ __host__ Matrix<T, Nc> operator*(const Matrix<T, Nc> &rhs)
  {
    Matrix<T, Nc> result;
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
};

template <typename T, int Nc>
__device__ __host__ T trace(const Matrix<T, Nc> &matrix)
{
  T result = 0;
  for (int i = 0; i < Nc; ++i) {
    result += matrix[i][i];
  }
  return result;
}

template <typename T, int Nc>
__device__ __host__ Matrix<T, Nc> adjoint(const Matrix<T, Nc> &matrix)
{
  Matrix<T, Nc> result;
  for (int i = 0; i < Nc; ++i) {
    for (int j = 0; j < Nc; ++j) {
      result[i][j] = conj(matrix[j][i]);
    }
  }
  return result;
}

template <typename T, int Nc>
__device__ __host__ Matrix<T, Nc> antiherm(const Matrix<T, Nc> &matrix)
{
  Matrix<T, Nc> result;
  for (int i = 0; i < Nc; ++i) {
    for (int j = 0; j < Nc; ++j) {
      result[i][j].real((matrix[j][i].imag() + matrix[i][j].imag()) / 2.);
      result[i][j].imag((matrix[j][i].real() - matrix[i][j].real()) / 2.);
    }
  }
  result -= trace(result) / T(Nc);
  return result;
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
  const int Nd = 4;
  const int Nc = 3;

  const int volume = Lx * Ly * Lz * Lt;
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int mu = blockIdx.y * blockDim.y + threadIdx.y;
  if (mu >= DIM || x >= volume) {
    return;
  }
  const int X[Nd] = {Lx, Ly, Lz, Lt};
  int coord[Nd] = {x % Lx, x / Lx % Ly, x / (Lx * Ly) % Lz, x / (Lx * Ly * Lz) % Lt};
  typedef Matrix<complex<T>, Nc> ColorMatrix;
  ColorMatrix U(U_in + (mu * volume + x) * Nc * Nc);
  ColorMatrix Q;
  for (int nu = 0; nu < DIM; ++nu) {
    if (nu != mu) {
      ColorMatrix staple1(U_in + (nu * volume + get_x(coord, X)) * Nc * Nc);
      coord[nu] = (coord[nu] + 1 + X[nu]) % X[nu];
      staple1 = staple1 * ColorMatrix(U_in + (mu * volume + get_x(coord, X)) * Nc * Nc);
      coord[mu] = (coord[mu] + 1 + X[mu]) % X[mu];
      coord[nu] = (coord[nu] - 1 + X[nu]) % X[nu];
      staple1 = staple1 * adjoint(ColorMatrix(U_in + (nu * volume + get_x(coord, X)) * Nc * Nc));
      coord[mu] = (coord[mu] - 1 + X[mu]) % X[mu];

      coord[nu] = (coord[nu] - 1 + X[nu]) % X[nu];
      ColorMatrix staple2(U_in + (nu * volume + get_x(coord, X)) * Nc * Nc);
      staple2 = adjoint(staple2) * ColorMatrix(U_in + (mu * volume + get_x(coord, X)) * Nc * Nc);
      coord[mu] = (coord[mu] + 1 + X[mu]) % X[mu];
      staple2 = staple2 * ColorMatrix(U_in + (nu * volume + get_x(coord, X)) * Nc * Nc);
      coord[nu] = (coord[nu] + 1 + X[nu]) % X[nu];
      coord[mu] = (coord[mu] - 1 + X[mu]) % X[mu];

      Q += staple1 + staple2;
    }
  }
  Q = antiherm((Q * rho) * adjoint(U));

  ColorMatrix Q_sq = Q * Q;
  double c0 = trace(Q_sq * Q).real() / 3;
  double c1 = trace(Q_sq).real() / 2;
  double c0_max = 2 * c1 / 3 * sqrt(c1 / 3);
  int parity = 0;
  if (c0 < 0) {
    parity = 1;
    c0 *= -1;
  }
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
  if (abs(w) > 0.05) {
    sinc_w = sin(w) / w;
  }
  double f_denom = 1 / (9 * u_sq - w_sq);
  double f0_real = ((u_sq - w_sq) * e_2iu_real + e_iu_real * 8 * u_sq * cos_w + e_iu_imag * 2 * u * (3 * u_sq + w_sq) * sinc_w) * f_denom;
  double f0_imag = ((u_sq - w_sq) * e_2iu_imag - e_iu_imag * 8 * u_sq * cos_w + e_iu_real * 2 * u * (3 * u_sq + w_sq) * sinc_w) * f_denom;
  double f1_real = (2 * u * e_2iu_real - e_iu_real * 2 * u * cos_w + e_iu_imag * (3 * u_sq - w_sq) * sinc_w) * f_denom;
  double f1_imag = (2 * u * e_2iu_imag + e_iu_imag * 2 * u * cos_w + e_iu_real * (3 * u_sq - w_sq) * sinc_w) * f_denom;
  double f2_real = (e_2iu_real - e_iu_real * cos_w - e_iu_imag * 3 * u * sinc_w) * f_denom;
  double f2_imag = (e_2iu_imag + e_iu_imag * cos_w - e_iu_real * 3 * u * sinc_w) * f_denom;
  if (parity) {
    f0_imag *= -1;
    f1_real *= -1;
    f2_imag *= -1;
  }
  complex<T> f0 = {T(f0_real), T(f0_imag)};
  complex<T> f1 = {T(f1_real), T(f1_imag)};
  complex<T> f2 = {T(f2_real), T(f2_imag)};
  ColorMatrix e_iQ = Q_sq * f2 + Q * f1 + f0;
  U = e_iQ * U;
  for (int i = 0; i < Nc; ++i) {
    for (int j = 0; j < Nc; ++j) {
      U_out[(mu * volume + x) * Nc * Nc + i * Nc + j] = U[i][j];
    }
  }
}

template <typename T, int DIM>
__global__ void compute_lambda_kernel(
    const complex<T> *sigma,
    complex<T> *lambda,
    const complex<T> *U_in,
    const T rho,
    const int Lx,
    const int Ly,
    const int Lz,
    const int Lt)
{
  const int Nd = 4;
  const int Nc = 3;

  const int volume = Lx * Ly * Lz * Lt;
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int mu = blockIdx.y * blockDim.y + threadIdx.y;
  if (mu >= DIM || x >= volume) {
    return;
  }
  const int X[Nd] = {Lx, Ly, Lz, Lt};
  int coord[Nd] = {x % Lx, x / Lx % Ly, x / (Lx * Ly) % Lz, x / (Lx * Ly * Lz) % Lt};
  typedef Matrix<complex<T>, Nc> ColorMatrix;
  ColorMatrix S(sigma + (mu * volume + x) * Nc * Nc);
  ColorMatrix U(U_in + (mu * volume + x) * Nc * Nc);
  ColorMatrix Q;
  for (int nu = 0; nu < DIM; ++nu) {
    if (nu != mu) {
      ColorMatrix staple1(U_in + (nu * volume + get_x(coord, X)) * Nc * Nc);
      coord[nu] = (coord[nu] + 1 + X[nu]) % X[nu];
      staple1 = staple1 * ColorMatrix(U_in + (mu * volume + get_x(coord, X)) * Nc * Nc);
      coord[mu] = (coord[mu] + 1 + X[mu]) % X[mu];
      coord[nu] = (coord[nu] - 1 + X[nu]) % X[nu];
      staple1 = staple1 * adjoint(ColorMatrix(U_in + (nu * volume + get_x(coord, X)) * Nc * Nc));
      coord[mu] = (coord[mu] - 1 + X[mu]) % X[mu];

      coord[nu] = (coord[nu] - 1 + X[nu]) % X[nu];
      ColorMatrix staple2(U_in + (nu * volume + get_x(coord, X)) * Nc * Nc);
      staple2 = adjoint(staple2) * ColorMatrix(U_in + (mu * volume + get_x(coord, X)) * Nc * Nc);
      coord[mu] = (coord[mu] + 1 + X[mu]) % X[mu];
      staple2 = staple2 * ColorMatrix(U_in + (nu * volume + get_x(coord, X)) * Nc * Nc);
      coord[nu] = (coord[nu] + 1 + X[nu]) % X[nu];
      coord[mu] = (coord[mu] - 1 + X[mu]) % X[mu];

      Q += staple1 + staple2;
    }
  }
  Q = antiherm((Q * rho) * adjoint(U));

  ColorMatrix Q_sq = Q * Q;
  double c0 = trace(Q_sq * Q).real() / 3;
  double c1 = trace(Q_sq).real() / 2;
  double c0_max = 2 * c1 / 3 * sqrt(c1 / 3);
  int parity = 0;
  if (c0 < 0) {
    parity = 1;
    c0 *= -1;
  }
  double theta = acos(c0 / c0_max);
  double u = sqrt(c1 / 3) * cos(theta / 3);
  double w = sqrt(c1) * sin(theta / 3);
  double u_sq = u * u;
  double w_sq = w * w;
  double e_iu_real = cos(u);
  double e_iu_imag = sin(u);
  double e_2iu_real = cos(2 * u);
  double e_2iu_imag = sin(2 * u);
  double sin_w = sin(w);
  double cos_w = cos(w);
  double sinc_w = 1 - w_sq / 6 * (1 - w_sq / 20 * (1 - w_sq / 42 * (1 - w_sq / 72)));
  if (abs(w) > 0.05) {
    sinc_w = sin(w) / w;
  }
  double f_denom = 1 / (9 * u_sq - w_sq);
  double f0_real = ((u_sq - w_sq) * e_2iu_real + e_iu_real * 8 * u_sq * cos_w + e_iu_imag * 2 * u * (3 * u_sq + w_sq) * sinc_w) * f_denom;
  double f0_imag = ((u_sq - w_sq) * e_2iu_imag - e_iu_imag * 8 * u_sq * cos_w + e_iu_real * 2 * u * (3 * u_sq + w_sq) * sinc_w) * f_denom;
  double f1_real = (2 * u * e_2iu_real - e_iu_real * 2 * u * cos_w + e_iu_imag * (3 * u_sq - w_sq) * sinc_w) * f_denom;
  double f1_imag = (2 * u * e_2iu_imag + e_iu_imag * 2 * u * cos_w + e_iu_real * (3 * u_sq - w_sq) * sinc_w) * f_denom;
  double f2_real = (e_2iu_real - e_iu_real * cos_w - e_iu_imag * 3 * u * sinc_w) * f_denom;
  double f2_imag = (e_2iu_imag + e_iu_imag * cos_w - e_iu_real * 3 * u * sinc_w) * f_denom;
  if (parity) {
    f0_imag *= -1;
    f1_real *= -1;
    f2_imag *= -1;
  }
  complex<T> f[3];
  f[0] = {T(f0_real), T(f0_imag)};
  f[1] = {T(f1_real), T(f1_imag)};
  f[2] = {T(f2_real), T(f2_imag)};

  double sinc0_w = sin_w / w;
  double sinc1_w = cos_w / w_sq - sin_w / (w_sq * w);
  double r_real[3][3], r_imag[3][3];
  double r_denom = 1 / 2 * sqrt(9 * u_sq - w_sq);
  r_real[0][1] = (2 * u * e_2iu_real - 2 * (u_sq - w_sq) * e_2iu_imag + 16 * u * cos_w * e_iu_real - 8 * u_sq * cos_w * e_iu_imag + (2 * e_iu_imag * (9 * u_sq + w_sq) + 2 * e_iu_real * u * (3 * u_sq + w_sq)) * sinc0_w) * r_denom;
  r_imag[0][1] = (2 * u * e_2iu_imag + 2 * (u_sq - w_sq) * e_2iu_real - 16 * u * cos_w * e_iu_imag - 8 * u_sq * cos_w * e_iu_real + (2 * e_iu_real * (9 * u_sq + w_sq) - 2 * e_iu_imag * u * (3 * u_sq + w_sq)) * sinc0_w) * r_denom;
  r_real[1][1] = (2 * e_iu_real - 4 * u * e_iu_imag - 2 * cos_w * e_iu_real + 2 * u * cos_w * e_iu_imag + e_iu_imag * 6 * u * sinc0_w - e_iu_real * (w_sq - 3 * u_sq) * sinc0_w) * r_denom;
  r_imag[1][1] = (2 * e_iu_imag + 4 * u * e_iu_real + 2 * cos_w * e_iu_imag + 2 * u * cos_w * e_iu_real + e_iu_real * 6 * u * sinc0_w + e_iu_imag * (w_sq - 3 * u_sq) * sinc0_w) * r_denom;
  r_real[2][1] = (e_iu_imag * cos_w - 2 * e_2iu_imag - (3 * e_iu_imag + 3 * u * e_iu_real) * sinc0_w) * r_denom;
  r_imag[2][1] = (e_iu_real * cos_w + 2 * e_2iu_real - (3 * e_iu_real - 3 * u * e_iu_imag) * sinc0_w) * r_denom;

  r_real[0][2] = (-2 * e_2iu_real + 2 * u * (cos_w + sinc0_w + 3 * u_sq * sinc1_w) * e_iu_imag - 8 * u * sinc0_w * e_iu_real) * r_denom;
  r_imag[0][2] = (-2 * e_2iu_imag + 2 * u * (cos_w + sinc0_w + 3 * u_sq * sinc1_w) * e_iu_real + 8 * u * sinc0_w * e_iu_imag) * r_denom;
  r_real[1][2] = (-1 * (cos_w + sinc0_w - 3 * u_sq * sinc1_w) * e_2iu_imag + 2 * u * sinc0_w * e_iu_real) * r_denom;
  r_imag[1][2] = (-1 * (cos_w + sinc0_w - 3 * u_sq * sinc1_w) * e_2iu_real - 2 * u * sinc0_w * e_iu_imag) * r_denom;
  r_real[2][2] = (-3 * u * sinc1_w * e_iu_imag + sinc0_w * e_iu_real) * r_denom;
  r_imag[2][2] = (-3 * u * sinc1_w * e_iu_real - sinc0_w * e_iu_imag) * r_denom;

  if (parity) {
    for (int i = 1; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        r_real[i][j] *= 1 - 2 * (((i + j + 1) & 1));
        r_imag[i][j] *= 2 * (((i + j + 1) & 1)) - 1;
      }
    }
  }

  complex<T> r[3][3];
  for (int i = 0; i < 3; ++i) {
    for (int j = 1; j < 3; ++j) {
      r[i][j] = {T(r_real[i][j]), T(r_imag[i][j])};
    }
  }

  complex<T> b[3][3];
  for (int j = 0; j < 3; ++j) {
    b[1][j] = 2 * u * r[j][1] + (3 * u_sq - w_sq) * r[j][2] - 2 * (15 * u_sq - w_sq) * f[j];
    b[2][j] = r[j][1] - 3 * u * r[j][2] - 24 * u * f[j];
  }
  ColorMatrix B1, B2;

  B1 = Q_sq * b[1][2] + Q * b[1][1] + b[1][0];
  B2 = Q_sq * b[2][2] + Q * b[2][1] + b[2][0];
  ColorMatrix gamma = Q * trace(S * B1 * U) + Q_sq * trace(S * B2 * U) + U * S * f[1] + Q * U * S * f[2] + U * S * Q * f[2];
  ColorMatrix L = antiherm(gamma);

  for (int i = 0; i < Nc; ++i) {
    for (int j = 0; j < Nc; ++j) {
      lambda[(mu * volume + x) * Nc * Nc + i * Nc + j] = L[i][j];
    }
  }
}

template <typename T, int DIM>
__global__ void stout_smear_reverse(
    complex<T> *sigma,
    const complex<T> *lambda,
    const complex<T> *U_in,
    const T rho,
    const int Lx,
    const int Ly,
    const int Lz,
    const int Lt)
{
  const int Nd = 4;
  const int Nc = 3;

  const int volume = Lx * Ly * Lz * Lt;
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int mu = blockIdx.y * blockDim.y + threadIdx.y;
  if (mu >= Nd - 1 || x >= volume) {
    return;
  }
  const int X[Nd] = {Lx, Ly, Lz, Lt};
  int coord[Nd] = {x % Lx, x / Lx % Ly, x / (Lx * Ly) % Lz, x / (Lx * Ly * Lz) % Lt};
  typedef Matrix<complex<T>, Nc> ColorMatrix;
  ColorMatrix U(U_in + (mu * volume + x) * Nc * Nc);
  ColorMatrix Q;
  for (int nu = 0; nu < DIM; ++nu) {
    if (nu != mu) {
      ColorMatrix staple1(U_in + (nu * volume + get_x(coord, X)) * Nc * Nc);
      coord[nu] = (coord[nu] + 1 + X[nu]) % X[nu];
      staple1 = staple1 * ColorMatrix(U_in + (mu * volume + get_x(coord, X)) * Nc * Nc);
      coord[mu] = (coord[mu] + 1 + X[mu]) % X[mu];
      coord[nu] = (coord[nu] - 1 + X[nu]) % X[nu];
      staple1 = staple1 * adjoint(ColorMatrix(U_in + (nu * volume + get_x(coord, X)) * Nc * Nc));
      coord[mu] = (coord[mu] - 1 + X[mu]) % X[mu];

      coord[nu] = (coord[nu] - 1 + X[nu]) % X[nu];
      ColorMatrix staple2(U_in + (nu * volume + get_x(coord, X)) * Nc * Nc);
      staple2 = adjoint(staple2) * ColorMatrix(U_in + (mu * volume + get_x(coord, X)) * Nc * Nc);
      coord[mu] = (coord[mu] + 1 + X[mu]) % X[mu];
      staple2 = staple2 * ColorMatrix(U_in + (nu * volume + get_x(coord, X)) * Nc * Nc);
      coord[nu] = (coord[nu] + 1 + X[nu]) % X[nu];
      coord[mu] = (coord[mu] - 1 + X[mu]) % X[mu];

      Q += staple1 + staple2;
    }
  }
  Q = antiherm((Q * rho) * adjoint(U));

  ColorMatrix Q_sq = Q * Q;
  double c0 = trace(Q_sq * Q).real() / 3;
  double c1 = trace(Q_sq).real() / 2;
  double c0_max = 2 * c1 / 3 * sqrt(c1 / 3);
  int parity = 0;
  if (c0 < 0) {
    parity = 1;
    c0 *= -1;
  }
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
  double sin_w = sin(w);
  double sinc_w = 1 - w_sq / 6 * (1 - w_sq / 20 * (1 - w_sq / 42 * (1 - w_sq / 72)));
  if (abs(w) > 0.05) {
    sinc_w = sin(w) / w;
  }
  double f_denom = 1 / (9 * u_sq - w_sq);
  double f0_real = ((u_sq - w_sq) * e_2iu_real + e_iu_real * 8 * u_sq * cos_w + e_iu_imag * 2 * u * (3 * u_sq + w_sq) * sinc_w) * f_denom;
  double f0_imag = ((u_sq - w_sq) * e_2iu_imag - e_iu_imag * 8 * u_sq * cos_w + e_iu_real * 2 * u * (3 * u_sq + w_sq) * sinc_w) * f_denom;
  double f1_real = (2 * u * e_2iu_real - e_iu_real * 2 * u * cos_w + e_iu_imag * (3 * u_sq - w_sq) * sinc_w) * f_denom;
  double f1_imag = (2 * u * e_2iu_imag + e_iu_imag * 2 * u * cos_w + e_iu_real * (3 * u_sq - w_sq) * sinc_w) * f_denom;
  double f2_real = (e_2iu_real - e_iu_real * cos_w - e_iu_imag * 3 * u * sinc_w) * f_denom;
  double f2_imag = (e_2iu_imag + e_iu_imag * cos_w - e_iu_real * 3 * u * sinc_w) * f_denom;
  if (parity) {
    f0_imag *= -1;
    f1_real *= -1;
    f2_imag *= -1;
  }
  complex<T> f[3];
  f[0] = {T(f0_real), T(f0_imag)};
  f[1] = {T(f1_real), T(f1_imag)};
  f[2] = {T(f2_real), T(f2_imag)};

  double sinc0_w = sin_w / w;
  double sinc1_w = cos_w / w_sq - sin_w / (w_sq * w);
  double r_real[3][3], r_imag[3][3];

  r_real[0][1] = 2 * u * e_2iu_real - 2 * (u_sq - w_sq) * e_2iu_imag + 16 * u * cos_w * e_iu_real - 8 * u_sq * cos_w * e_iu_imag + (2 * e_iu_imag * (9 * u_sq + w_sq) + 2 * e_iu_real * u * (3 * u_sq + w_sq)) * sinc0_w;
  r_imag[0][1] = 2 * u * e_2iu_imag + 2 * (u_sq - w_sq) * e_2iu_real - 16 * u * cos_w * e_iu_imag - 8 * u_sq * cos_w * e_iu_real + (2 * e_iu_real * (9 * u_sq + w_sq) - 2 * e_iu_imag * u * (3 * u_sq + w_sq)) * sinc0_w;
  r_real[1][1] = 2 * e_iu_real - 4 * u * e_iu_imag - 2 * cos_w * e_iu_real + 2 * u * cos_w * e_iu_imag + e_iu_imag * 6 * u * sinc0_w - e_iu_real * (w_sq - 3 * u_sq) * sinc0_w;
  r_imag[1][1] = 2 * e_iu_imag + 4 * u * e_iu_real + 2 * cos_w * e_iu_imag + 2 * u * cos_w * e_iu_real + e_iu_real * 6 * u * sinc0_w + e_iu_imag * (w_sq - 3 * u_sq) * sinc0_w;
  r_real[2][1] = e_iu_imag * cos_w - 2 * e_2iu_imag - (3 * e_iu_imag + 3 * u * e_iu_real) * sinc0_w;
  r_imag[2][1] = e_iu_real * cos_w + 2 * e_2iu_real - (3 * e_iu_real - 3 * u * e_iu_imag) * sinc0_w;

  r_real[0][2] = -2 * e_2iu_real + 2 * u * (cos_w + sinc0_w + 3 * u_sq * sinc1_w) * e_iu_imag - 8 * u * sinc0_w * e_iu_real;
  r_imag[0][2] = -2 * e_2iu_imag + 2 * u * (cos_w + sinc0_w + 3 * u_sq * sinc1_w) * e_iu_real + 8 * u * sinc0_w * e_iu_imag;
  r_real[1][2] = -1 * (cos_w + sinc0_w - 3 * u_sq * sinc1_w) * e_2iu_imag + 2 * u * sinc0_w * e_iu_real;
  r_imag[1][2] = -1 * (cos_w + sinc0_w - 3 * u_sq * sinc1_w) * e_2iu_real - 2 * u * sinc0_w * e_iu_imag;
  r_real[2][2] = -3 * u * sinc1_w * e_iu_imag + sinc0_w * e_iu_real;
  r_imag[2][2] = -3 * u * sinc1_w * e_iu_real - sinc0_w * e_iu_imag;

  if (parity) {
    for (int i = 1; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        r_real[i][j] *= 1 - 2 * (((i + j + 1) & 1));
        r_imag[i][j] *= 2 * (((i + j + 1) & 1)) - 1;
      }
    }
  }

  complex<T> r[3][3];
  for (int i = 0; i < 3; ++i) {
    for (int j = 1; j < 3; ++j) {
      r[i][j] = {T(r_real[i][j]), T(r_imag[i][j])};
    }
  }

  double b_denom = 1 / 2 * sqrt(9 * u_sq - w_sq);
  complex<T> b[3][3];
  for (int j = 0; j < 3; ++j) {
    b[1][j] = (2 * u * r[j][1] + (3 * u_sq - w_sq) * r[j][2] - 2 * (15 * u_sq - w_sq) * f[j]) * b_denom;
    b[2][j] = (r[j][1] - 3 * u * r[j][2] - 24 * u * f[j]) * b_denom;
  }
  ColorMatrix B1, B2;
  B1 = Q_sq * b[1][2] + Q * b[1][1] + b[1][0];
  B2 = Q_sq * b[2][2] + Q * b[2][1] + b[2][0];
  ColorMatrix S(sigma);
  ColorMatrix S_new = S * (Q_sq * f[2] + Q * f[1] + f[0]);
  complex<T> i = {0, 1};
  S_new = S_new + adjoint(Q) * ColorMatrix(lambda + (mu * volume + get_x(coord, X))) * i;
  ColorMatrix dU;
  for (int nu = 0; nu < Nd - 1; ++nu) {
    if (nu != mu) {
      coord[mu] = (coord[mu] + 1 + X[mu]) % X[mu];
      ColorMatrix dU1(U_in + (nu * volume + get_x(coord, X)) * Nc * Nc);
      coord[mu] = (coord[mu] - 1 + X[mu]) % X[mu];
      coord[nu] = (coord[nu] + 1 + X[nu]) % X[nu];
      dU1 = dU1 * adjoint(ColorMatrix(U_in + (mu * volume + get_x(coord, X)) * Nc * Nc));
      coord[nu] = (coord[nu] - 1 + X[nu]) % X[nu];
      dU1 = dU1 * adjoint(ColorMatrix(U_in + (nu * volume + get_x(coord, X)) * Nc * Nc));
      dU1 = dU1 * ColorMatrix(lambda + (nu * volume + get_x(coord, X)) * Nc * Nc);

      coord[nu] = (coord[nu] - 1 + X[nu]) % X[nu];
      coord[mu] = (coord[mu] + 1 + X[mu]) % X[mu];
      ColorMatrix dU2(U_in + (nu * volume + get_x(coord, X)) * Nc * Nc);
      coord[mu] = (coord[mu] - 1 + X[mu]) % X[mu];
      dU2 = adjoint(dU2) * adjoint(ColorMatrix(U_in + (mu * volume + get_x(coord, X)) * Nc * Nc));
      dU2 = dU2 * ColorMatrix(lambda + (nu * volume + get_x(coord, X)) * Nc * Nc);
      dU2 = dU2 * ColorMatrix(U_in + (nu * volume + get_x(coord, X)) * Nc * Nc);

      coord[mu] = (coord[mu] + 1 + X[mu]) % X[mu];
      ColorMatrix dU3(U_in + (nu * volume + get_x(coord, X)) * Nc * Nc);
      dU3 = adjoint(dU3) * ColorMatrix(lambda + (nu * volume + get_x(coord, X)) * Nc * Nc);
      coord[mu] = (coord[mu] - 1 + X[mu]) % X[mu];
      dU3 = dU3 * adjoint(ColorMatrix(U_in + (mu * volume + get_x(coord, X)) * Nc * Nc));
      dU3 = dU3 * ColorMatrix(U_in + (nu * volume + get_x(coord, X)) * Nc * Nc);

      coord[mu] = (coord[mu] + 1 + X[mu]) % X[mu];
      ColorMatrix dU4(U_in + (nu * volume + get_x(coord, X)) * Nc * Nc);
      coord[mu] = (coord[mu] - 1 + X[mu]) % X[mu];
      dU4 = adjoint(dU4) * adjoint(ColorMatrix(U_in + (mu * volume + get_x(coord, X)) * Nc * Nc));
      dU4 = dU4 * ColorMatrix(lambda + (nu * volume + get_x(coord, X)) * Nc * Nc);
      dU4 = dU4 * ColorMatrix(U_in + (nu * volume + get_x(coord, X)) * Nc * Nc);

      coord[nu] = (coord[nu] + 1 + X[nu]) % X[nu];
      coord[mu] = (coord[mu] + 1 + X[mu]) % X[mu];
      ColorMatrix dU5 = ColorMatrix(lambda + (nu * volume + get_x(coord, X)) * Nc * Nc);
      dU5 = dU5 * ColorMatrix(U_in + (nu * volume + get_x(coord, X)) * Nc * Nc);
      coord[mu] = (coord[mu] - 1 + X[mu]) % X[mu];
      coord[nu] = (coord[nu] + 1 + X[nu]) % X[nu];
      dU5 = dU5 * adjoint(ColorMatrix(U_in + (mu * volume + get_x(coord, X)) * Nc * Nc));
      coord[nu] = (coord[nu] - 1 + X[nu]) % X[nu];
      dU5 = dU5 * adjoint(ColorMatrix(U_in + (nu * volume + get_x(coord, X)) * Nc * Nc));

      coord[mu] = (coord[mu] + 1 + X[mu]) % X[mu];
      ColorMatrix dU6(U_in + (nu * volume + get_x(coord, X)) * Nc * Nc);
      coord[mu] = (coord[mu] - 1 + X[mu]) % X[mu];
      coord[nu] = (coord[nu] + 1 + X[nu]) % X[nu];
      dU6 = dU6 * adjoint(ColorMatrix(U_in + (nu * volume + get_x(coord, X)) * Nc * Nc));
      dU6 = dU6 * ColorMatrix(lambda + (nu * volume + get_x(coord, X)) * Nc * Nc);
      coord[nu] = (coord[nu] - 1 + X[nu]) % X[nu];
      dU6 = dU6 * adjoint(ColorMatrix(U_in + (nu * volume + get_x(coord, X)) * Nc * Nc));

      dU = dU + dU1 + dU2 + dU3 - dU4 - dU5 + dU6;
    }
  }
  dU = dU * rho;
  S_new = S_new - dU * i;
  for (int i = 0; i < Nc; ++i) {
    for (int j = 0; j < Nc; ++j) {
      sigma[(mu * volume + x) * Nc * Nc + i * Nc + j] = S_new[i][j];
    }
  }
}