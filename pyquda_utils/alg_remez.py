# This file is a Python implementation of https://github.com/maddyscientist/AlgRemez
# LU decomposition is copied from https://github.com/mpmath/mpmath

import math

import gmpy2
from gmpy2 import mpfr


def LU_decomp(A):
    tol = gmpy2.fsum([gmpy2.fsum([abs(Aij) for Aij in Ai]) for Ai in A]) * gmpy2.exp2(
        1 - gmpy2.get_context().precision
    )  # each pivot element has to be bigger
    n = len(A)
    p = [None for _ in range(n - 1)]
    for j in range(n - 1):
        # pivoting, choose max(abs(reciprocal row sum)*abs(pivot element))
        biggest = 0
        for k in range(j, n):
            s = gmpy2.fsum([abs(Akl) for Akl in A[k][j:n]])
            if s <= tol:
                raise ZeroDivisionError("matrix is numerically singular")
            elif gmpy2.is_nan(s):
                raise ValueError("matrix contains nans")
            current = abs(A[k][j]) / s
            if current > biggest:  # TODO: what if equal?
                biggest = current
                p[j] = k
        # swap rows according to p
        A[j], A[p[j]] = A[p[j]], A[j]
        if abs(A[j][j]) <= tol:
            raise ZeroDivisionError("matrix is numerically singular")
        # calculate elimination factors and add rows
        for i in range(j + 1, n):
            A[i][j] /= A[j][j]
            for k in range(j + 1, n):
                A[i][k] -= A[i][j] * A[j][k]
    if abs(A[n - 1][n - 1]) <= tol:
        raise ZeroDivisionError("matrix is numerically singular")
    return p


def L_solve(L, b, p=None):
    """
    Solve the lower part of a LU factorized matrix for y.
    The diagonal of L is assumed to be 1.

    b may be a vector or matrix.
    """
    n = len(L)
    if p:  # swap b according to p
        for k in range(len(p)):
            b[k], b[p[k]] = b[p[k]], b[k]
    # solve
    for i in range(n):
        for j in range(i):
            b[i] -= L[i][j] * b[j]


def U_solve(U, y):
    """
    Solve the upper part of a LU factorized matrix for x.

    y may be a vector or matrix.
    """
    n = len(U)
    for i in range(n - 1, -1, -1):
        for j in range(i + 1, n):
            y[i] -= U[i][j] * y[j]
        y[i] /= U[i][i]


def lu_solve(A, b):
    """
    Ax = b => LUx = b => Ly = b, Ux = y

    b may be a vector or matrix.

    Solve a determined or overdetermined linear equations system.
    Fast LU decomposition is used, which is less accurate than QR decomposition
    (especially for overdetermined systems), but it's twice as efficient.
    Use qr_solve if you want more precision or have to solve a very ill-
    conditioned system.

    If you specify real=True, it does not check for overdeterminded complex
    systems.
    """
    # LU factorization
    p = LU_decomp(A)
    L_solve(A, b, p)
    U_solve(A, b)


class AlgRemez:
    def __init__(self, lower_bound, upper_bound, precision):
        gmpy2.get_context().precision = round((precision + 1) * 3.3219280948873626)

        self.left = mpfr(lower_bound)
        self.right = mpfr(upper_bound)
        self.width = self.right - self.left

        print(f"Approximation bounds are [{self.left},{self.right}]")
        print(f"Precision of arithmetic is {precision}")

        self.n = 0
        self.d = 0

        self.tolerance = mpfr(1e-15)

    def allocate(self, num_degree, den_degree):
        self.param = [None for _ in range(num_degree + den_degree + 1)]
        self.roots = [None for _ in range(num_degree)]
        self.poles = [None for _ in range(den_degree)]
        self.xx = [None for _ in range(num_degree + den_degree + 3)]
        self.mm = [None for _ in range(num_degree + den_degree + 2)]

    def generateApprox(self, num_degree, den_degree, func):
        print(f"Degree of the approximation is ({num_degree},{den_degree})")
        print(func)

        if num_degree != self.n or den_degree != self.d:
            self.allocate(num_degree, den_degree)

        step = [None for _ in range(num_degree + den_degree + 2)]

        self.func = func
        self.spread = mpfr(1.0e37)

        self.n = num_degree
        self.d = den_degree
        self.neq = self.n + self.d + 1

        self.initialGuess()
        self.stpini(step)

        iter = 0
        while self.spread > self.tolerance:
            if iter % 100 == 0:
                print(f"Iteration {iter}, spread {float(self.spread)} delta {float(self.delta)}")
            iter += 1

            self.equations()
            if self.delta < self.tolerance:
                print("Delta too small, try increasing precision")
                exit(0)

            self.search(step)

        error, sign = self.getErr(self.mm[0])
        print(f"Converged at {iter} iterations, error = {error:18.16e}")

        if not self.root():
            print("Root finding failed")
            exit(0)

        return error

    def initialGuess(self):
        ncheb = self.neq

        self.mm[0] = self.left
        for i in range(1, ncheb):
            r = 0.5 * (1 - math.cos((math.pi * i) / ncheb))
            r = math.expm1(r) / math.expm1(1)
            self.mm[i] = self.left + r * self.width
        self.mm[ncheb] = self.right

        for i in range(ncheb + 1):
            r = 0.5 * (1 - math.cos((math.pi * (2 * i + 1)) / (2 * ncheb)))
            r = math.expm1(r) / math.expm1(1)
            self.xx[i] = self.left + r * self.width

    def stpini(self, step):
        self.xx[self.neq + 1] = self.right
        self.delta = mpfr(0.25)
        step[0] = self.xx[0] - self.left
        for i in range(1, self.neq):
            step[i] = self.xx[i] - self.xx[i - 1]
        step[self.neq] = step[self.neq - 1]

    def search(self, step):
        self.meq = self.neq + 1
        yy = [None for _ in range(self.meq)]
        eclose = mpfr(1e30)
        farther = mpfr(0)

        xx0 = self.left

        for i in range(self.meq):
            steps = 0
            xx1 = self.xx[i]
            if i == self.meq - 1:
                xx1 = self.right
            xm = self.mm[i]
            ym, emsign = self.getErr(xm)
            q = step[i]
            xn = xm + q
            if xn < xx0 or xn >= xx1:
                q = -q
                xn = xm
                yn = ym
                ensign = emsign
            else:
                yn, ensign = self.getErr(xn)
                if yn < ym:
                    q = -q
                    xn = xm
                    yn = ym
                    ensign = emsign

            while yn >= ym:
                steps += 1
                if steps > 10:
                    break
                ym = yn
                xm = xn
                emsign = ensign
                a = xm + q
                if a == xm or a <= xx0 or a >= xx1:
                    break
                xn = a
                yn, ensign = self.getErr(xn)

            self.mm[i] = xm
            yy[i] = ym

            if eclose > ym:
                eclose = ym
            if farther < ym:
                farther = ym

            xx0 = xx1

        q = farther - eclose
        if eclose != 0:
            q /= eclose
        if q >= self.spread:
            self.delta *= 0.5
        self.spread = q

        for i in range(self.neq):
            q = yy[i + 1]
            if q != 0:
                q = yy[i] / q - 1
            else:
                q = 0.0625
            if q > 0.25:
                q = 0.25
            q *= self.mm[i + 1] - self.mm[i]
            step[i] = q * self.delta
        step[self.neq] = step[self.neq - 1]

        for i in range(self.neq):
            xm = self.xx[i] - step[i]
            if xm < self.left:
                continue
            if xm >= self.right:
                continue
            if xm <= self.mm[i]:
                xm = 0.5 * (self.mm[i] + self.xx[i])
            if xm >= self.mm[i + 1]:
                xm = 0.5 * (self.mm[i + 1] + self.xx[i])
            self.xx[i] = xm

    def getErr(self, x):
        f = self.func(x)
        e = self.approx(x) - f
        if f != 0:
            e /= f
        if e < 0.0:
            sign = -1
            e = -e
        else:
            sign = 1
        return e, sign

    def approx(self, x):
        yn = self.param[self.n]
        for i in range(self.n - 1, -1, -1):
            yn = x * yn + self.param[i]
        yd = x + self.param[self.n + self.d]
        for i in range(self.n + self.d - 1, self.n, -1):
            yd = x * yd + self.param[i]
        return yn / yd

    def equations(self):
        AA = [[None for _ in range(self.neq)] for _ in range(self.neq)]

        for i in range(self.neq):
            x = self.xx[i]
            y = self.func(x)

            z = mpfr(1)
            for j in range(self.n + 1):
                AA[i][j] = z
                z *= x

            z = mpfr(1)
            for j in range(self.d):
                AA[i][self.n + 1 + j] = -y * z
                z *= x
            self.param[i] = y * z

        lu_solve(AA, self.param)

    def root(self):
        upper = mpfr(1)
        lower = mpfr(-100000)
        tol = mpfr(1e-20)
        poly = [None for _ in range(self.neq + 1)]

        for i in range(self.n + 1):
            poly[i] = self.param[i]

        for i in range(self.n - 1, -1, -1):
            self.roots[i] = self.rtnewt(poly, i + 1, lower, upper, tol)
            if self.roots[i] == 0:
                print(f"Failure to converge on root {i+1}/{self.n}")
                return False
            poly[0] = -poly[0] / self.roots[i]
            for j in range(1, i + 1):
                poly[j] = (poly[j - 1] - poly[j]) / self.roots[i]

        poly[self.d] = mpfr(1)
        for i in range(self.d):
            poly[i] = self.param[self.n + 1 + i]

        for i in range(self.d - 1, -1, -1):
            self.poles[i] = self.rtnewt(poly, i + 1, lower, upper, tol)
            if self.poles[i] == 0.0:
                print(f"Failure to converge on pole {i+1}/{self.d}")
                return False
            poly[0] = -poly[0] / self.poles[i]
            for j in range(1, i + 1):
                poly[j] = (poly[j - 1] - poly[j]) / self.poles[i]

        self.norm = self.param[self.n]
        return True

    def rtnewt(self, poly, i, x1, x2, xacc):
        rtn = 0.5 * (x1 + x2)
        for j in range(1, 10001):
            f = self.polyEval(rtn, poly, i)
            df = self.polyDiff(rtn, poly, i)
            dx = f / df
            rtn -= dx
            if abs(dx) < xacc:
                return rtn
        print("Maximum number of iterations exceeded in rtnewt")
        return mpfr(0)

    def polyEval(self, x, poly, size):
        f = poly[size]
        for i in range(size - 1, -1, -1):
            f = f * x + poly[i]
        return f

    def polyDiff(self, x, poly, size):
        df = size * poly[size]
        for i in range(size - 1, 0, -1):
            df = df * x + i * poly[i]
        return df

    def getPFE(self):
        r, p = self.pfe(self.roots, self.poles, self.norm)

        return float(self.norm), [float(i) for i in r], [float(i) for i in p]

    def getIPFE(self):
        r, p = self.pfe(self.poles, self.roots, 1 / self.norm)

        return float(1 / self.norm), [float(i) for i in r], [float(i) for i in p]

    def pfe(self, res, poles, norm):
        import numpy as np

        res = np.array(res)
        poles = np.array(poles)
        indices = np.argsort(-poles)
        numerator = np.empty((self.n), object)
        denominator = np.empty((self.d), object)

        numerator[0] = mpfr(1)
        numerator[1:] = mpfr(0)
        denominator[0] = mpfr(1)
        denominator[1:] = mpfr(0)

        for j in range(self.n):
            numerator[1:] = -res[j] * numerator[1:] + numerator[:-1]
            denominator[1:] = -poles[j] * denominator[1:] + denominator[:-1]
            numerator[0] *= -res[j]
            denominator[0] *= -poles[j]

        numerator -= denominator

        for i in range(self.n):
            res[i] = mpfr(0)
            for j in range(self.n - 1, -1, -1):
                res[i] = poles[i] * res[i] + numerator[j]
            for j in range(self.n - 1, -1, -1):
                if i != j:
                    res[i] = res[i] / (poles[i] - poles[j])
            res[i] *= norm

        res[:] = res[indices]
        poles[:] = -poles[indices]
        return res, poles
