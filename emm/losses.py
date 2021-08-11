import cvxpy as cp
import numpy as np
import pandas as pd
from scipy.special import lambertw, kl_div
from numbers import Number
from scipy import stats


class EqualityLoss:
    def __init__(self, fdes):
        if isinstance(fdes, Number):
            fdes = np.array([fdes])
        self.fdes = fdes
        self.m = fdes.size

    def prox(self, f, lam):
        return self.fdes

    def cvx(self, f, w):
        return [cp.max(cp.abs(f @ w - self.fdes)) <= 1e-10], "c"


class InequalityLoss:
    def __init__(self, fdes, lower, upper):
        if isinstance(fdes, Number):
            fdes = np.array([fdes])
        self.fdes = fdes
        self.m = fdes.size
        self.lower = lower
        self.upper = upper
        assert (self.lower <= self.upper).all()

    def prox(self, f, lam):
        return np.clip(f, self.fdes + self.lower, self.fdes + self.upper)

    def cvx(self, f, w):
        if (self.upper == -self.lower).all():
            return [cp.max(cp.abs(f @ w - self.fdes)) <= self.upper], "c"
        else:
            return [
                cp.max(f @ w - self.fdes) <= self.upper,
                cp.min(f @ w - self.fdes) >= self.lower,
            ], "c"


class LeastSquaresLoss:
    def __init__(self, fdes, diag_weight=None, scale=1.0):
        if isinstance(fdes, Number):
            fdes = np.array([fdes])
        self.fdes = fdes
        self.m = fdes.size
        self.scale = scale
        if diag_weight is None:
            diag_weight = 1.0
        self.diag_weight = diag_weight

    def prox(self, f, lam):
        return (self.diag_weight ** 2 * self.fdes + f / lam) / (
            self.diag_weight ** 2 + 1 / lam
        )

    def evaluate(self, f):
        return self.scale * np.sum(np.square(self.diag_weight * (f - self.fdes)))

    def cvx(self, f, w):
        return self.scale * cp.sum_squares(f @ w - self.fdes), "o"


def _entropy_prox(f, lam):
    return lam * np.real(lambertw(np.exp(f / lam - 1) / lam, tol=1e-10))


class KLLoss:
    def __init__(self, fdes, scale=1, bins=10):
        if isinstance(fdes, Number):
            fdes = np.array([fdes])
        self.fdes = fdes
        self.m = fdes.size
        self.scale = scale

    def prox(self, f, lam):
        return _entropy_prox(f + lam * self.scale * np.log(self.fdes), lam * self.scale)

    def evaluate(self, f):
        return self.scale * np.sum(kl_div(f, self.fdes))

    def cvx(self, f, w):
        return self.scale * cp.sum(cp.kl_div(f @ w, self.fdes)), "o"


class CorpusKLLoss:
    def __init__(self, fdes, scale=1, bins='auto'):
        if isinstance(fdes, Number):
            fdes = np.array([fdes])
        self.fdes = fdes
        self.m = fdes.size
        self.scale = scale
        self.bins = bins

    def prox(self, f, lam):
        return _entropy_prox(f + lam * self.scale * np.log(self.fdes), lam * self.scale)

    def evaluate(self, f):
        return self.scale * np.sum(kl_div(f, self.fdes))

    def cvx(self, f, w):
        return self.scale * cp.sum(cp.kl_div(f @ w, self.fdes)), "o"


if __name__ == "__main__":
    m = 10
    f = np.random.randn(m)
    fdes = np.random.randn(m)
    lam = 1

    equality = EqualityLoss(fdes)
    fhat = cp.Variable(m)
    cp.Problem(cp.Minimize(1 / lam * cp.sum_squares(fhat - f)), [fhat == fdes]).solve()
    np.testing.assert_allclose(fhat.value, equality.prox(f, lam))

    lower = np.array([-0.3])
    upper = np.array([0.3])
    inequality = InequalityLoss(fdes, lower, upper)
    fhat = cp.Variable(m)
    cp.Problem(
        cp.Minimize(1 / lam * cp.sum_squares(fhat - f)),
        [lower <= fhat - fdes, fhat - fdes <= upper],
    ).solve()
    np.testing.assert_allclose(fhat.value, inequality.prox(f, lam))

    d = np.random.uniform(0, 1, size=m)
    lstsq = LeastSquaresLoss(fdes, d)
    fhat = cp.Variable(m)
    cp.Problem(
        cp.Minimize(
            1 / 2 * cp.sum_squares(cp.multiply(d, fhat - fdes))
            + 1 / (2 * lam) * cp.sum_squares(fhat - f)
        )
    ).solve()
    np.testing.assert_allclose(fhat.value, lstsq.prox(f, lam))

    f = np.random.uniform(0, 1, size=m)
    f /= f.sum()
    fdes = np.random.uniform(0, 1, size=m)
    fdes /= fdes.sum()

    fhat = cp.Variable(m)
    cp.Problem(
        cp.Minimize(cp.sum(-cp.entr(fhat)) + 1 / (2 * lam) * cp.sum_squares(fhat - f))
    ).solve()
    np.testing.assert_allclose(fhat.value, _entropy_prox(f, lam), atol=1e-5)

    kl = KLLoss(fdes, scale=0.5)
    fhat = cp.Variable(m, nonneg=True)
    cp.Problem(
        cp.Minimize(
            0.5 * (cp.sum(-cp.entr(fhat) - cp.multiply(fhat, np.log(fdes))))
            + 1 / (2 * lam) * cp.sum_squares(fhat - f)
        )
    ).solve()
    np.testing.assert_allclose(fhat.value, kl.prox(f, lam), atol=1e-5)
