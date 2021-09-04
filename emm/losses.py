import cvxpy as cp
import numpy as np
import pandas as pd
from scipy.special import lambertw, kl_div
from numbers import Number
from scipy import stats
from emm.utils import onehot_hist


class EqualityLoss:
    def __init__(self, fdes):
        if isinstance(fdes, Number):
            fdes = np.array([fdes])
        self.fdes = fdes
        self.m = fdes.size

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


    def evaluate(self, f):
        return self.scale * np.sum(np.square(self.diag_weight * (f - self.fdes)))

    def cvx(self, f, w):
        return self.scale * cp.sum_squares(f @ w - self.fdes), "o"



class KLLoss:
    def __init__(self, fdes, scale=1, bins=10):
        if isinstance(fdes, Number):
            fdes = np.array([fdes])
        self.fdes = fdes
        self.m = fdes.size
        self.scale = scale

    def evaluate(self, f):
        return self.scale * np.sum(kl_div(f, self.fdes))

    def cvx(self, f, w):
        return self.scale * cp.sum(cp.kl_div(f @ w, self.fdes)), "o"


class CorpusKLLoss:
    def __init__(self, mean=None, std=None, scale=1, bins='auto'):
        self.scale = scale
        self.bins = bins
        self.mean = mean
        self.std = std
    
    def fun(self, data):
        if self.mean == None:
            self.mean = data.mean()
        if self.std == None:
            self.std = data.std()

        F, self.bins = onehot_hist(data, self.bins)

        target = (data - data.mean() ) * (self.std / data.std()) + self.mean 
        hist, _ = onehot_hist(target, bins=self.bins)
        self.fdes = np.array(hist).T @ (np.ones(target.shape[0]) / target.shape[0])
        self.m = self.fdes.size

        return F

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

    lower = np.array([-0.3])
    upper = np.array([0.3])
    inequality = InequalityLoss(fdes, lower, upper)
    fhat = cp.Variable(m)
    cp.Problem(
        cp.Minimize(1 / lam * cp.sum_squares(fhat - f)),
        [lower <= fhat - fdes, fhat - fdes <= upper],
    ).solve()

    d = np.random.uniform(0, 1, size=m)
    lstsq = LeastSquaresLoss(fdes, d)
    fhat = cp.Variable(m)
    cp.Problem(
        cp.Minimize(
            1 / 2 * cp.sum_squares(cp.multiply(d, fhat - fdes))
            + 1 / (2 * lam) * cp.sum_squares(fhat - f)
        )
    ).solve()

    f = np.random.uniform(0, 1, size=m)
    f /= f.sum()
    fdes = np.random.uniform(0, 1, size=m)
    fdes /= fdes.sum()

    fhat = cp.Variable(m)
    cp.Problem(
        cp.Minimize(cp.sum(-cp.entr(fhat)) + 1 / (2 * lam) * cp.sum_squares(fhat - f))
    ).solve()

    kl = KLLoss(fdes, scale=0.5)
    fhat = cp.Variable(m, nonneg=True)
    cp.Problem(
        cp.Minimize(
            0.5 * (cp.sum(-cp.entr(fhat) - cp.multiply(fhat, np.log(fdes))))
            + 1 / (2 * lam) * cp.sum_squares(fhat - f)
        )
    ).solve()
