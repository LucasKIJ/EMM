import cvxpy as cp
import numpy as np
from scipy.special import lambertw


class ZeroRegularizer:
    def __init__(self):
        pass

    def prox(self, w, lam):
        return w

    def cvx(self, w, lam):
        return 0, "o"


class EntropyRegularizer:
    def __init__(self, limit=None):
        if limit is not None and limit <= 1:
            raise ValueError("limit is %.3f. It must be > 1." % limit)
        self.limit = limit

    def prox(self, w, lam):
        what = lam * np.real(lambertw(np.exp(w / lam - 1) / lam, tol=1e-12))
        if self.limit is not None:
            what = np.clip(what, 1 / (self.limit * w.size), self.limit / w.size)
        return what

    def cvx(self,w, lam):
        return -lam * cp.sum(cp.entr(w)), "o"


class KLRegularizer:
    def __init__(self, prior, w_min=0, w_max=float("inf")):
        self.prior = prior
        self.entropy_reg = EntropyRegularizer(w_min, w_max)

    def cvx(self, w, lam):
        return lam *  cp.sum(cp.kl_div(w, prior)), "o"

class BooleanRegularizer:
    def __init__(self, k):
        self.k = k

    def cvx(self, w, lam):
        # Outside the capabilities of cvx since this is
        # a non convex constraint. Possible to implement using
        # mixed integer programming
        raise NotImplementedError


if __name__ == "__main__":
    w = np.random.randn(10)
    prior = np.random.uniform(10)
    prior /= 10
    lam = 0.5
    zero_reg = ZeroRegularizer()
    np.testing.assert_allclose(zero_reg.prox(w, 0.5), w)

    entropy_reg = EntropyRegularizer()
    what = cp.Variable(10)
    cp.Problem(
        cp.Minimize(-cp.sum(cp.entr(what)) + 1 / (2 * lam) * cp.sum_squares(what - w))
    ).solve()
    kl_reg = KLRegularizer(prior)
    what = cp.Variable(10)
    cp.Problem(
        cp.Minimize(
            -cp.sum(cp.entr(what))
            - cp.sum(cp.multiply(what, np.log(prior)))
            + 1 / (2 * lam) * cp.sum_squares(what - w)
        )
    ).solve()
