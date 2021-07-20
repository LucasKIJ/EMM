import pandas as pd
import cvxpy as cp
import numpy as np
from scipy import sparse
import time

from rsw.solver import admm


def rsw(df, funs, losses, regularizer, lam=1, **kwargs):
    """Optimal representative sample weighting.

    Arguments:
        - df: Pandas dataframe
        - funs: functions to apply to each row of df.
        - losses: list of losses, each one of rsw.EqualityLoss, rsw.InequalityLoss, rsw.LeastSquaresLoss,
            or rsw.KLLoss()
        - regularizer: One of rsw.ZeroRegularizer, rsw.EntropyRegularizer,
            or rsw.KLRegularizer, rsw.BooleanRegularizer
        - lam (optional): Regularization hyper-parameter (default=1).
        - kwargs (optional): additional arguments to be sent to solver. For example: verbose=False,
            maxiter=5000, rho=50, eps_rel=1e-5, eps_abs=1e-5.

    Returns:
        - w: Final sample weights.
        - out: Final induced expected values as a list of numpy arrays.
        - sol: Dictionary of final ADMM variables. Can be ignored.
    """



    if funs is None:
        F = np.array(df).T
        print(F)
    if type(funs) is list:
        F = []
        for f in funs:
            F += [df.apply(f, axis=1)]
        F = np.array(F, dtype=float)
    if type(funs) is dict:
        F = []
        for f in funs['mean']:
            F += [df[f]]

        for f in funs['std']:
            F += [abs(df[f] - df[f].mean())]

        for f in funs['skew']:
            F += [((df[f] - df[f].mean()) / df[f].std())**3]

        for f in funs['kurtosis']:
            F += [((df[f] - df[f].mean()) / df[f].std())**4]
        F = np.array(F, dtype=float)

    m, n = F.shape

    # remove nans by changing F
    rows_nan, cols_nan = np.where(np.isnan(F))
    desireds = [l.fdes for l in losses]
    desired = np.concatenate(desireds)
    if rows_nan.size > 0:
        for i in np.unique(rows_nan):
            F[i, cols_nan[rows_nan == i]] = desired[i]

    F_sparse = sparse.csc_matrix(F)
    tic = time.time()
    sol = admm(F_sparse, losses, regularizer, lam, **kwargs)
    toc = time.time()
    if kwargs.get("verbose", False):
        print("ADMM took %3.5f seconds" % (toc - tic))

    out = []
    means = F @ sol["w_best"]
    ct = 0
    for m in [l.m for l in losses]:
        out += [means[ct:ct + m]]
        ct += m
    return sol["w_best"], out, sol
