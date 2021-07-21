import pandas as pd
import numpy as np
import cvxpy as cp
import numpy as np
from scipy import sparse
import time
from emm.solvers import admm, gurobi
from emm.regularizers import ZeroRegularizer


def emm(corpus, marginals, losses, regularizer = ZeroRegularizer, 
optimizer = 'admm', lam = 1, **kwargs):
    """Optimal representative sample weighting.

    Arguments:
        - corpus: Pandas dataframe of corpus dataset
        - funs: Dictionary i.e. {feature : [funs]} of which functions to apply
            to each feature or list i.e [fun1, fun2,...] of functions to apply 
            to every feature of the corpus dataset.
        - losses: Dist of losses, each one of emm.EqualityLoss, emm.InequalityLoss, rsw.LeastSquaresLoss,
            or emm.KLLoss()
        - optimizer: Optimiser used to find weights, current choices are 
            'admm', 
        - regularizer: Weights regularizer, emm.ZeroRegularizer, 
            emm.EntropyRegularizer, or emm.KLRegularizer, emm.BooleanRegularizer
        - lam (optional): Regularization hyper-parameter (default=1).
        - kwargs (optional): additional arguments to be sent to solver. For example: verbose=False,
            maxiter=5000, rho=50, eps_rel=1e-5, eps_abs=1e-5.
    Returns:
        - w: Final sample weights.
        - out: Final induced expected values (weighted marginals)
            as a list of numpy arrays.
    """

    if marginals is None:
        # Just match means
        F = np.array(corpus).T
    if type(marginals) is list:
        # If list is used, apply each function in list
        # to every instance of every feature
        F = []
        for f in marginals:
            F += [corpus.apply(f, axis=1)]
        F = np.array(F, dtype=float)
    
    if type(marginals) is dict:
        # If dictionary is used, each key represents a feature.
        # Value for that key are the functions to be apply to
        # that feature.
        F = []
        for feature in marginals:
            for fun in marginals[feature]:
                # Special case commands
                if fun == 'mean':
                    F += [corpus[feature]]
                if fun == 'std':
                    F += [abs(corpus[feature] 
                                - corpus[feature].mean())]
                if fun == 'skew': 
                    F += [((corpus[feature] - corpus[feature].mean()) 
                            / corpus[feature].std())**3]
                if fun == 'kurtosis':
                    F += [((corpus[feature] - corpus[feature].mean()) 
                            / corpus[feature].std())**4]
                #else:
                #    F += [corpus[feature].apply(fun,axis=1)]

        F = np.array(F, dtype=float)

    m, n = F.shape

    # remove nans by changing F
    rows_nan, cols_nan = np.where(np.isnan(F))
    desireds = [l.fdes for l in losses]
    desired = np.concatenate(desireds)
    if rows_nan.size > 0:
        for i in np.unique(rows_nan):
            F[i, cols_nan[rows_nan == i]] = desired[i]
    
    if optimizer == 'admm':
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

    if optimizer == 'gurobi':
        w = gurobi(F,losses,regularizer,lam)
        return w,0,0