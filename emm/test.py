from __future__ import division
import collections

import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, RepeatedKFold
from sklearn.metrics import make_scorer
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import scipy as sp


# customized scorer


def weight_remover_scorer(estimator, X, y):

    y_pred = estimator.predict(X)
    w = X[:, -1]
    return accuracy_score(y, y_pred, sample_weight=w)


# customized transformer
class WeightRemover(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        return X[:, :-1]


# in your main function

if __name__ == "__main__":
    import emm
    import sklearn as sk

    # Generate example data
    m = 10000
    # Target distribution
    mu0 = np.array([-0.3])
    sig0 = np.array([0.35])
    mu1 = np.array([0.2])
    sig1 = np.array([0.3])
    rv0 = sp.stats.skewnorm(a=0, loc=mu0[0], scale=sig0[0])
    rv1 = sp.stats.skewnorm(a=0, loc=mu1[0], scale=sig1[0])
    X0 = rv0.rvs(size=m // 4)
    X1 = rv1.rvs(size=m // 4)
    y0 = np.zeros(m // 4)
    y1 = np.ones(m // 4)
    X = np.concatenate([X0, X1])
    y = np.concatenate([y0, y1])
    target = pd.DataFrame({"feature": X})
    target["Outcome"] = y

    # Corpus distribution
    mu = np.array([-1])
    sig = np.array([2])
    rvc = sp.stats.skewnorm(a=2, loc=mu, scale=sig)
    corpus = rvc.rvs(size=m)
    corpus = pd.DataFrame({"feature": corpus})
    sample_weight = np.ones(len(X)) / len(X)
    sample_weight = np.random.rand(len(X))
    sample_weight = sample_weight / sample_weight.sum()

    histLoss0 = emm.losses.CorpusKLLoss(mean=mu0[0], std=sig0[0],scale=2)
    histLoss1 = emm.losses.CorpusKLLoss(mean=mu1[0], std=sig1[0], scale=2)


    lam = 0
    margsKL = {
        0: [emm.reweighting.marginal("feature", histLoss0.fun, histLoss0)],
        1: [emm.reweighting.marginal("feature", histLoss1.fun, histLoss1)],
    }

    margsLS = {
        0: [
            emm.reweighting.marginal(
                "feature", "mean", emm.losses.LeastSquaresLoss(mu0[0], scale=100), False
            ),
            emm.reweighting.marginal(
                "feature",
                "var",
                emm.losses.LeastSquaresLoss(sig0[0] ** 2, scale=100),
                False,
            ),
        ],
        1: [
            emm.reweighting.marginal(
                "feature", "mean", emm.losses.LeastSquaresLoss(mu1[0], scale=100), False
            ),
            emm.reweighting.marginal(
                "feature",
                "var",
                emm.losses.LeastSquaresLoss(sig1[0] ** 2, scale=100),
                False,
            ),
        ],
    }

    margs =  margsLS


    rwc = emm.reweighting.generate_synth(
        corpus, margs, regularizers=emm.regularizers.EntropyRegularizer(), lam=1
    )

    # model = RandomForestClassifier(
    #     n_estimators=5,
    #     criterion="entropy",
    #     warm_start=False,
    #     n_jobs=1,
    # )

    #model = sk.linear_model.LogisticRegression()
    model = sk.tree.DecisionTreeClassifier(max_depth=2)

    pipe = Pipeline([("remove_weight", WeightRemover()), ("model", model)])
    search_params = {}
    params_grid = {"model__" + k: v for k, v in search_params.items()}
    X = np.array(rwc.drop(columns='Outcome'))
    y = np.array(rwc[['Outcome']])
    X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(X,y)

    grid = GridSearchCV(pipe, params_grid, cv=10, scoring=weight_remover_scorer)
    
    grid.fit(X_train, y_train, model__sample_weight=X_train[:,1])

    print(
        "This is the best out-of-sample score using GridSearchCV: %.6f."
        % grid.score(X_train,y_train)
    )

    print(grid.score(X_test,y_test))
