from math import pi
import numpy as np
import sklearn as sk
import scipy as sp
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.spatial import distance
import emm


def compute_probs(data, bins="auto"):
    """
    Computes probability of data point falling in a given bins

    Arguments:
        data : pandas series or 2-d dataframe with "weights" column
        bins : number of bins to use in histogram approximation

    Returns : 1-d array of probabilities
    """
    # If data is dataframe then get weights
    if not isinstance(data, pd.Series):
        args = {"weights": np.array(data["weights"])}
        data = np.array(data.drop(columns=["weights"]), ndmin=0)
        data = data.flatten()
    else:
        args = {}
        data = np.array(data, ndmin=1)
    # Get bins from data
    bins = np.histogram_bin_edges(
        data, bins=bins, range=(np.nanmin(data), np.nanmax(data))
    )
    # Calculate histogram
    h, e = np.histogram(data, bins=bins, **args)
    # Calculate probabilities
    p = h / data.shape[0]
    # Return bin edges and probs
    return e, p


def support_intersection(p, q):
    """
    Get overlapping parts of distribution

    Arguments:
        p : 1-d array of probabilites
        q : 1-d array of probabilites

    Returns:
        sup_int : tuple of overlapping probabilities
    """

    sup_int = list(filter(lambda x: (x[0] != 0) & (x[1] != 0), zip(p, q)))
    return sup_int


def get_probs(list_of_tuples):
    """
    Gets probabilties from tuples

    Arguments:
        list_of_tuples : list of tuples with overlapping probabilities

    Returns:
        p : 1-d array of probabilities
        q : 1-d array of probabilities
    """
    p = np.array([p[0] for p in list_of_tuples])
    q = np.array([p[1] for p in list_of_tuples])
    return p, q


def kl_divergence(p, q):
    """
    Compute KL Divergence from two lists of probabilities from a distribution

    Arguments:
        p : 1-d array of probabilities
        q : 1-d array of probabilities

    Returns:
        KL Divergence, D_KL(p || q)
    """
    return np.sum(sp.special.kl_div(p, q))


def js_distance(p, q):
    """
    Compute JS Distance from two lists of probabilities from a distribution

    Arguments:
        p : 1-d array of probabilities
        q : 1-d array of probabilities

    Returns:
        JS Distance, D_JS(p || q)
    """
    return distance.jensenshannon(p, q, base=2)


def compute_kl_divergence(original_sample, weighted_sample, bins=10):
    """
    Computes the KL Divergence using the support
    intersection between two different samples.

    Arguments:
        original_sample : 1-d array or dataframe with weights of samples
        weighted_sample : 1-d array or dataframe with weights of samples
        bins : number of bins to use in histogram

    Returns:
        KL Divergence of from two samples distributions
    """
    e, p = compute_probs(original_sample, bins=bins)
    _, q = compute_probs(weighted_sample, bins=e)

    list_of_tuples = support_intersection(p, q)
    p, q = get_probs(list_of_tuples)

    return kl_divergence(p, q)


def compute_js_distance(target, weighted, bins="auto"):
    """
    Computes the JS Distance using the support
    intersection between two different samples.

    Arguments:
        target : 1-d array or dataframe with weights of samples
        weighted : 1-d array or dataframe with weights of samples
        bins : number of bins to use in histogram

    Returns:
        KL Divergence of from two samples distributions
    """
    js_s = {}

    weighted["weights"] = weighted["weights"] * weighted["Outcome"].nunique()
    for outcome in target["Outcome"].unique():
        total_js = {}
        for feature in target.drop(columns="Outcome").columns:

            e, p = compute_probs(
                target[target["Outcome"] == outcome][feature], bins=bins
            )
            _, q = compute_probs(
                weighted[weighted["Outcome"] == outcome][[feature, "weights"]], bins=e
            )
            total_js[feature] = js_distance(p, q)
        js_s[outcome] = total_js
    return js_s


def compare_model(
    target_model,
    weighted_model,
    X_target_test,
    y_target_test,
    X_weighted_test,
    y_weighted_test,
    classifier="Unknown",
    metrics=[sk.metrics.accuracy_score],
):

    """
    Compares target ML model and weighted ML for given metrics

    Arguments:
        target_model : sklearn model or pipeline
        weighted_model : sklearn model or pipeline
        X_target_test : test feature data from target dataset
        y_target_test : test label data from target dataset
        X_weighted_test : test feature data from reweighted dataset
        y_weighted_test : test label data from reweighted dataset
        classifier : Name of classifier being tested
        metrics : list of metrics to return scores for

    Returns:
        list with dictionary of scores
    """
    if type(metrics) is not list:
        metrics = [metrics]
    scores = []

    weights_test = X_weighted_test[:, 0]

    for i, metric in enumerate(metrics):
        if metric == sk.metrics.roc_auc_score:
            RR_pred = target_model.predict_proba(X_target_test)[:, 1]
            RS_pred = target_model.predict_proba(X_weighted_test)[:, 1]
            SS_pred = weighted_model.predict_proba(X_weighted_test)[:, 1]
            SR_pred = weighted_model.predict_proba(X_target_test)[:, 1]
        else:
            RR_pred = target_model.predict(X_target_test)
            RS_pred = target_model.predict(X_weighted_test)
            SS_pred = weighted_model.predict(X_weighted_test)
            SR_pred = weighted_model.predict(X_target_test)

        RR_score = metric(y_target_test, RR_pred)
        RS_score = metric(y_weighted_test, RS_pred, sample_weight=weights_test)
        SS_score = metric(y_weighted_test, SS_pred, sample_weight=weights_test)
        SR_score = metric(y_target_test, SR_pred)
        scores.append(
            {
                metric.__name__: type(classifier[0]).__name__,
                "RR": RR_score,
                "RS": RS_score,
                "SS": SS_score,
                "SR": SR_score,
            }
        )

    return scores


def train_test_splits_create(target_data, weighted_data, test_size, label):

    X_target = target_data.drop(columns=label)
    X_target.insert(0, "weights", np.ones(X_target.shape[0]) / X_target.shape[0])
    y_target = target_data[label]

    X_target_train, X_target_test, y_target_train, y_target_test = train_test_split(
        np.array(X_target),
        np.array(y_target),
        test_size=test_size,
    )

    # Training data from weighted corpus
    X_weighted = weighted_data.drop(columns=label)
    # shift column 'Name' to first position
    first_column = X_weighted.pop("weights")

    # insert column using insert(position,column_name,
    # first_column) function
    X_weighted.insert(0, "weights", first_column)
    y_weighted = weighted_data[label]
    (
        X_weighted_train,
        X_weighted_test,
        y_weighted_train,
        y_weighted_test,
    ) = train_test_split(
        np.array(X_weighted), np.array(y_weighted), test_size=test_size
    )

    return (
        X_target_train,
        y_target_train,
        X_target_test,
        y_target_test,
        X_weighted_train,
        y_weighted_train,
        X_weighted_test,
        y_weighted_test,
    )


def print_cv_results(best_clf, data_name=""):
    print(
        "{} data: the best parameters are given by \n {}".format(
            data_name, best_clf.best_params_["classifier"]
        )
        + "\n the best mean cross-validation accuracy {} +/- {}% on training dataset \n".format(
            round(best_clf.best_score_ * 100, 5),
            round(
                best_clf.cv_results_["std_test_score"][best_clf.best_index_] * 100,
                5,
            ),
        )
    )


# customized transformer
class WeightRemover(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        return X[:, 1:]


def classifier_metric(
    target_data,
    weighted_data,
    param_grid,
    test_size=0.2,
    label="Outcome",
    return_models=False,
    **kwargs
):

    (
        X_target_train,
        y_target_train,
        X_target_test,
        y_target_test,
        X_weighted_train,
        y_weighted_train,
        X_weighted_test,
        y_weighted_test,
    ) = train_test_splits_create(target_data, weighted_data, test_size, label)
    scoring = kwargs.pop("scoring", sk.metrics.accuracy_score)
    scoring = sk.metrics.get_scorer(scoring)

    def weight_remover_scorer(estimator, X, y):
        if scoring == sk.metrics.roc_auc_score:
            print("test")
            y_pred = estimator.predict_proba(X)[:, 1]
        else:
            y_pred = estimator.predict(X)
        w = X[:, 0]
        return scoring(y, y_pred, sample_weight=w)

    classifier = param_grid["classifier"]

    pipe_steps = kwargs.get("pipeline_steps", []).copy()
    pipe_steps.extend(
        [("remove_weight", WeightRemover()), ("classifier", classifier[0])]
    )
    pipe = Pipeline(pipe_steps)
    cv = kwargs.get("cv", 5)
    verbose = kwargs.get("verbose", False)

    target_clf = sk.model_selection.GridSearchCV(
        pipe,
        param_grid=[param_grid],
        cv=cv,
        verbose=verbose,
        n_jobs=-1,
        scoring=weight_remover_scorer,
    )
    target_clf.fit(
        X_target_train, y_target_train, classifier__sample_weight=X_target_train[:, 0]
    )

    pipe = Pipeline(pipe_steps)
    weighted_clf = sk.model_selection.GridSearchCV(
        pipe,
        param_grid=[param_grid],
        cv=cv,
        verbose=verbose,
        n_jobs=-1,
        scoring=weight_remover_scorer,
    )

    weighted_clf.fit(
        np.array(X_weighted_train),
        y_weighted_train,
        classifier__sample_weight=X_weighted_train[:, 0],
    )
    if verbose:
        print_cv_results(target_clf, "Target")
        print_cv_results(weighted_clf, "Weighted")

    metrics = kwargs.pop("metrics", [sk.metrics.accuracy_score])
    scores = compare_model(
        target_clf.best_estimator_,
        weighted_clf.best_estimator_,
        X_target_test,
        y_target_test,
        X_weighted_test,
        y_weighted_test,
        classifier,
        metrics=metrics,
    )

    if return_models:
        return [target_clf, weighted_clf, scores]

    return scores


def multiple_models(target, corpus, margs, param_grid, test_size=0.2, **kwargs):
    rws = []
    js = []
    metrics = []
    bins = kwargs.pop("bins", "auto")
    if type(param_grid) != list:
        param_grid = [param_grid]
    if type(margs) != list:
        margs = [margs]

    for marg in margs:
        rw = emm.reweighting.generate_synth(corpus, marg, **kwargs)
        rws += [rw]
        js += [compute_js_distance(target, rw, bins=bins)]
        metric = []
        for params in param_grid:
            metric += [
                classifier_metric(target, rw, params, test_size=test_size, **kwargs)
            ]
        metrics += metric
    if len(rws) == 1:
        rws = rws[0]
    if len(js) == 1:
        js = js[0]
    if len(metrics) == 1:
        metrics = metrics[0]

    return rws, js, metrics


if __name__ == "__main__":
    import emm
    import sklearn as sk

    def weight_remover_scorer(estimator, X, y):

        y_pred = estimator.predict(X)
        w = X[:, -1]
        return sk.metrics.accuracy_score(y, y_pred, sample_weight=w)

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

    histLoss0 = emm.losses.CorpusKLLoss(mean=mu0[0], std=sig0[0], scale=2)
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

    margs = margsLS

    rwc = emm.reweighting.generate_synth(
        corpus, margs, regularizers=emm.regularizers.EntropyRegularizer(), lam=1
    )

    # model = RandomForestClassifier(
    #     n_estimators=5,
    #     criterion="entropy",
    #     warm_start=False,
    #     n_jobs=1,
    # )

    # model = sk.linear_model.LogisticRegression()
    model = sk.tree.DecisionTreeClassifier(max_depth=2)

    pipe = Pipeline([("remove_weight", WeightRemover()), ("model", model)])
    search_params = {}
    params_grid = {"model__" + k: v for k, v in search_params.items()}
    X = np.array(rwc.drop(columns="Outcome"))
    y = np.array(rwc[["Outcome"]])
    X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(X, y)

    grid = sk.model_selection.GridSearchCV(
        pipe, params_grid, cv=10, scoring=weight_remover_scorer
    )

    grid.fit(X_train, y_train, model__sample_weight=X_train[:, 1])

    print(
        "This is the best out-of-sample score using GridSearchCV: %.6f."
        % grid.score(X_train, y_train)
    )

    print(grid.score(X_test, y_test))
