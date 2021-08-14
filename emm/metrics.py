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

    if not isinstance(data, pd.Series):
        args = {"weights": np.array(data["weights"])}
        data = np.array(data.drop(columns=["weights"]), ndmin=0)
        data = data.flatten()
    else:
        args = {}
        data = np.array(data, ndmin=1)

    bins = np.histogram_bin_edges(data, bins=bins)
    h, e = np.histogram(data, bins=bins, **args)
    p = h / data.shape[0]
    # Return bin edges and probs
    return e, p


def support_intersection(p, q):
    sup_int = list(filter(lambda x: (x[0] != 0) & (x[1] != 0), zip(p, q)))
    return sup_int


def get_probs(list_of_tuples):
    p = np.array([p[0] for p in list_of_tuples])
    q = np.array([p[1] for p in list_of_tuples])
    return p, q


def kl_divergence(p, q):
    return np.sum(sp.special.kl_div(p, q))


def js_distance(p, q):
    return distance.jensenshannon(p, q)


def compute_kl_divergence(original_sample, weighted_sample, bins=10):
    """
    Computes the KL Divergence using the support
    intersection between two different samples
    """
    e, p = compute_probs(original_sample, bins=bins)
    _, q = compute_probs(weighted_sample, bins=e)

    list_of_tuples = support_intersection(p, q)
    p, q = get_probs(list_of_tuples)

    return kl_divergence(p, q)


def compute_js_distance(target, weighted, bins="auto"):
    """
    Computes the JS Divergence using the support
    intersection between two different samples
    """
    js_s = {}
    total_js = 0
    weighted["weights"] = weighted["weights"] * weighted["Outcome"].nunique()
    for outcome in target["Outcome"].unique():
        for feature in target.drop(columns="Outcome").columns:

            e, p = compute_probs(
                target[target["Outcome"] == outcome][feature], bins=bins
            )
            _, q = compute_probs(
                weighted[weighted["Outcome"] == outcome][[feature, "weights"]], bins=e
            )
            total_js += js_distance(p, q)
        js_s[outcome] = total_js
    return js_s


def compare_model(
    target_model,
    weighted_model,
    X_target_test,
    y_target_test,
    X_weighted_test,
    y_weighted_test,
    classifier,
    metrics=[sk.metrics.accuracy_score],
):  
    scores = []
    RR_pred = target_model.predict(X_target_test)
    RS_pred = target_model.predict(X_weighted_test)
    SS_pred = weighted_model.predict(X_weighted_test)
    SR_pred = weighted_model.predict(X_target_test)
    weights_test = X_weighted_test[:,0]
    for i, metric in enumerate(metrics):
        RR_score = metric(y_target_test, RR_pred)
        RS_score = metric(y_weighted_test, RS_pred, sample_weight=weights_test)
        SS_score = metric(y_weighted_test, SS_pred, sample_weight=weights_test)
        SR_score = metric(y_target_test, SR_pred)
        scores.append({metric.__name__: type(classifier[0]).__name__,
            "RR": RR_score,
            "RS": RS_score,
            "SS": SS_score,
            "SR": SR_score,
        })

    return scores


def train_test_splits_create(target_data, weighted_data, test_size, label):

    X_target = target_data.drop(columns=label)
    X_target.insert(0, "weights", np.ones(X_target.shape[0]) / X_target.shape[0])
    y_target = target_data[label]

    X_target_train, X_target_test, y_target_train, y_target_test = train_test_split(
        np.array(X_target), np.array(y_target), test_size=test_size,
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
    ) = train_test_split(np.array(X_weighted), np.array(y_weighted), test_size=test_size)

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
    target_data, weighted_data, param_grid, test_size=0.2, label="Outcome", **kwargs
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

    scoring = kwargs.get("scoring", sk.metrics.accuracy_score)
    def weight_remover_scorer(estimator, X, y):
        y_pred = estimator.predict(X)
        w = X[:, 0]
        return scoring(y, y_pred, sample_weight=w)

    classifier = param_grid["classifier"]
    pipe = Pipeline([("remove_weight", WeightRemover()), ("classifier", classifier[0])])
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
    target_clf = target_clf
    target_clf.fit(X_target_train, y_target_train)
    

    pipe = Pipeline([("remove_weight", WeightRemover()), ("classifier", classifier[0])])
    weighted_clf = sk.model_selection.GridSearchCV(
        pipe,
        param_grid=[param_grid],
        cv=cv,
        verbose=verbose,
        n_jobs=-1,
        scoring=weight_remover_scorer,
    )
    
    weighted_clf.fit(np.array(X_weighted_train), y_weighted_train)

    if verbose:
        print_cv_results(target_clf, "Target")
        print_cv_results(weighted_clf, "Weighted")

    scores = compare_model(
        target_clf.best_estimator_,
        weighted_clf.best_estimator_,
        X_target_test,
        y_target_test,
        X_weighted_test,
        y_weighted_test,
        classifier
    )

    return scores


def multiple_models(target, corpus, margs, param_grid, test_size = 0.2, **kwargs):
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
            metric += [classifier_metric(target, rw, params, test_size=test_size, **kwargs)]
        metrics += metric

    return rws, js, metrics


