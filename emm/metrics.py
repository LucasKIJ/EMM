import numpy as np
import sklearn as sk
import scipy as sp
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def compute_probs(data, bins="auto"):
    if "weights" in data.columns:
        w = data["weight"]
    else:
        w = None

    bins = np.histogram_bin_edges(data, bins=bins, weights=w)
    h, e = np.histogram(data, bins=bins)
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


def js_divergence(p, q):
    m = (1.0 / 2.0) * (p + q)
    return (1.0 / 2.0) * kl_divergence(p, m) + (1.0 / 2.0) * kl_divergence(q, m)


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


def compute_js_divergence(original_sample, weighted_sample, n_bins=10):
    """
    Computes the JS Divergence using the support
    intersection between two different samples
    """
    e, p = compute_probs(original_sample, n=n_bins)
    _, q = compute_probs(weighted_sample, n=e)

    list_of_tuples = support_intersection(p, q)
    p, q = get_probs(list_of_tuples)

    return js_divergence(p, q)


def compare_model(
    target_model,
    weighted_model,
    X_target_test,
    y_target_test,
    X_weighted_test,
    y_weighted_test,
    weights_test,
    metrics=[sk.metrics.accuracy_score],
):
    scores = {}
    RR_pred = target_model.predict(X_target_test)
    RS_pred = target_model.predict(X_weighted_test)
    SS_pred = weighted_model.predict(X_weighted_test)
    SR_pred = weighted_model.predict(X_target_test)

    for i, metric in enumerate(metrics):
        RR_score = metric(y_target_test, RR_pred)
        RS_score = metric(y_weighted_test, RS_pred, sample_weight=weights_test)
        SS_score = metric(y_weighted_test, SS_pred, sample_weight=weights_test)
        SR_score = metric(y_target_test, SR_pred)
        scores[metric.__name__] = {
            "RR": RR_score,
            "RS": RS_score,
            "SS": SS_score,
            "SR": SR_score,
        }

    return scores


def train_test_splits_create(target_data, weighted_data, test_size, label):

    X_target = target_data.drop(columns=label)
    y_target = target_data[label]

    X_target_train, X_target_test, y_target_train, y_target_test = train_test_split(
        X_target, y_target, test_size=test_size
    )

    # Training data from weighted corpus
    X_weighted = weighted_data.drop(columns=label)
    y_weighted = weighted_data[label]
    (
        X_weighted_train,
        X_weighted_test,
        y_weighted_train,
        y_weighted_test,
    ) = train_test_split(X_weighted, y_weighted, test_size=test_size)

    weights_train = X_weighted_train["weights"]
    weights_test = X_weighted_test["weights"]
    X_weighted_train = X_weighted_train.drop(columns="weights")
    X_weighted_test = X_weighted_test.drop(columns="weights")

    return (
        X_target_train,
        y_target_train,
        X_target_test,
        y_target_test,
        X_weighted_train,
        y_weighted_train,
        weights_train,
        X_weighted_test,
        y_weighted_test,
        weights_test,
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
        weights_train,
        X_weighted_test,
        y_weighted_test,
        weights_test,
    ) = train_test_splits_create(target_data, weighted_data, test_size, label)

    classifier = param_grid["classifier"]
    pipe = Pipeline([("classifier", classifier[0])])

    cv = kwargs.get("cv", 5)
    verbose = kwargs.get("verbose", False)
    scoring = kwargs.get("scoring", None)
    target_clf = sk.model_selection.GridSearchCV(
        pipe,
        param_grid=[param_grid],
        cv=cv,
        verbose=verbose,
        n_jobs=-1,
        scoring=scoring,
    )

    # weighted_clf = sk.model_selection.GridSearchCV(
    #     pipe,
    #     param_grid=[param_grid],
    #     cv=cv,
    #     verbose=verbose,
    #     n_jobs=-1,
    #     scoring=scoring,
    # )

    best_target_clf = target_clf.fit(X_target_train, y_target_train)
    best_weighted_clf = best_target_clf.best_params_["classifier"]
    best_weighted_clf.fit(
        X_weighted_train, y_weighted_train, sample_weight=weights_train
    )

    # argw = {pipe.steps[-1][0] + "__sample_weight": weights_train}
    # best_weighted_clf = weighted_clf.fit(
    #     X_weighted_train, y_weighted_train_frame, **argw
    # )

    if verbose:
        print_cv_results(best_target_clf, "Target")
        # print_cv_results(best_weighted_clf, "Weighted")

    scores = compare_model(
        best_target_clf.best_estimator_,
        best_weighted_clf,
        X_target_test,
        y_target_test,
        X_weighted_test,
        y_weighted_test,
        weights_test,
    )

    return scores


############################################################################################################
# def cross_val_scores_weighted(model, X, y, weights, cv, metrics):
#     kf = KFold(n_splits=cv)
#     kf.get_n_splits(X)
#     scores = [[] for metric in metrics]
#     for train_index, test_index in kf.split(X):
#         model_clone = sk.base.clone(model)
#         X_train, X_test = X[train_index], X[test_index]
#         y_train, y_test = y[train_index], y[test_index]
#         weights_train, weights_test = weights[train_index], weights[test_index]
#         model_clone.fit(X_train, y_train, sample_weight=weights_train)
#         y_pred = model_clone.predict(X_test)
#         for i, metric in enumerate(metrics):
#             score = metric(y_test, y_pred, sample_weight=weights_test)
#             scores[i].append(score)
#     return np.array(scores)


# # function for fitting trees of various depths on the training data using cross-validation
# def run_cross_validation_on_trees(
#     X, y, weights, tree_depths, cv, metrics, verbose=False
# ):
#     X = np.array(X)
#     y = np.array(y)

#     # If no weights given, allocate even weighting
#     if weights is None:
#         verb = "Target"
#         weights = np.ones(X.shape[0]) / X.shape[0]
#     else:
#         verb = "Weighted"
#         weights = np.array(weights)
#     cv_scores_std = []
#     cv_scores_mean = []
#     for depth in tree_depths:
#         model = DecisionTreeClassifier(max_depth=depth)
#         cv_scores = cross_val_scores_weighted(model, X, y, weights, cv, metrics)
#         cv_scores_mean.append(cv_scores.mean())
#         cv_scores_std.append(cv_scores.std())
#     cv_scores_mean = np.array(cv_scores_mean)
#     cv_scores_std = np.array(cv_scores_std)
#     idx_max = cv_scores_mean.argmax()
#     best_tree_depth = tree_depths[idx_max]
#     best_tree_cv_score = cv_scores_mean[idx_max]
#     best_tree_cv_score_std = cv_scores_std[idx_max]
#     if verbose:
#         print(
#             "{} data: The depth-{} tree achieves the best mean cross-validation accuracy {} +/- {}% on training dataset".format(
#                 verb,
#                 best_tree_depth,
#                 round(best_tree_cv_score * 100, 5),
#                 round(best_tree_cv_score_std * 100, 5),
#             )
#         )

#     return best_tree_depth


# def compute_decision_tree_metric(
#     target_data, weighted_data, tree_depths, test_size=0.2, label="Outcome", **kwargs
# ):
#     (
#         X_target_train,
#         y_target_train,
#         X_target_test,
#         y_target_test,
#         X_weighted_train,
#         y_weighted_train,
#         weights_train,
#         X_weighted_test,
#         y_weighted_test,
#         weights_test,
#     ) = train_test_splits_create(target_data, weighted_data, test_size, label)

#     cv = kwargs.get("cv", 5)
#     metrics = kwargs.get("metrics", [sk.metrics.accuracy_score])
#     verbose = kwargs.get("verbose", False)

#     best_target_depth = run_cross_validation_on_trees(
#         X_target_train, y_target_train, None, tree_depths, cv, metrics, verbose
#     )
#     best_weighted_depth = run_cross_validation_on_trees(
#         X_weighted_train,
#         y_weighted_train,
#         weights_train,
#         tree_depths,
#         cv,
#         metrics,
#         verbose,
#     )

#     target_model = DecisionTreeClassifier(max_depth=best_target_depth).fit(
#         X_target_train, y_target_train
#     )
#     weighted_model = DecisionTreeClassifier(max_depth=best_weighted_depth).fit(
#         X_weighted_train, y_weighted_train, sample_weight=weights_train
#     )

#     scores = compare_model(
#         target_model,
#         weighted_model,
#         X_target_test,
#         y_target_test,
#         X_weighted_test,
#         y_weighted_test,
#         weights_test,
#     )
#     return scores


# def run_cross_validation_on_trees(
#     X, y, weights, tree_depths, cv, metrics, verbose=False
# ):
#     X = np.array(X)
#     y = np.array(y)

#     # If no weights given, allocate even weighting
#     if weights is None:
#         verb = "Target"
#         weights = np.ones(X.shape[0]) / X.shape[0]
#     else:
#         verb = "Weighted"
#         weights = np.array(weights)
#     cv_scores_std = []
#     cv_scores_mean = []
#     for depth in tree_depths:
#         model = DecisionTreeClassifier(max_depth=depth)
#         cv_scores = cross_val_scores_weighted(model, X, y, weights, cv, metrics)
#         cv_scores_mean.append(cv_scores.mean())
#         cv_scores_std.append(cv_scores.std())
#     cv_scores_mean = np.array(cv_scores_mean)
#     cv_scores_std = np.array(cv_scores_std)
#     idx_max = cv_scores_mean.argmax()
#     best_tree_depth = tree_depths[idx_max]
#     best_tree_cv_score = cv_scores_mean[idx_max]
#     best_tree_cv_score_std = cv_scores_std[idx_max]
#     if verbose:
#         print(
#             "{} data: The depth-{} tree achieves the best mean cross-validation accuracy {} +/- {}% on training dataset".format(
#                 verb,
#                 best_tree_depth,
#                 round(best_tree_cv_score * 100, 5),
#                 round(best_tree_cv_score_std * 100, 5),
#             )
#         )

#     return best_tree_depth

# def run_cross_validation_on_trees(
#     X, y, weights, tree_depths, cv, metrics, verbose=False
# ):
#     X = np.array(X)
#     y = np.array(y)

#     # If no weights given, allocate even weighting
#     if weights is None:
#         verb = "Target"
#         weights = np.ones(X.shape[0]) / X.shape[0]
#     else:
#         verb = "Weighted"
#         weights = np.array(weights)
#     cv_scores_std = []
#     cv_scores_mean = []
#     for depth in tree_depths:
#         model = DecisionTreeClassifier(max_depth=depth)
#         cv_scores = cross_val_scores_weighted(model, X, y, weights, cv, metrics)
#         cv_scores_mean.append(cv_scores.mean())
#         cv_scores_std.append(cv_scores.std())
#     cv_scores_mean = np.array(cv_scores_mean)
#     cv_scores_std = np.array(cv_scores_std)
#     idx_max = cv_scores_mean.argmax()
#     best_tree_depth = tree_depths[idx_max]
#     best_tree_cv_score = cv_scores_mean[idx_max]
#     best_tree_cv_score_std = cv_scores_std[idx_max]
#     if verbose:
#         print(
#             "{} data: The depth-{} tree achieves the best mean cross-validation accuracy {} +/- {}% on training dataset".format(
#                 verb,
#                 best_tree_depth,
#                 round(best_tree_cv_score * 100, 5),
#                 round(best_tree_cv_score_std * 100, 5),
#             )
#         )

#     return best_tree_depth
