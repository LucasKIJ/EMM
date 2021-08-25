from emm.test import weight_remover_scorer
import numpy as np
import pandas as pd
import seaborn as sns


def onehot_hist(f, bins="auto"):
    bins = np.histogram_bin_edges(f, bins=bins, range=(np.nanmin(f), np.nanmax(f)))
    onehot_f = pd.get_dummies(pd.cut(f, bins=bins))
    return onehot_f, bins


def weighted_hist(data, weights, bins="auto", **kwargs):
    bins = np.histogram_bin_edges(
        data, bins=bins, range=(np.nanmin(data), np.nanmax(data))
    )
    nans = ~np.isnan(data).copy()
    data = data[nans]
    weights = weights[nans]
    hist_kws = kwargs.pop("hist_kws", {}) | {"weights": weights}
    kde_kws = kwargs.pop("kde_kws", {}) | {"weights": weights}
    sns.distplot(data, bins=bins, hist_kws=hist_kws, kde_kws=kde_kws, **kwargs)


def weighted_mean(data, weights):
    return data.T @ (weights) / weights.sum()


def weighted_var(data, weights):
    data.fillna()
    weights = weights / weights.sum()
    return ((data - data.T @ weights) ** 2).T @ weights
