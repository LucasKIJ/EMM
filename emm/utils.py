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
    sns.distplot(
        data,
        bins=bins,
        hist_kws={"weights": weights},
        kde_kws={"weights": weights},
        **kwargs
    )

