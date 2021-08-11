import numpy as np
import pandas as pd


def onehot_hist(f, bins="auto"):
    bins = np.histogram_bin_edges(f, bins=bins)
    onehot_f = pd.get_dummies(pd.cut(f, bins=bins))
    return onehot_f, bins
