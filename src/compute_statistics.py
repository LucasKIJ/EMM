"""
compute_statistics.py
==============================
The aim of this script is to enable a dataframe of real values to be converted into a table of marginal statistics that
can be used to generate data from.

This would not be done in practice, but is highly useful in development as it allows us to build mock statistical
datasets from real data.
"""
import pandas as pd


def marginal_statistics_from_dataframe(frame):
    """Method for generating marginal statistics from dataframe.

    This function should accept a dataframe that has a binary column marked label.

    It should group by the label column and compute the mean and standard deviation (or potentially some other method)
    conditional on the label being 0 or 1. These results should be stored in some usable format.
    """
    X = frame.drop(columns="label")
    y = frame["label"]

    mean = frame.groupby("label").mean()
    std = frame.groupby("label").std()
    skew = frame.groupby("label").skew()
    kurt = frame.groupby("label").apply(pd.DataFrame.kurt)

    pass
