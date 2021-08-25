import pandas as pd
import numpy as np

import os
from pathlib import Path

DATA_DIR = Path("../assets/data/processed")
CORPUS_FILE = DATA_DIR / "hospital_admissions/df.csv"
TARGET_FILE = DATA_DIR / "covid_data/corona_tested_individuals_ver_006.english.csv"

# Check data is setup correctly
assert os.path.isdir(
    "../assets/data"
), "You must make a /data/ directory from root and add the admissionprediction.csv file in /data/raw"
assert os.path.exists(CORPUS_FILE), "Cannot find file at {}.".format(CORPUS_FILE)
assert os.path.exists(TARGET_FILE), "Cannot find file at {}.".format(TARGET_FILE)


def get_target():
    """
    Get target data from covid_data folder

    Returns:
        Processed target data as Dataframe
    """

    # Read in covid data
    target = pd.read_csv(TARGET_FILE)

    # Remove entries with covid result == other
    target = target[target["corona_result"] != "other"]
    # Reset index
    target = target.reset_index(drop=True)
    # Remove test_date column
    target = target.drop(columns=["test_date"])
    # Convert string entries to binary values
    cleanup_nums = {
        "corona_result": {"negative": 0, "positive": 1},
        "age_60_and_above": {"No": 0, "Yes": 1},
        "gender": {"female": 0, "male": 1},
    }
    # Replace string entries
    target = target.replace(cleanup_nums)
    # Change string Nan to numpy nan
    target = target.replace("None", np.nan)
    # Get numeric columns
    num_cols = target.drop(columns="test_indication").columns
    target[num_cols] = target[num_cols].apply(pd.to_numeric, errors="coerce")
    target = target.rename(columns={"corona_result": "Outcome"})

    # Reorder columns
    cols = [
        "age_60_and_above",
        "gender",
        "cough",
        "fever",
        "sore_throat",
        "shortness_of_breath",
        "head_ache",
        "Outcome",
    ]
    # Return dataframe with dtypes float
    return target[cols].astype(float)


def get_corpus():
    """
    Gets corpus data from hospital admissions data set and select relavent columns.

    Returns: 
        Processed corpus dataset in Dataframe
    """

    # Desired columns
    cols = [
        "age",
        "gender",
        "cc_cough",
        "cc_fever",
        "cc_sorethroat",
        "cc_shortnessofbreath",
        "cc_headache",
        "cc_fever-75yearsorolder",
        "cc_fever-9weeksto74years",
    ]

    # Import corpus file
    corpus = pd.read_csv(CORPUS_FILE)[cols]

    # Change age to binary type for values above 60
    corpus["age"] = np.where(corpus["age"] >= 60, 1, 0)
    # Change age to binary type
    corpus = corpus.replace({"gender": {"Female": 0, "Male": 1}})

    # Combine fever variables into one variable
    corpus["cc_fever"] = (
        corpus["cc_fever"]
        + corpus["cc_fever-75yearsorolder"]
        + corpus["cc_fever-9weeksto74years"]
    )

    # Drop other fever variables
    corpus = corpus.drop(
        columns=["cc_fever-75yearsorolder", "cc_fever-9weeksto74years"]
    )

    # Give columns desired names
    corpus.columns = [
        "age_60_and_above",
        "gender",
        "cough",
        "fever",
        "sore_throat",
        "shortness_of_breath",
        "head_ache",
    ]
    # Drop nans, convert dtypes to float and return dataframe
    return corpus.dropna().astype(float)


def main(lam=0.01, **kwargs):
    # Generate marginal objects using least squares loss function
    import emm

    corpus = get_corpus()
    target = get_target()
    marg_tab = target.groupby("Outcome").mean().T

    margs = {}
    margs[0] = []
    margs[1] = []
    for feature in marg_tab.index:
        margs[0].append(
            emm.reweighting.marginal(
                feature, "mean", emm.losses.LeastSquaresLoss(marg_tab.loc[feature, 0])
            )
        )
        margs[1].append(
            emm.reweighting.marginal(
                feature, "mean", emm.losses.LeastSquaresLoss(marg_tab.loc[feature, 1])
            )
        )

    rwc = emm.reweighting.generate_synth(
        corpus,
        margs,
        regularizer=emm.regularizers.EntropyRegularizer(),
        lam=lam,
        **kwargs
    )
    return rwc


if __name__ == "__main__":
    main()
