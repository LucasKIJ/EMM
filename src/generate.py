"""
generate.py
========================
To contain methods for generating data from marginals.

This should contain at least:
    1. An rsw method using a corpus.
    2. Generation as independent normals.
    3. Another synthetic generation method.
They should all accept and produce data in the same format!!!
"""


def generate_weighted_dataset_from_marginals(corpus_dataframe, target_statistics):
    """This function returns a weighted version of the corpus dataset that matches (close to) some target statistics.

    This needs to accept a generic form of target statistics, check the corpus has the relevant columns (else raise
    an error) and then setup the rsw problem to tune the weights. Finally return the re-weighted corpus.

    Arguments:
        corpus_dataframe: The corpus, i.e. a dataframe of all the real patient data we have access to.
        target_statistics: The statistics (means/vars) that we seek to optimise towards.
    """
    pass


def generate_dataset_using_independent_normals(target_statistics):
    """Generates a dataset using the independent standard normal assumption."""
    pass


def generate_dataset_using_some_better_method(target_statistics):
    """Generates a dataset using an improved synthetic generation method."""
    pass
