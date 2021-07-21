"""
common.py
====================
Reused functions for downloading and processing data.
"""
import os
from sklearn.model_selection import train_test_split


def make_directory_if_not_exists(location):
    """Make a directory at a given location if it doesnt already exist."""
    if not os.path.isdir(location):
        os.mkdir(location)


def create_splits(
    frame,
    ratios=(0.4, 0.4, 0.2),
    shuffle=True,
    stratify=True,
    random_state=1,
):
    """Creates three splits for the data according to the specified ratios.

    Arguments:
        frame: The dataframe to split.
        ratios: The ratios for which to make the splits, must sum to 1.
        shuffle: Whether to first shuffle the dataset.
        stratify: Whether to stratify the splits. If this is set, there must be a column called 'label' that will be
            stratified on.
        random_state: Seed for reproducibility.

    Returns:
        Three dataframes that have been split according to the corresponding sizes.
    """
    assert sum(ratios) == 1
    stratify_indexes = None
    if stratify:
        assert (
            "label" in frame.columns
        ), "Function expects a label column if we are stratifying."
        stratify_indexes = frame["label"]

    # Perform the first split to get the test data
    inner_frame, frame_3 = train_test_split(
        frame,
        train_size=sum(ratios[:-1]),
        test_size=ratios[-1],
        stratify=stratify_indexes,
        shuffle=shuffle,
        random_state=random_state,
    )

    # Perform the statistical/train split
    if stratify:
        stratify_indexes = inner_frame["label"]
    frame_1, frame_2 = train_test_split(
        inner_frame,
        train_size=ratios[0],
        test_size=ratios[1],
        stratify=stratify_indexes,
        shuffle=shuffle,
        random_state=random_state,
    )

    return frame_1, frame_2, frame_3
