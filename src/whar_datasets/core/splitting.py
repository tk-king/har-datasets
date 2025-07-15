import random
from typing import List, Tuple

import pandas as pd
from whar_datasets.core.config import WHARConfig


def get_split(
    cfg: WHARConfig,
    session_metadata: pd.DataFrame,
    window_metadata: pd.DataFrame,
    subj_cross_val_group_index: int | None = None,
) -> Tuple[List[int], List[int], List[int]]:
    # if no split group is specified, use given split
    if subj_cross_val_group_index is None:
        assert cfg.dataset.training.split.given_split is not None

        # get subject_ids for each group
        train_subj_ids = cfg.dataset.training.split.given_split.train_subj_ids
        test_subj_ids = cfg.dataset.training.split.given_split.test_subj_ids

        # get window indices for each group
        train_indices = get_window_indices(
            session_metadata, window_metadata, train_subj_ids
        )
        test_indices = get_window_indices(
            session_metadata, window_metadata, test_subj_ids
        )

        # shuffle train set
        random.seed(cfg.dataset.training.seed)
        shuffled_train_indices = train_indices.copy()
        random.shuffle(shuffled_train_indices)

        # split train into train and val
        num_val_indices = int(
            cfg.dataset.training.split.val_percentage * len(train_indices)
        )
        val_indices = shuffled_train_indices[:num_val_indices]
        train_indices = shuffled_train_indices[num_val_indices:]

    # if split group is specified, use subject cross validation
    else:
        split = cfg.dataset.training.split.subj_cross_val_split
        assert split is not None
        assert subj_cross_val_group_index < len(split.subj_id_groups)

        # get subject_ids for both groups
        test_subj_ids = split.subj_id_groups[subj_cross_val_group_index]
        train_subj_ids = [
            subj_id
            for i, group in enumerate(split.subj_id_groups)
            if i != subj_cross_val_group_index
            for subj_id in group
        ]

        # get window indices for all groups
        train_indices = get_window_indices(
            session_metadata, window_metadata, train_subj_ids
        )
        test_indices = get_window_indices(
            session_metadata, window_metadata, test_subj_ids
        )

        # shuffle train set
        random.seed(cfg.dataset.training.seed)
        shuffled_train_indices = train_indices.copy()
        random.shuffle(shuffled_train_indices)

        # split train into train and val
        num_val_indices = int(
            cfg.dataset.training.split.val_percentage * len(train_indices)
        )
        val_indices = shuffled_train_indices[:num_val_indices]
        train_indices = shuffled_train_indices[num_val_indices:]

    return train_indices, val_indices, test_indices


def get_window_indices(
    session_metadata: pd.DataFrame,
    window_metadata: pd.DataFrame,
    subject_ids: List[int],
) -> List[int]:
    # get session_ids of sessions with subjects in group
    sessions = session_metadata[session_metadata["subject_id"].isin(subject_ids)]
    session_ids = sessions["session_id"].unique()

    # get window indices of sessions with subjects in group
    windows = window_metadata[window_metadata["session_id"].isin(session_ids)]
    window_indices = windows.index.to_list()

    return window_indices
