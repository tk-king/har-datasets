import math
import random
from typing import List, Tuple

import pandas as pd
from whar_datasets.core.config import WHARConfig
from whar_datasets.core.utils.logging import logger


def get_split_train_test(
    cfg: WHARConfig,
    session_metadata: pd.DataFrame,
    window_metadata: pd.DataFrame,
    subj_cross_val_group_index: int,
) -> Tuple[List[int], List[int]]:
    groups = cfg.subj_cross_val_split_groups
    assert groups is not None
    assert subj_cross_val_group_index < len(groups)

    # get subject_ids for both groups
    test_subj_ids = groups[subj_cross_val_group_index]
    train_subj_ids = [
        subj_id
        for i, group in enumerate(groups)
        if i != subj_cross_val_group_index
        for subj_id in group
    ]

    # get window indices for all groups
    train_indices = get_window_indices(
        session_metadata, window_metadata, train_subj_ids
    )
    test_indices = get_window_indices(session_metadata, window_metadata, test_subj_ids)

    logger.info(f"train: {len(train_indices)} | test: {len(test_indices)}")

    return train_indices, test_indices


def split_indices(
    cfg: WHARConfig, indices: List[int], percentages: Tuple[float, ...]
) -> Tuple[List[int], ...]:
    assert math.isclose(sum(percentages), 1.0)

    # shuffle so subjects are not in order
    random.seed(cfg.seed)
    shuffled_indices = indices.copy()
    random.shuffle(shuffled_indices)

    # calculate split sizes
    nums = [int(p * len(indices)) for p in percentages]

    # adjust last split to take remaining items (to handle rounding)
    nums[-1] = len(indices) - sum(nums[:-1])

    # create splits
    split = []
    start = 0
    for n in nums:
        split.append(shuffled_indices[start : start + n])
        start += n

    return tuple(split)


def get_split_train_val_test(
    cfg: WHARConfig,
    session_metadata: pd.DataFrame,
    window_metadata: pd.DataFrame,
    subj_cross_val_group_index: int | None = None,
) -> Tuple[List[int], List[int], List[int]]:
    # if no split group is specified, use given split
    if subj_cross_val_group_index is None:
        assert cfg.given_train_test_subj_ids is not None

        # get subject_ids for each group
        train_subj_ids, test_subj_ids = cfg.given_train_test_subj_ids

        # get window indices for each group
        train_indices = get_window_indices(
            session_metadata, window_metadata, train_subj_ids
        )

        test_indices = get_window_indices(
            session_metadata, window_metadata, test_subj_ids
        )

        # split train into train and val
        train_indices, val_indices = split_indices(
            cfg, train_indices, (1 - cfg.val_percentage, cfg.val_percentage)
        )

    # if split group is specified, use subject cross validation
    else:
        groups = cfg.subj_cross_val_split_groups
        assert groups is not None
        assert subj_cross_val_group_index < len(groups)

        # get subject_ids for both groups
        test_subj_ids = groups[subj_cross_val_group_index]
        train_subj_ids = [
            subj_id
            for i, group in enumerate(groups)
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

        # split train into train and val
        train_indices, val_indices = split_indices(
            cfg, train_indices, (1 - cfg.val_percentage, cfg.val_percentage)
        )

    logger.info(
        f"train: {len(train_indices)} | val: {len(val_indices)} | test: {len(test_indices)}"
    )

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
