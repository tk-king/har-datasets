from typing import List, Tuple

import pandas as pd
from whar_datasets.core.config import WHARConfig


def get_group_indices(
    session_index: pd.DataFrame, window_index: pd.DataFrame, group: List[int]
) -> List[int]:
    sessions = session_index[session_index["subject_id"].isin(group)]
    relevant_session_ids = sessions["session_id"].unique()
    relevant_windows = window_index[
        window_index["session_id"].isin(relevant_session_ids)
    ]
    return relevant_windows.index.to_list()


def get_split(
    cfg: WHARConfig,
    session_index: pd.DataFrame,
    window_index: pd.DataFrame,
    subj_cross_val_group_index: int | None = None,
) -> Tuple[List[int], List[int], List[int]]:
    # if no split group is specified, use given split
    if subj_cross_val_group_index is None:
        split_g = cfg.dataset.training.split.given_split
        assert split_g is not None

        train_indices = get_group_indices(
            session_index, window_index, split_g.train_subj_ids
        )

        val_indices = get_group_indices(
            session_index, window_index, split_g.val_subj_ids
        )

        test_indices = get_group_indices(
            session_index, window_index, split_g.test_subj_ids
        )

    # if split group is specified, use subject cross validation
    else:
        split_scv = cfg.dataset.training.split.subj_cross_val_split
        assert split_scv is not None

        assert subj_cross_val_group_index < len(split_scv.subj_id_groups)

        val_subj_ids = split_scv.subj_id_groups[subj_cross_val_group_index]

        val_indices = get_group_indices(session_index, window_index, val_subj_ids)

        train_indices = [
            i for i in window_index.index.to_list() if i not in val_indices
        ]

        test_indices = []

    return train_indices, val_indices, test_indices
