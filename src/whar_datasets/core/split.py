from typing import List, Tuple

import pandas as pd
from whar_datasets.core.config import WHARConfig


def get_split(
    cfg: WHARConfig,
    window_index: pd.DataFrame,
    subj_cross_val_group_index: int | None = None,
) -> Tuple[List[int], List[int], List[int]]:
    # if no split group is specified, use given split
    if subj_cross_val_group_index is None:
        split_g = cfg.dataset.training.split.given_split
        assert split_g is not None

        train_indices = window_index[
            window_index["subject_id"].isin(split_g.train_subj_ids)
        ].index.to_list()

        val_indices = window_index[
            window_index["subject_id"].isin(split_g.val_subj_ids)
        ].index.to_list()

        test_indices = window_index[
            window_index["subject_id"].isin(split_g.test_subj_ids)
        ].index.to_list()

    # if split group is specified, use subject cross validation
    else:
        split_scv = cfg.dataset.training.split.subj_cross_val_split
        assert split_scv is not None

        assert subj_cross_val_group_index < len(split_scv.subj_id_groups)

        val_subj_ids = split_scv.subj_id_groups[subj_cross_val_group_index]

        train_indices = window_index[
            ~window_index["subject_id"].isin(val_subj_ids)
        ].index.to_list()

        val_indices = window_index[
            window_index["subject_id"].isin(val_subj_ids)
        ].index.to_list()

        test_indices = []

    return train_indices, val_indices, test_indices
