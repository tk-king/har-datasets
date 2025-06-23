from typing import List, Tuple

import pandas as pd
from har_datasets.config.config import HARConfig, SplitType


def get_split(
    cfg: HARConfig,
    window_index: pd.DataFrame,
    subj_cross_val_group_index: int | None = None,
) -> Tuple[List[int], List[int], List[int]]:
    # specify split indices depending on split type
    match cfg.dataset.split.split_type:
        case SplitType.GIVEN:
            split_g = cfg.dataset.split.given_split
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

        case SplitType.SUBJ_CROSS_VAL:
            split_scv = cfg.dataset.split.subj_cross_val_split
            assert split_scv is not None

            assert subj_cross_val_group_index is not None
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
