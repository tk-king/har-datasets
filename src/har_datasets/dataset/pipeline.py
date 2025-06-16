from typing import Callable, List, Tuple

import pandas as pd
from har_datasets.config.config import HARConfig, NormType, SplitType
from har_datasets.pipeline.checking import check_format
from har_datasets.pipeline.loading import load_df
from har_datasets.pipeline.normalizing import (
    min_max,
    normalize_globally,
    normalize_per_sample,
    normalize_per_subject,
    standardize,
)
from har_datasets.pipeline.selecting import select_activities, select_channels
from har_datasets.pipeline.windowing import generate_windows


def pipeline(
    cfg: HARConfig, parse: Callable[[str], pd.DataFrame], override_csv: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame, List[pd.DataFrame]]:
    print("1. loading dataframe...")

    # load dataframe
    df = load_df(
        url=cfg.dataset.info.url,
        datasets_dir=cfg.common.datasets_dir,
        csv_file=cfg.dataset.info.id + ".csv",
        parse=parse,
        override_csv=override_csv,
    )

    print("2. checking format...")

    # check format
    check_format(df=df, required_cols=cfg.common.non_channel_cols)

    print("3. applying selections...")

    # apply selections
    df = select_activities(df=df, activity_names=cfg.dataset.selections.activity_names)
    df = select_channels(
        df=df,
        channels=cfg.dataset.selections.channels,
        exclude_cols=cfg.common.non_channel_cols,
    )

    # # apply resampling
    # if cfg.common.resampling_freq is not None:
    #     df = resample(
    #         df=df,
    #         sampling_freq=cfg.dataset.sampling_freq,
    #         resampling_freq=cfg.common.resampling_freq,
    #     )

    print("4. applying normalizations...")

    # apply global or per subject normalization
    match cfg.common.normalization:
        case NormType.STD_GLOBALLY:
            df = normalize_globally(df, standardize, cfg.common.non_channel_cols)
        case NormType.MIN_MAX_GLOBALLY:
            df = normalize_globally(df, min_max, cfg.common.non_channel_cols)
        case NormType.STD_PER_SUBJ:
            df = normalize_per_subject(df, standardize, cfg.common.non_channel_cols)
        case NormType.MIN_MAX_PER_SUBJ:
            df = normalize_per_subject(df, min_max, cfg.common.non_channel_cols)

    print("5. generating windows...")

    # generate windows and window index
    window_index, windows = generate_windows(
        df=df,
        window_time=cfg.common.sliding_window.window_time,
        overlap=cfg.common.sliding_window.overlap,
        exclude_cols=cfg.common.non_channel_cols,
    )

    print("6. applying per sample normalizations...")

    # apply per sample normalization
    match cfg.common.normalization:
        case NormType.STD_PER_SAMPLE:
            windows = normalize_per_sample(
                windows, standardize, cfg.common.non_channel_cols
            )
        case NormType.MIN_MAX_PER_SAMPLE:
            windows = normalize_per_sample(
                windows, min_max, cfg.common.non_channel_cols
            )

    return df, window_index, windows


def split(
    cfg: HARConfig, window_index: pd.DataFrame
) -> Tuple[List[int], List[int], List[int]]:
    print("7. splitting into train, test and val...")

    # specify split indices depending on split type
    match cfg.dataset.split.split_type:
        case SplitType.GIVEN:
            split_g = cfg.dataset.split.given_split
            assert split_g is not None

            train_indices = window_index[
                window_index["subject_id"].isin(split_g.train_subj_ids)
            ].index.to_list()

            test_indices = window_index[
                window_index["subject_id"].isin(split_g.test_subj_ids)
            ].index.to_list()

            val_indices = window_index[
                window_index["subject_id"].isin(split_g.val_subj_ids)
            ].index.to_list()

        case SplitType.SUBJ_CROSS_VAL:
            split_scv = cfg.dataset.split.subj_cross_val_split
            assert split_scv is not None

            test_subj_ids = split_scv.subj_id_groups[split_scv.subj_id_group_index]

            train_indices = window_index[
                ~window_index["subject_id"].isin(test_subj_ids)
            ].index.to_list()

            test_indices = window_index[
                window_index["subject_id"].isin(test_subj_ids)
            ].index.to_list()

            val_indices = []

    print("7. done")

    return train_indices, test_indices, val_indices
