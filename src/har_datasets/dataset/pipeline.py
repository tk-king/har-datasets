import os
from typing import Callable, List, Tuple

import numpy as np
import pandas as pd
from har_datasets.config.config import HARConfig, NormType, SplitType
from har_datasets.config.hashing import create_cfg_hash
from har_datasets.pipeline.checking import check_format
from har_datasets.pipeline.loading import load_df
from har_datasets.pipeline.normalizing import (
    min_max,
    normalize_globally,
    normalize_per_subject,
    standardize,
)
from har_datasets.pipeline.resampling import resample
from har_datasets.pipeline.selecting import select_activities, select_channels
from har_datasets.pipeline.spectrogram import get_spectrograms, load_spectrogram
from har_datasets.pipeline.windowing import get_windowing, load_window


def pipeline(
    cfg: HARConfig, parse: Callable[[str], pd.DataFrame], override_csv: bool = False
) -> Tuple[str, pd.DataFrame, List[pd.DataFrame] | None, List[np.ndarray] | None]:
    # create config hash
    cfg_hash = create_cfg_hash(cfg)

    # load dataframe and dir of dataset
    df, dataset_dir = load_df(
        url=cfg.dataset.info.url,
        datasets_dir=cfg.common.datasets_dir,
        csv_file=cfg.dataset.info.id + ".csv",
        parse=parse,
        override_csv=override_csv,
    )

    # check format of dataframe
    check_format(df=df, required_cols=cfg.common.non_channel_cols)

    # apply resampling to dataframe for equidistant measurements
    df = resample(
        df=df,
        resampling_freq=cfg.dataset.info.sampling_freq,
        exclude_columns=cfg.common.non_channel_cols,
    )

    # apply selections to dataframe
    df = select_activities(df=df, activity_names=cfg.dataset.selections.activity_names)
    df = select_channels(
        df=df,
        channels=cfg.dataset.selections.channels,
        exclude_cols=cfg.common.non_channel_cols,
    )

    # apply resampling to dataframe to convert to common sampling frequency
    if cfg.common.resampling_freq is not None:
        df = resample(
            df=df,
            resampling_freq=cfg.common.resampling_freq,
            exclude_columns=cfg.common.non_channel_cols,
        )

    # apply global or per subject normalization to dataframe
    match cfg.common.normalization:
        case NormType.STD_GLOBALLY:
            df = normalize_globally(df, standardize, cfg.common.non_channel_cols)
        case NormType.MIN_MAX_GLOBALLY:
            df = normalize_globally(df, min_max, cfg.common.non_channel_cols)
        case NormType.STD_PER_SUBJ:
            df = normalize_per_subject(df, standardize, cfg.common.non_channel_cols)
        case NormType.MIN_MAX_PER_SUBJ:
            df = normalize_per_subject(df, min_max, cfg.common.non_channel_cols)

    # get window_index and windows
    window_index, windows = get_windowing(
        dataset_dir=dataset_dir,
        cfg_hash=cfg_hash,
        df=df,
        window_time=cfg.common.sliding_window.window_time,
        overlap=cfg.common.sliding_window.overlap,
        exclude_cols=cfg.common.non_channel_cols,
        normalization=cfg.common.normalization,
    )

    # get spectrograms
    spectrograms = (
        get_spectrograms(
            dataset_dir=dataset_dir,
            cfg_hash=cfg_hash,
            window_index=window_index,
            windows=windows,
            sampling_freq=cfg.dataset.info.sampling_freq,
            window_size=cfg.common.spectrogram.window_size,
            overlap=cfg.common.spectrogram.overlap,
            mode=cfg.common.spectrogram.mode,
        )
        if cfg.common.spectrogram.use_spectrogram
        else None
    )

    if cfg.dataset.in_memory:
        return dataset_dir, window_index, windows, spectrograms
    else:
        return dataset_dir, window_index, None, None


def split(
    cfg: HARConfig, window_index: pd.DataFrame
) -> Tuple[List[int], List[int], List[int]]:
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

    return train_indices, test_indices, val_indices


def get_sample(
    cfg: HARConfig,
    index: int,
    dataset_dir: str,
    window_index: pd.DataFrame,
    windows: List[pd.DataFrame] | None,
    spectograms: List[np.ndarray] | None,
) -> Tuple[np.integer, np.ndarray, np.ndarray | None]:
    # get class label of window
    label = window_index.loc[index]["activity_id"]
    assert isinstance(label, np.integer)

    # get window_id
    window_id = window_index.loc[index]["window_id"]
    assert isinstance(window_id, np.integer)

    if cfg.dataset.in_memory:
        assert windows is not None

        # get window and spectogram
        window = windows[window_id].values
        spect = (
            spectograms[window_id]
            if cfg.common.spectrogram.use_spectrogram and spectograms is not None
            else None
        )
    else:
        # defined dirs
        windowing_dir = os.path.join(dataset_dir, "windowing/")
        windows_dir = os.path.join(windowing_dir, "windows/")
        spectograms_dir = os.path.join(windowing_dir, "spectograms/")

        # load window and spectogram
        window = load_window(windows_dir, int(window_id)).values
        spect = (
            load_spectrogram(spectograms_dir, int(window_id))
            if cfg.common.spectrogram.use_spectrogram
            else None
        )

    return label, window, spect
