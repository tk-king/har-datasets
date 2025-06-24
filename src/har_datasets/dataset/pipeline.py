import os
from typing import Callable, List, Tuple

import numpy as np
import pandas as pd
from har_datasets.config.config import NON_CHANNEL_COLS, HARConfig, NormType
from har_datasets.config.hashing import create_cfg_hash
from har_datasets.features.checking import check_format
from har_datasets.features.loading import get_df
from har_datasets.features.normalizing import (
    min_max,
    normalize_globally,
    normalize_per_sample,
    normalize_per_subject,
    standardize,
)
from har_datasets.features.resampling import resample
from har_datasets.features.selecting import select_activities, select_channels
from har_datasets.features.spectrogram import get_spectrograms
from har_datasets.features.windowing import (
    generate_windowing,
    load_cfg_hash,
    load_windowing,
    save_windowing,
)


def pipeline(
    cfg: HARConfig, parse: Callable[[str], pd.DataFrame], override_csv: bool = False
) -> Tuple[str, pd.DataFrame, List[pd.DataFrame] | None, List[np.ndarray] | None]:
    # create config hash
    cfg_hash = create_cfg_hash(cfg)

    # define directories
    datasets_dir = cfg.common.datasets_dir
    dataset_dir = os.path.join(datasets_dir, cfg.dataset.info.id)
    windowing_dir = os.path.join(dataset_dir, "windowing/")
    windows_dir = os.path.join(windowing_dir, "windows/")

    # check if windowing exists and corresponds to cfg
    if os.path.exists(windowing_dir) and cfg_hash == load_cfg_hash(windowing_dir):
        # load windowing
        window_index, windows = load_windowing(windowing_dir, windows_dir)

        # load or generate spectrograms
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
            if cfg.common.use_spectrogram
            else None
        )

        # return in-memory windowing or windowing on disk
        if cfg.dataset.training.in_memory:
            return dataset_dir, window_index, windows, spectrograms
        else:
            return dataset_dir, window_index, None, None

    # load or generate dataframe
    df = get_df(
        datasets_dir=datasets_dir,
        dataset_id=cfg.dataset.info.id,
        download_url=cfg.dataset.info.download_url,
        parse=parse,
        override_csv=override_csv,
    )

    # check format of dataframe
    check_format(df=df, required_cols=NON_CHANNEL_COLS)

    # apply resampling to dataframe for equidistant measurements
    df = resample(
        df=df,
        resampling_freq=cfg.dataset.info.sampling_freq,
        exclude_columns=NON_CHANNEL_COLS,
    )

    # apply selections to dataframe
    df = select_activities(
        df=df, activity_names=cfg.dataset.preprocessing.selections.activity_names
    )
    df = select_channels(
        df=df,
        channels=cfg.dataset.preprocessing.selections.sensor_channels,
        exclude_cols=NON_CHANNEL_COLS,
    )

    # apply resampling to dataframe to convert to common sampling frequency
    if cfg.common.resampling_freq is not None:
        df = resample(
            df=df,
            resampling_freq=cfg.common.resampling_freq,
            exclude_columns=NON_CHANNEL_COLS,
        )

    # apply global or per subject normalization to dataframe
    match cfg.dataset.preprocessing.normalization:
        case NormType.STD_GLOBALLY:
            df = normalize_globally(
                df=df,
                normalize=standardize,
                exclude_columns=NON_CHANNEL_COLS,
            )
        case NormType.MIN_MAX_GLOBALLY:
            df = normalize_globally(
                df=df, normalize=min_max, exclude_columns=NON_CHANNEL_COLS
            )
        case NormType.STD_PER_SUBJ:
            df = normalize_per_subject(
                df=df,
                normalize=standardize,
                exclude_columns=NON_CHANNEL_COLS,
            )
        case NormType.MIN_MAX_PER_SUBJ:
            df = normalize_per_subject(
                df=df, normalize=min_max, exclude_columns=NON_CHANNEL_COLS
            )

    # generate windowing
    window_index, windows = generate_windowing(
        df=df,
        window_time=cfg.dataset.preprocessing.sliding_window.window_time,
        overlap=cfg.dataset.preprocessing.sliding_window.overlap,
        exclude_cols=NON_CHANNEL_COLS,
    )

    # apply per sample normalization
    match cfg.dataset.preprocessing.normalization:
        case NormType.STD_PER_SAMPLE:
            windows = normalize_per_sample(
                windows=windows,
                normalize=standardize,
                exclude_columns=NON_CHANNEL_COLS,
            )
        case NormType.MIN_MAX_PER_SAMPLE:
            windows = normalize_per_sample(
                windows=windows,
                normalize=min_max,
                exclude_columns=NON_CHANNEL_COLS,
            )

    # save windowing
    save_windowing(
        cfg_hash=cfg_hash,
        windowing_dir=windowing_dir,
        windows_dir=windows_dir,
        window_index=window_index,
        windows=windows,
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
        if cfg.common.use_spectrogram
        else None
    )

    # return in-memory windowing or windowing on disk
    if cfg.dataset.training.in_memory:
        return dataset_dir, window_index, windows, spectrograms
    else:
        return dataset_dir, window_index, None, None
