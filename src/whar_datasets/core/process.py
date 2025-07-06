import os
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from dask.delayed import delayed
from dask.base import compute
from tqdm import tqdm
from dask.diagnostics.progress import ProgressBar

from whar_datasets.core.config import NormType, WHARConfig
from whar_datasets.core.utils.checking import (
    check_common_format,
    check_download,
    check_windowing,
)
from whar_datasets.core.utils.downloading import download, extract
from whar_datasets.core.steps.normalizing import (
    min_max,
    normalize_per_sample,
    standardize,
)
from whar_datasets.core.steps.resampling import resample
from whar_datasets.core.steps.selecting import select_activities, select_channels
from whar_datasets.core.steps.windowing import generate_windowing
from whar_datasets.core.utils.caching import (
    cache_cfg_hash,
    cache_common_format,
    cache_window_metadata,
    cache_windows,
)
from whar_datasets.core.utils.hashing import create_cfg_hash
from whar_datasets.core.utils.loading import (
    load_activity_metadata,
    load_session_metadata,
)


def process(cfg: WHARConfig, override_cache: bool = False) -> Tuple[str, str]:
    if override_cache:
        print("Overriding cache...")

    # create config hash
    cfg_hash = create_cfg_hash(cfg)

    # define directories
    datasets_dir = cfg.common.datasets_dir
    dataset_dir = os.path.join(datasets_dir, cfg.dataset.info.id)
    cache_dir = os.path.join(dataset_dir, "cache/")
    sessions_dir = os.path.join(cache_dir, "sessions/")
    windows_dir = os.path.join(cache_dir, "windows/")

    # check if windowing is up-to-date
    if check_windowing(cache_dir, windows_dir, cfg_hash) and not override_cache:
        return cache_dir, windows_dir

    # if not yet done, download and extract
    if not check_download(dataset_dir):
        # download dataset file
        file_path = download(datasets_dir, dataset_dir, cfg.dataset.info.download_url)

        # extract all archives
        extract(file_path, dataset_dir)

    # if not yet done, parse and cache common format
    if not check_common_format(cfg, cache_dir, sessions_dir) or override_cache:
        # parse original dataset to common format
        print("Parsing...")
        activity_metadata, session_metadata, sessions = cfg.dataset.parsing.parse(
            dataset_dir, cfg.dataset.parsing.activity_id_col
        )

        # cache common format
        cache_common_format(
            cache_dir, sessions_dir, activity_metadata, session_metadata, sessions
        )

    assert check_common_format(cfg, cache_dir, sessions_dir)

    # load session and activity index
    session_metadata = load_session_metadata(cache_dir)
    activity_metadata = load_activity_metadata(cache_dir)

    # select sessions with selected activities
    session_metadata = select_activities(
        session_metadata,
        activity_metadata,
        cfg.dataset.preprocessing.selections.activity_names,
    )

    # process sessions
    if cfg.dataset.preprocessing.in_parallel:
        window_metadata, windows = process_sessions_parallely(
            cfg, sessions_dir, session_metadata
        )
    else:
        window_metadata, windows = process_sessions_sequentially(
            cfg, sessions_dir, session_metadata
        )

    cache_window_metadata(cache_dir, window_metadata)
    cache_windows(windows_dir, window_metadata, windows)
    cache_cfg_hash(cache_dir, cfg_hash)

    return cache_dir, windows_dir


def process_sessions_sequentially(
    cfg: WHARConfig, sessions_dir: str, session_metadata: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    # create empty lists
    window_metadatas = []
    window_dicts = []

    # loop over sessions
    loop = tqdm(session_metadata["session_id"])
    loop.set_description("Processing sessions")

    for session_id in loop:
        assert isinstance(session_id, np.integer)

        # load session
        session_path = os.path.join(sessions_dir, f"session_{session_id}.parquet")
        session_df = pd.read_parquet(session_path)

        # select only configured channels
        session_df = select_channels(
            session_df, cfg.dataset.preprocessing.selections.sensor_channels
        )

        # resample
        session_df = resample(session_df, cfg.dataset.info.sampling_freq)

        # generate windowing
        window_metadata, windows = generate_windowing(
            session_id,
            session_df,
            cfg.dataset.preprocessing.sliding_window.window_time,
            cfg.dataset.preprocessing.sliding_window.overlap,
            cfg.common.resampling_freq or cfg.dataset.info.sampling_freq,
        )

        # skip if no window could be generated
        if window_metadata is None or windows is None:
            continue

        # select normalization
        normalize = None
        match cfg.dataset.preprocessing.normalization:
            case NormType.STD_PER_SAMPLE:
                normalize = standardize
            case NormType.MIN_MAX_PER_SAMPLE:
                normalize = min_max
            case _:
                normalize = None

        # normalize windows
        windows = normalize_per_sample(windows, normalize)

        # append to lists
        window_metadatas.append(window_metadata)
        window_dicts.append(windows)

    # compute global window metadata and windows
    window_metadata = pd.concat(window_metadatas, ignore_index=True)
    window_metadata.reset_index(drop=True, inplace=True)
    windows = {k: v for d in window_dicts for k, v in d.items()}

    # assert uniqueness of window ids
    assert window_metadata["window_id"].nunique() == len(window_metadata)

    return window_metadata, windows


def process_sessions_parallely(
    cfg: WHARConfig, sessions_dir: str, session_metadata: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    @delayed
    def process_session(
        cfg: WHARConfig,
        sessions_dir: str,
        session_id: int,
    ) -> Tuple[pd.DataFrame | None, Dict[str, pd.DataFrame] | None]:
        assert isinstance(session_id, np.integer)

        # load session
        session_path = os.path.join(sessions_dir, f"session_{session_id}.parquet")
        session_df = pd.read_parquet(session_path)

        # apply selections
        session_df = select_channels(
            session_df, cfg.dataset.preprocessing.selections.sensor_channels
        )

        # resample
        session_df = resample(session_df, cfg.dataset.info.sampling_freq)

        # generate windowing
        window_metadata, windows = generate_windowing(
            session_id,
            session_df,
            cfg.dataset.preprocessing.sliding_window.window_time,
            cfg.dataset.preprocessing.sliding_window.overlap,
            cfg.common.resampling_freq or cfg.dataset.info.sampling_freq,
        )

        if window_metadata is None or windows is None:
            return None, None

        # normalize per window
        normalize = None
        match cfg.dataset.preprocessing.normalization:
            case NormType.STD_PER_SAMPLE:
                normalize = standardize
            case NormType.MIN_MAX_PER_SAMPLE:
                normalize = min_max
            case _:
                normalize = None

        windows = normalize_per_sample(windows, normalize)

        return window_metadata, windows

    # define processing tasks
    ProgressBar().register()
    tasks = [
        process_session(cfg, sessions_dir, session_id)
        for session_id in session_metadata["session_id"].unique()
    ]

    # execute tasks in parallel
    pairs = list(compute(*tasks))
    window_metadatas, window_dicts = zip(*pairs)

    # compute global window metadata and windows
    window_metadata = pd.concat([w for w in window_metadatas if w is not None])
    window_metadata.reset_index(drop=True, inplace=True)
    windows: Dict[str, pd.DataFrame] = {
        k: v for d in window_dicts if d is not None for k, v in d.items()
    }

    # assert uniqueness of window ids
    assert window_metadata["window_id"].nunique() == len(window_metadata)

    return window_metadata, windows
