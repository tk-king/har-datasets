import os
from typing import Dict, Tuple
import pandas as pd
from dask.delayed import delayed
from dask.base import compute
from tqdm import tqdm
from dask.diagnostics.progress import ProgressBar

from whar_datasets.core.config import WHARConfig
from whar_datasets.core.utils.checking import (
    check_download,
    check_sessions,
    check_windowing,
)
from whar_datasets.core.utils.validation import validate_common_format
from whar_datasets.core.utils.downloading import download, extract
from whar_datasets.core.utils.resampling import resample
from whar_datasets.core.utils.selecting import select_activities, select_channels
from whar_datasets.core.utils.windowing import generate_windowing
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


def preprocess(cfg: WHARConfig, override_cache: bool = False) -> Tuple[str, str, str]:
    if override_cache:
        print("Overriding cache...")

    # create config hash
    cfg_hash = create_cfg_hash(cfg)

    # define directories
    datasets_dir = cfg.datasets_dir
    dataset_dir = os.path.join(datasets_dir, cfg.dataset_id)
    cache_dir = os.path.join(dataset_dir, "cache/")
    sessions_dir = os.path.join(cache_dir, "sessions/")
    windows_dir = os.path.join(cache_dir, "windows/")
    hashes_dir = os.path.join(cache_dir, "hashes/")

    # if not yet done, download and extract
    if not check_download(dataset_dir):
        file_path = download(datasets_dir, dataset_dir, cfg.download_url)
        extract(file_path, dataset_dir)

    # if not yet done, parse and cache common format
    if not check_sessions(cache_dir, sessions_dir) or override_cache:
        print("Parsing...")
        activity_metadata, session_metadata, sessions = cfg.parse(
            dataset_dir, cfg.activity_id_col
        )
        cache_common_format(
            cache_dir, sessions_dir, activity_metadata, session_metadata, sessions
        )

    # check if parser respects common format
    assert validate_common_format(cfg, cache_dir, sessions_dir)

    # if windowing not up-to-date, generate and cache windowing
    if (
        not check_windowing(cache_dir, windows_dir, hashes_dir, cfg_hash)
        or override_cache
    ):
        session_metadata = load_session_metadata(cache_dir)
        activity_metadata = load_activity_metadata(cache_dir)

        # select activities
        session_metadata = select_activities(
            session_metadata,
            activity_metadata,
            cfg.activity_names,
        )

        # generate windowing
        window_metadata, windows = (
            process_sessions_parallely(cfg, sessions_dir, session_metadata)
            if cfg.in_parallel
            else process_sessions_sequentially(cfg, sessions_dir, session_metadata)
        )

        # cache windows
        cache_window_metadata(cache_dir, window_metadata)
        cache_windows(windows_dir, window_metadata, windows)
        cache_cfg_hash(hashes_dir, cfg_hash)

    return cache_dir, windows_dir, hashes_dir


def process_sessions_sequentially(
    cfg: WHARConfig, sessions_dir: str, session_metadata: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    # loop over sessions
    loop = tqdm([int(x) for x in session_metadata["session_id"].unique()])
    loop.set_description("Processing sessions")

    pairs = [process_session(cfg, sessions_dir, session_id) for session_id in loop]
    window_metadatas, window_dicts = zip(*pairs)

    # compute global window metadata and windows
    window_metadata = pd.concat([w for w in window_metadatas if w is not None])
    window_metadata.reset_index(drop=True, inplace=True)
    windows = {k: v for d in window_dicts if d is not None for k, v in d.items()}

    # assert uniqueness of window ids
    assert window_metadata["window_id"].nunique() == len(window_metadata)

    return window_metadata, windows


def process_sessions_parallely(
    cfg: WHARConfig, sessions_dir: str, session_metadata: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    @delayed
    def process_session_delayed(
        session_id: int,
    ) -> Tuple[pd.DataFrame | None, Dict[str, pd.DataFrame] | None]:
        return process_session(cfg, sessions_dir, session_id)

    # define processing tasks
    tasks = [
        process_session_delayed(session_id)
        for session_id in [int(x) for x in session_metadata["session_id"].unique()]
    ]

    # execute tasks in parallel
    pbar = ProgressBar()
    pbar.register()
    pairs = list(compute(*tasks, scheduler="processes"))
    pbar.unregister()

    window_metadatas, window_dicts = zip(*pairs)

    # compute global window metadata and windows
    window_metadata = pd.concat([w for w in window_metadatas if w is not None])
    window_metadata.reset_index(drop=True, inplace=True)
    windows = {k: v for d in window_dicts if d is not None for k, v in d.items()}

    # assert uniqueness of window ids
    assert window_metadata["window_id"].nunique() == len(window_metadata)

    return window_metadata, windows


def process_session(
    cfg: WHARConfig,
    sessions_dir: str,
    session_id: int,
) -> Tuple[pd.DataFrame | None, Dict[str, pd.DataFrame] | None]:
    assert isinstance(session_id, int)

    # load session
    session_path = os.path.join(sessions_dir, f"session_{session_id}.parquet")
    session_df = pd.read_parquet(session_path)

    # apply selections
    session_df = select_channels(session_df, cfg.sensor_channels)

    # resample
    session_df = resample(session_df, cfg.sampling_freq)

    # generate windowing
    window_metadata, windows = generate_windowing(
        session_id,
        session_df,
        cfg.window_time,
        cfg.window_overlap,
        cfg.sampling_freq,
    )

    if window_metadata is None or windows is None:
        return None, None

    return window_metadata, windows
