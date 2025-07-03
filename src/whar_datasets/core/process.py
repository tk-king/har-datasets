import os
from typing import Callable, List, Tuple
import pandas as pd
from dask.delayed import delayed
from dask.base import compute

from whar_datasets.core.config import NormType, WHARConfig
from whar_datasets.core.steps.checking import (
    check_common_format,
    check_download,
    check_windowing,
)
from whar_datasets.core.steps.downloading import download, extract
from whar_datasets.core.steps.normalizing import (
    min_max,
    normalize_per_sample,
    standardize,
)
from whar_datasets.core.steps.resampling import resample
from whar_datasets.core.steps.selecting import select_activities, select_channels
from whar_datasets.core.steps.windowing import generate_windowing
from whar_datasets.core.utils.caching import (
    cache_common_format,
    cache_window_index,
    cache_windows,
)
from whar_datasets.core.utils.loading import load_session_index


def process(
    cfg: WHARConfig,
    parse: Callable[[str, str], Tuple[pd.DataFrame, pd.DataFrame, List[pd.DataFrame]]],
    override_cache: bool = False,
) -> Tuple[str, str]:
    # define directories
    datasets_dir = cfg.common.datasets_dir
    dataset_dir = os.path.join(datasets_dir, cfg.dataset.info.id)
    cache_dir = os.path.join(dataset_dir, "cache/")
    sessions_dir = os.path.join(dataset_dir, "sessions/")
    windows_dir = os.path.join(dataset_dir, "windows/")

    # check if windowing is up-to-date
    if check_windowing(cache_dir, windows_dir, cfg) and not override_cache:
        return cache_dir, windows_dir

    # if not yet done, download and extract
    if not check_download(dataset_dir, cfg):
        # download dataset file
        file_path = download(datasets_dir, dataset_dir, cfg.dataset.info.download_url)

        # extract all archives
        extract(file_path, dataset_dir)

    # if not yet done, parse and cache common format
    if not check_common_format(cache_dir, sessions_dir) or override_cache:
        # parse original dataset to common format
        activity_index, session_index, sessions = parse(
            dataset_dir, cfg.dataset.preprocessing.activity_id_col
        )

        # cache common format
        cache_common_format(
            cache_dir, sessions_dir, activity_index, session_index, sessions
        )

    # load session index
    session_index = load_session_index(cache_dir)

    # define processing tasks
    tasks = [
        process_session(cfg, sessions_dir, windows_dir, session_id)
        for session_id in session_index["session_id"]
    ]

    # execute tasks in parallel
    window_indexs = list(compute(*tasks))
    assert isinstance(window_indexs, list)

    # cache window index
    cache_window_index(windows_dir, window_indexs)

    return cache_dir, windows_dir


@delayed
def process_session(
    cfg: WHARConfig, sessions_dir: str, windows_dir: str, session_id: int
) -> pd.DataFrame:
    session_path = os.path.join(sessions_dir, f"session_{session_id}.parquet")
    session_df = pd.read_parquet(session_path)

    # apply selections
    sessions_df = select_activities(
        session_df, cfg.dataset.preprocessing.selections.activity_names
    )
    sessions_df = select_channels(
        sessions_df, cfg.dataset.preprocessing.selections.sensor_channels
    )

    # resample
    sessions_df = resample(sessions_df, cfg.dataset.info.sampling_freq)

    # generate windowing
    window_index, windows = generate_windowing(
        sessions_df,
        session_id,
        cfg.dataset.preprocessing.sliding_window.window_time,
        cfg.dataset.preprocessing.sliding_window.overlap,
    )

    # normalize per window
    normalize = None

    match cfg.dataset.preprocessing.normalization:
        case NormType.STD_PER_SAMPLE:
            normalize = standardize

        case NormType.MIN_MAX_PER_SAMPLE:
            normalize = min_max

    windows = normalize_per_sample(windows, normalize)

    # save windowing
    cache_windows(windows_dir, window_index, windows)

    return window_index
