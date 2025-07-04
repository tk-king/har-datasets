import os
import shutil
from typing import Callable, List, Tuple
import pandas as pd
from dask.delayed import delayed
from dask.base import compute
from tqdm import tqdm
from dask.diagnostics.progress import ProgressBar

from whar_datasets.core.config import NormType, WHARConfig
from whar_datasets.core.utils.checking import (
    check_common_format,
    check_download,
    check_sessions,
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
    cache_window_index,
    cache_windows,
)
from whar_datasets.core.utils.hashing import create_cfg_hash
from whar_datasets.core.utils.loading import load_activity_index, load_session_index


def process(
    cfg: WHARConfig,
    parse: Callable[[str, str], Tuple[pd.DataFrame, pd.DataFrame, List[pd.DataFrame]]],
    override_cache: bool = False,
) -> Tuple[str, str]:
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
    if not check_sessions(cache_dir, sessions_dir) or override_cache:
        # parse original dataset to common format
        activity_index, session_index, sessions = parse(
            dataset_dir, cfg.dataset.preprocessing.activity_id_col
        )

        # cache common format
        cache_common_format(
            cache_dir, sessions_dir, activity_index, session_index, sessions
        )

        assert check_common_format(cache_dir, sessions_dir)

    # load session and activity index
    session_index = load_session_index(cache_dir)
    activity_index = load_activity_index(cache_dir)

    # select sessions with selected activities
    session_index = select_activities(
        session_index,
        activity_index,
        cfg.dataset.preprocessing.selections.activity_names,
    )

    # process sessions
    if cfg.dataset.preprocessing.in_parallel:
        window_index, windows = process_sessions_parallely(
            cfg, sessions_dir, session_index
        )
    else:
        window_index, windows = process_sessions_sequentially(
            cfg, sessions_dir, session_index
        )

    # delete windowing directory if it exists
    if os.path.exists(windows_dir):
        shutil.rmtree(windows_dir)

    cache_window_index(cache_dir, window_index)
    cache_windows(windows_dir, window_index, windows)
    cache_cfg_hash(cache_dir, cfg_hash)

    return cache_dir, windows_dir


def process_sessions_sequentially(
    cfg: WHARConfig, sessions_dir: str, session_index: pd.DataFrame
) -> Tuple[pd.DataFrame, List[pd.DataFrame]]:
    window_index_list = []
    windows = []

    # loop over sessions
    loop = tqdm(session_index["session_id"])
    loop.set_description("Processing sessions")

    for session_id in loop:
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
        wi, w = generate_windowing(
            session_df,
            int(session_id),
            cfg.dataset.preprocessing.sliding_window.window_time,
            cfg.dataset.preprocessing.sliding_window.overlap,
        )

        # skip if no window could be generated
        if wi is None or w is None:
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
        w = normalize_per_sample(w, normalize)

        # append to lists
        window_index_list.append(wi)
        windows.extend(w)

    # compute global window index
    window_index = pd.concat(window_index_list)

    return window_index, windows


def process_sessions_parallely(
    cfg: WHARConfig, sessions_dir: str, session_index: pd.DataFrame
):
    @delayed
    def process_session(
        cfg: WHARConfig,
        sessions_dir: str,
        session_id: int,
    ) -> Tuple[pd.DataFrame | None, List[pd.DataFrame] | None]:
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
        window_index, windows = generate_windowing(
            session_df,
            session_id,
            cfg.dataset.preprocessing.sliding_window.window_time,
            cfg.dataset.preprocessing.sliding_window.overlap,
        )

        if window_index is None or windows is None:
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

        return window_index, windows

    # define processing tasks
    ProgressBar().register()
    tasks = [
        process_session(cfg, sessions_dir, session_id)
        for session_id in session_index["session_id"].unique()
    ]

    # execute tasks in parallel
    pairs = list(compute(*tasks))
    window_indexes, windowss = zip(*pairs)

    # assert isinstance(window_indexes, list)
    window_index = pd.concat([wi for wi in window_indexes if wi is not None])
    windows = [item for sublist in windowss if sublist is not None for item in sublist]

    return window_index, windows


# session_dfs = []
# for session_id in session_index["session_id"]:
#     session_df = pd.read_parquet(
#         os.path.join(sessions_dir, f"session_{session_id}.parquet")
#     )
#     session_df["session_id"] = session_id
#     session_dfs.append(session_df)
# df = pd.concat(session_dfs)

# # apply selections
# df = select_channels(df, cfg.dataset.preprocessing.selections.sensor_channels)

# window_indexes = []
# windows = []

# loop = tqdm(df.groupby("session_id"))
# loop.set_description("Generating windows")

# for session_id, group in loop:
#     assert isinstance(session_id, int)

#     group = resample(group, cfg.dataset.info.sampling_freq)
#     wi, w = generate_windowing(
#         group,
#         int(session_id),
#         cfg.dataset.preprocessing.sliding_window.window_time,
#         cfg.dataset.preprocessing.sliding_window.overlap,
#     )

#     if wi is None or w is None:
#         continue

#     normalize = None

#     match cfg.dataset.preprocessing.normalization:
#         case NormType.STD_PER_SAMPLE:
#             normalize = standardize

#         case NormType.MIN_MAX_PER_SAMPLE:
#             normalize = min_max

#     w = normalize_per_sample(w, normalize)

#     window_indexes.append(wi)
#     windows.extend(w)
