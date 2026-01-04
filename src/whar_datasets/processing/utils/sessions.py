from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from dask.base import compute
from dask.delayed import delayed
from dask.diagnostics.progress import ProgressBar
from tqdm import tqdm

from whar_datasets.config.config import WHARConfig
from whar_datasets.processing.utils.resampling import resample
from whar_datasets.processing.utils.selecting import select_channels
from whar_datasets.processing.utils.windowing import generate_windowing
from whar_datasets.utils.loading import load_session


def process_sessions_seq(
    cfg: WHARConfig, sessions_dir: Path, session_df: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    # loop over sessions
    loop = tqdm([int(x) for x in session_df["session_id"].unique()])
    loop.set_description("Processing sessions")

    pairs = [process_session(cfg, sessions_dir, session_id) for session_id in loop]
    window_dfs, window_dicts = zip(*pairs)

    # compute global window metadata and windows
    window_df = pd.concat([w for w in window_dfs if w is not None])
    window_df.reset_index(drop=True, inplace=True)
    windows = {k: v for d in window_dicts if d is not None for k, v in d.items()}

    # assert uniqueness of window ids
    assert window_df["window_id"].nunique() == len(window_df)

    return window_df, windows


def process_sessions_para(
    cfg: WHARConfig, sessions_dir: Path, session_df: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    @delayed
    def process_session_delayed(
        session_id: int,
    ) -> Tuple[pd.DataFrame | None, Dict[str, pd.DataFrame] | None]:
        return process_session(cfg, sessions_dir, session_id)

    # define processing tasks
    tasks = [
        process_session_delayed(session_id)
        for session_id in [int(x) for x in session_df["session_id"]]
    ]

    # execute tasks in parallel
    pbar = ProgressBar()
    pbar.register()
    pairs = list(compute(*tasks, scheduler="processes"))
    pbar.unregister()

    window_dfs, window_dicts = zip(*pairs)

    # compute global window metadata and windows
    window_df = pd.concat([w for w in window_dfs if w is not None])
    window_df.reset_index(drop=True, inplace=True)
    windows = {k: v for d in window_dicts if d is not None for k, v in d.items()}

    # assert uniqueness of window ids
    assert window_df["window_id"].nunique() == len(window_df)

    return window_df, windows


def process_session(
    cfg: WHARConfig, sessions_dir: Path, session_id: int
) -> Tuple[pd.DataFrame | None, Dict[str, pd.DataFrame] | None]:
    # laod and process session
    session = load_session(sessions_dir, session_id)
    session = select_channels(session, cfg.sensor_channels)
    session = resample(session, cfg.sampling_freq)

    # generate windowing
    window_df, windows = generate_windowing(
        session_id,
        session,
        cfg.window_time,
        cfg.window_overlap,
        cfg.sampling_freq,
    )

    if window_df is None or windows is None:
        return None, None

    return window_df, windows
