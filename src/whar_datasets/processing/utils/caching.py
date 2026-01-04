import shutil
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm


def cache_samples(
    samples_dir: Path,
    window_df: pd.DataFrame,
    samples: Dict[str, List[np.ndarray]],
) -> None:
    # delete windowing directory if it exists
    if samples_dir.exists():
        shutil.rmtree(samples_dir)

    # create samples directory if it does not exist
    samples_dir.mkdir(parents=True, exist_ok=True)

    # loop over index of window index
    loop = tqdm(window_df["window_id"])
    loop.set_description("Caching samples")

    # save samples
    for window_id in loop:
        assert isinstance(window_id, str)
        sample_path = samples_dir / f"sample_{window_id}.npy"
        np.save(sample_path, np.array(samples[window_id], dtype=object))


def cache_windows(
    windows_dir: Path, window_df: pd.DataFrame, windows: Dict[str, pd.DataFrame]
) -> None:
    # delete windowing directory if it exists
    if windows_dir.exists():
        shutil.rmtree(windows_dir)

    # create windowing directory if it does not exist
    windows_dir.mkdir(parents=True, exist_ok=True)

    # loop over index of window index
    loop = tqdm(window_df["window_id"])
    loop.set_description("Caching windows")

    # save windows
    for window_id in loop:
        assert isinstance(window_id, str)
        window_path = windows_dir / f"window_{window_id}.parquet"
        windows[window_id].to_parquet(window_path, index=False)


def cache_window_df(metadata_dir: Path, window_df: pd.DataFrame) -> None:
    # create directories if do not exist
    metadata_dir.mkdir(parents=True, exist_ok=True)

    # define window index path
    window_df_path = metadata_dir / "window_df.csv"

    # save window index
    window_df.to_csv(window_df_path, index=True)


def cache_common_format(
    metadata_dir: Path,
    sessions_dir: Path,
    activity_df: pd.DataFrame,
    session_df: pd.DataFrame,
    sessions: Dict[int, pd.DataFrame],
) -> None:
    # delete sessions directory if it exists
    if sessions_dir.exists():
        shutil.rmtree(sessions_dir)

    # create directories if do not exist
    metadata_dir.mkdir(parents=True, exist_ok=True)
    sessions_dir.mkdir(parents=True, exist_ok=True)

    # define paths
    activity_df_path = metadata_dir / "activity_df.csv"
    session_df_path = metadata_dir / "session_df.csv"

    # save activity and session index
    activity_df.to_csv(activity_df_path, index=True)
    session_df.to_csv(session_df_path, index=True)

    # loop over sessions
    loop = tqdm(session_df["session_id"])
    loop.set_description("Caching sessions")

    # save sessions
    for session_id in loop:
        assert isinstance(session_id, int)
        session_path = sessions_dir / f"session_{session_id}.csv"
        sessions[session_id].to_csv(session_path, index=False)
