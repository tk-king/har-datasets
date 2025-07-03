import os
from typing import List
import numpy as np
import pandas as pd
from tqdm import tqdm


def cache_cfg_hash(cache_dir: str, cfg_hash: str) -> None:
    # create windowing directory if it does not exist
    os.makedirs(cache_dir, exist_ok=True)

    # save config hash
    with open(os.path.join(cache_dir, "cfg_hash.txt"), "w") as f:
        f.write(cfg_hash)


def cache_windows(
    windows_dir: str,
    window_index: pd.DataFrame,
    windows: List[pd.DataFrame],
) -> None:
    # create directories if do not exist
    os.makedirs(windows_dir, exist_ok=True)

    # save windows
    loop = tqdm(enumerate(windows))
    loop.set_description("Saving windows")

    for i, window in loop:
        # get window_id
        window_id = window_index.loc[i]["window_id"]
        assert isinstance(window_id, np.integer)

        # save window
        window_path = os.path.join(windows_dir, f"window_{window_id}.parquet")
        window.to_parquet(window_path, index=False)


def cache_window_index(
    cache_dir: str,
    window_indexs: List[pd.DataFrame],
):
    # create directories if do not exist
    os.makedirs(cache_dir, exist_ok=True)

    # save window index
    window_index = pd.concat(window_indexs)
    window_index.to_parquet(
        os.path.join(cache_dir, "window_index.parquet"), index=False
    )


def cache_common_format(
    cache_dir: str,
    sessions_dir: str,
    activity_index: pd.DataFrame,
    session_index: pd.DataFrame,
    sessions: List[pd.DataFrame],
):
    # create directories if do not exist
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(sessions_dir, exist_ok=True)

    # save activity index
    activity_index.to_parquet(
        os.path.join(cache_dir, "activity_index.parquet"), index=False
    )

    # save session index
    session_index.to_parquet(
        os.path.join(cache_dir, "session_index.parquet"), index=False
    )

    # save sessions
    loop = tqdm(enumerate(sessions))
    loop.set_description("Saving sessions")

    for i, session in loop:
        # get session_id
        session_id = session_index.loc[i]["session_id"]
        assert isinstance(session_id, np.integer)

        # get and save session
        session_path = os.path.join(sessions_dir, f"session_{session_id}.parquet")
        session.to_parquet(session_path, index=False)
