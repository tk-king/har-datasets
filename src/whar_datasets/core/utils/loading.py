import os
from typing import Dict
import pandas as pd
from tqdm import tqdm


def load_windows(
    window_metadata: pd.DataFrame, windows_dir: str
) -> Dict[str, pd.DataFrame]:
    # initialize map from window_id to window
    windows: Dict[str, pd.DataFrame] = {}

    # load windows
    loop = tqdm(window_metadata["window_id"])
    loop.set_description("Loading windows")

    for window_id in loop:
        assert isinstance(window_id, str)

        # load window and add to map
        window = load_window(windows_dir, window_id)
        windows[window_id] = window

    return windows


def load_window(windows_dir: str, window_id: str) -> pd.DataFrame:
    window_path = os.path.join(windows_dir, f"window_{window_id}.parquet")
    return pd.read_parquet(window_path)


def load_window_metadata(cache_dir: str) -> pd.DataFrame:
    window_metadata_path = os.path.join(cache_dir, "window_metadata.parquet")
    return pd.read_parquet(window_metadata_path)


def load_session_metadata(cache_dir: str) -> pd.DataFrame:
    session_metadata_path = os.path.join(cache_dir, "session_metadata.parquet")
    return pd.read_parquet(session_metadata_path)


def load_activity_metadata(cache_dir: str) -> pd.DataFrame:
    activity_metadata_path = os.path.join(cache_dir, "activity_metadata.parquet")
    return pd.read_parquet(activity_metadata_path)
