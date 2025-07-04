import os
from typing import List, Tuple
import pandas as pd
from tqdm import tqdm

from whar_datasets.core.config import WHARConfig


def load_windowing(
    cache_dir: str, windows_dir: str, cfg: WHARConfig
) -> Tuple[pd.DataFrame, List[pd.DataFrame] | None]:
    # load window index
    window_index = load_window_index(cache_dir)

    if not cfg.dataset.training.in_memory:
        return window_index, None

    windows: List[pd.DataFrame] = []

    # load windows
    loop = tqdm(range(len(window_index)))
    loop.set_description("Loading windows")

    for i in loop:
        # get window_id
        window_id = window_index.loc[i]["window_id"]
        assert isinstance(window_id, str)

        # load window and append
        window = load_window(windows_dir, window_id)
        windows.append(window)

    return window_index, windows


def load_window(windows_dir: str, window_id: str) -> pd.DataFrame:
    return pd.read_parquet(os.path.join(windows_dir, f"window_{window_id}.parquet"))


def load_window_index(cache_dir: str) -> pd.DataFrame:
    return pd.read_parquet(os.path.join(cache_dir, "window_index.parquet"))


def load_session_index(cache_dir: str) -> pd.DataFrame:
    return pd.read_parquet(os.path.join(cache_dir, "session_index.parquet"))


def load_activity_index(cache_dir: str) -> pd.DataFrame:
    return pd.read_parquet(os.path.join(cache_dir, "activity_index.parquet"))
