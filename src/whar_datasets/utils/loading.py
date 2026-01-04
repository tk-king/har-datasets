from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm


def load_window_df(cache_dir: Path) -> pd.DataFrame:
    window_df_path = cache_dir / "window_df.csv"
    return pd.read_csv(window_df_path)


def load_session_df(cache_dir: Path) -> pd.DataFrame:
    session_df_path = cache_dir / "session_df.csv"
    return pd.read_csv(session_df_path)


def load_activity_df(cache_dir: Path) -> pd.DataFrame:
    activity_df_path = cache_dir / "activity_df.csv"
    return pd.read_csv(activity_df_path)


def load_samples(
    window_df: pd.DataFrame, samples_dir: Path
) -> Dict[str, List[np.ndarray]]:
    # initialize map from window_id to sample
    samples: Dict[str, List[np.ndarray]] = {}

    # load samples
    loop = tqdm(window_df["window_id"])
    loop.set_description("Loading samples")

    for window_id in loop:
        assert isinstance(window_id, str)
        sample = load_sample(samples_dir, window_id)
        samples[window_id] = sample

    return samples


def load_windows(window_df: pd.DataFrame, windows_dir: Path) -> Dict[str, pd.DataFrame]:
    # initialize map from window_id to window
    windows: Dict[str, pd.DataFrame] = {}

    # load windows
    loop = tqdm(window_df["window_id"])
    loop.set_description("Loading windows")

    for window_id in loop:
        assert isinstance(window_id, str)
        window = load_window(windows_dir, window_id)
        windows[window_id] = window

    return windows


def load_sessions(
    sessions_dir: Path, session_df: pd.DataFrame
) -> Dict[int, pd.DataFrame]:
    sessions = {}

    for session_id in session_df["session_id"]:
        assert isinstance(session_id, int)
        session = load_session(sessions_dir, session_id)
        sessions[session_id] = session

    return sessions


def load_sample(samples_dir: Path, window_id: str) -> List[np.ndarray]:
    sample_path = Path(samples_dir) / f"sample_{window_id}.npy"
    return np.load(sample_path, allow_pickle=True).tolist()


def load_window(windows_dir: Path, window_id: str) -> pd.DataFrame:
    window_path = windows_dir / f"window_{window_id}.parquet"
    return pd.read_parquet(window_path)


def load_session(sessions_dir: Path, session_id: int) -> pd.DataFrame:
    session_path = sessions_dir / f"session_{session_id}.csv"
    return pd.read_csv(session_path, parse_dates=["timestamp"])
