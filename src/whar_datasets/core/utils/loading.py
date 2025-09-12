from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
from tqdm import tqdm


def load_samples(
    window_metadata: pd.DataFrame, samples_dir: Path
) -> Dict[str, List[np.ndarray]]:
    # initialize map from window_id to sample
    samples: Dict[str, List[np.ndarray]] = {}

    # load samples
    loop = tqdm(window_metadata["window_id"])
    loop.set_description("Loading samples")

    for window_id in loop:
        assert isinstance(window_id, str)
        sample = load_sample(samples_dir, window_id)
        samples[window_id] = sample

    return samples


def load_windows(
    window_metadata: pd.DataFrame, windows_dir: Path
) -> Dict[str, pd.DataFrame]:
    # initialize map from window_id to window
    windows: Dict[str, pd.DataFrame] = {}

    # load windows
    loop = tqdm(window_metadata["window_id"])
    loop.set_description("Loading windows")

    for window_id in loop:
        assert isinstance(window_id, str)
        window = load_window(windows_dir, window_id)
        windows[window_id] = window

    return windows


def load_sessions(
    sessions_dir: Path, session_metadata: pd.DataFrame
) -> Dict[int, pd.DataFrame]:
    sessions = {}

    for session_id in session_metadata["session_id"]:
        assert isinstance(session_id, int)
        session = load_session(sessions_dir, session_id)
        sessions[session_id] = session

    return sessions


def load_sample(samples_dir: Path, window_id: str) -> List[np.ndarray]:
    sample_path = Path(samples_dir) / Path(f"sample_{window_id}.npy")
    return np.load(sample_path, allow_pickle=True).tolist()


def load_window(windows_dir: Path, window_id: str) -> pd.DataFrame:
    window_path = windows_dir / Path(f"window_{window_id}.parquet")
    return pd.read_parquet(window_path)


def load_session(sessions_dir: Path, session_id: int) -> pd.DataFrame:
    session_path = sessions_dir / Path(f"session_{session_id}.parquet")
    return pd.read_parquet(session_path)


def load_window_metadata(cache_dir: Path) -> pd.DataFrame:
    window_metadata_path = cache_dir / Path("window_metadata.parquet")
    return pd.read_parquet(window_metadata_path)


def load_session_metadata(cache_dir: Path) -> pd.DataFrame:
    session_metadata_path = cache_dir / Path("session_metadata.parquet")
    return pd.read_parquet(session_metadata_path)


def load_activity_metadata(cache_dir: Path) -> pd.DataFrame:
    activity_metadata_path = cache_dir / Path("activity_metadata.parquet")
    return pd.read_parquet(activity_metadata_path)
