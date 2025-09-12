import os
from pathlib import Path
import shutil
from typing import Dict, List
import numpy as np
import pandas as pd
from tqdm import tqdm


def cache_cfg_hash(hashes_dir: str, cfg_hash: str) -> None:
    # create windowing directory if it does not exist
    os.makedirs(hashes_dir, exist_ok=True)

    # save config hash
    with open(os.path.join(hashes_dir, "cfg_hash.txt"), "w") as f:
        f.write(cfg_hash)


def cache_norm_params_hash(hashes_dir: str, cfg_hash: str) -> None:
    # create windowing directory if it does not exist
    os.makedirs(hashes_dir, exist_ok=True)

    # save config hash
    with open(os.path.join(hashes_dir, "norm_params_hash.txt"), "w") as f:
        f.write(cfg_hash)


def cache_samples(
    samples_dir: Path,
    window_metadata: pd.DataFrame,
    samples: Dict[str, List[np.ndarray]],
) -> None:
    # delete windowing directory if it exists
    if os.path.exists(samples_dir):
        shutil.rmtree(samples_dir)

    # create samples directory if it does not exist
    os.makedirs(samples_dir, exist_ok=True)

    # loop over index of window index
    loop = tqdm(window_metadata["window_id"])
    loop.set_description("Caching samples")

    # save samples
    for window_id in loop:
        assert isinstance(window_id, str)
        sample_path = os.path.join(samples_dir, f"sample_{window_id}.npy")
        np.save(sample_path, np.array(samples[window_id], dtype=object))


def cache_windows(
    windows_dir: Path, window_metadata: pd.DataFrame, windows: Dict[str, pd.DataFrame]
) -> None:
    # delete windowing directory if it exists
    if os.path.exists(windows_dir):
        shutil.rmtree(windows_dir)

    # create windowing directory if it does not exist
    os.makedirs(windows_dir, exist_ok=True)

    # loop over index of window index
    loop = tqdm(window_metadata["window_id"])
    loop.set_description("Caching windows")

    # save windows
    for window_id in loop:
        assert isinstance(window_id, str)
        window_path = os.path.join(windows_dir, f"window_{window_id}.parquet")
        windows[window_id].to_parquet(window_path, index=False)


def cache_window_metadata(cache_dir: Path, window_metadata: pd.DataFrame) -> None:
    # create directories if do not exist
    os.makedirs(cache_dir, exist_ok=True)

    # define window index path
    window_metadata_path = os.path.join(cache_dir, "window_metadata.parquet")

    # save window index
    window_metadata.to_parquet(window_metadata_path, index=True)


def cache_common_format(
    cache_dir: str,
    sessions_dir: str,
    activity_metadata: pd.DataFrame,
    session_metadata: pd.DataFrame,
    sessions: Dict[int, pd.DataFrame],
) -> None:
    # delete sessions directory if it exists
    if os.path.exists(sessions_dir):
        shutil.rmtree(sessions_dir)

    # create directories if do not exist
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(sessions_dir, exist_ok=True)

    # define paths
    activity_metadata_path = os.path.join(cache_dir, "activity_metadata.parquet")
    session_metadata_path = os.path.join(cache_dir, "session_metadata.parquet")

    # save activity and session index
    activity_metadata.to_parquet(activity_metadata_path, index=True)
    session_metadata.to_parquet(session_metadata_path, index=True)

    # loop over sessions
    loop = tqdm(session_metadata["session_id"])
    loop.set_description("Caching sessions")

    # save sessions
    for session_id in loop:
        assert isinstance(session_id, int)
        session_path = os.path.join(sessions_dir, f"session_{session_id}.parquet")
        sessions[session_id].to_parquet(session_path, index=False)
