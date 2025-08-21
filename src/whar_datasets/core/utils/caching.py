import os
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
    samples_dir: str,
    window_metadata: pd.DataFrame,
    samples: Dict[str, List[np.ndarray]],
) -> None:
    # delete samples directory if it exists
    if os.path.exists(samples_dir):
        shutil.rmtree(samples_dir)

    # create samples directory if it does not exist
    os.makedirs(samples_dir, exist_ok=True)

    # combine all samples into a single DataFrame
    combined_samples = []
    
    loop = tqdm(window_metadata["window_id"])
    loop.set_description("Combining samples")
    
    for window_id in loop:
        assert isinstance(window_id, str)
        # convert list of numpy arrays to a single serialized format for parquet
        sample_arrays = samples[window_id]
        
        # create a row for each array in the sample
        for array_idx, array in enumerate(sample_arrays):
            combined_samples.append({
                'window_id': window_id,
                'array_index': array_idx,
                'array_data': array.tobytes(),  # serialize numpy array to bytes
                'array_shape': array.shape,
                'array_dtype': str(array.dtype)
            })
    
    # create DataFrame and save to parquet
    if combined_samples:
        all_samples_df = pd.DataFrame(combined_samples)
        
        # save all samples to a single parquet file
        samples_path = os.path.join(samples_dir, "all_samples.parquet")
        all_samples_df.to_parquet(samples_path, index=False)


def cache_windows(
    windows_dir: str, window_metadata: pd.DataFrame, windows: Dict[str, pd.DataFrame]
) -> None:
    # delete windowing directory if it exists
    if os.path.exists(windows_dir):
        shutil.rmtree(windows_dir)

    # create windowing directory if it does not exist
    os.makedirs(windows_dir, exist_ok=True)

    # combine all windows into a single DataFrame
    combined_windows = []
    
    loop = tqdm(window_metadata["window_id"])
    loop.set_description("Combining windows")
    
    for window_id in loop:
        assert isinstance(window_id, str)
        window_df = windows[window_id].copy()
        # add window_id column to identify which window each row belongs to
        window_df['window_id'] = window_id
        combined_windows.append(window_df)
    
    # concatenate all windows into a single DataFrame
    if combined_windows:
        all_windows_df = pd.concat(combined_windows, ignore_index=True)
        
        # save all windows to a single parquet file
        windows_path = os.path.join(windows_dir, "all_windows.parquet")
        all_windows_df.to_parquet(windows_path, index=False)


def cache_window_metadata(cache_dir: str, window_metadata: pd.DataFrame) -> None:
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
