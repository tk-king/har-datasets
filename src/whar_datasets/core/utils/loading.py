import os
from typing import Dict, List
import numpy as np
import pandas as pd
from tqdm import tqdm


def load_samples(
    window_metadata: pd.DataFrame, samples_dir: str
) -> Dict[str, List[np.ndarray]]:
    # initialize map from window_id to sample
    samples: Dict[str, List[np.ndarray]] = {}

    # load all samples from single parquet file
    samples_path = os.path.join(samples_dir, "all_samples.parquet")
    
    if os.path.exists(samples_path):
        print("Loading all samples from single parquet file...")
        # load the entire parquet file once
        all_samples_df = pd.read_parquet(samples_path)
        
        # group by window_id for efficient processing
        grouped = all_samples_df.groupby('window_id')
        
        # reconstruct samples for each window
        loop = tqdm(window_metadata["window_id"])
        loop.set_description("Reconstructing samples")
        
        for window_id in loop:
            assert isinstance(window_id, str)
            if window_id in grouped.groups:
                window_samples_df = grouped.get_group(window_id).sort_values('array_index')
                
                # reconstruct list of numpy arrays
                sample_arrays = []
                for _, row in window_samples_df.iterrows():
                    # deserialize numpy array from bytes
                    array_data = np.frombuffer(row['array_data'], dtype=row['array_dtype'])
                    array = array_data.reshape(row['array_shape'])
                    sample_arrays.append(array)
                
                samples[window_id] = sample_arrays
    else:
        raise ValueError("Samples file not found")

    return samples


def load_windows(
    window_metadata: pd.DataFrame, windows_dir: str
) -> Dict[str, pd.DataFrame]:
    # initialize map from window_id to window
    windows: Dict[str, pd.DataFrame] = {}

    # load all windows from single parquet file
    windows_path = os.path.join(windows_dir, "all_windows.parquet")
    
    if os.path.exists(windows_path):
        print("Loading all windows from single parquet file...")
        # load the combined parquet file once
        all_windows_df = pd.read_parquet(windows_path)
        
        # group by window_id for efficient splitting
        grouped = all_windows_df.groupby('window_id')
        
        # split back into individual windows
        loop = tqdm(window_metadata["window_id"])
        loop.set_description("Splitting windows")
        
        for window_id in loop:
            assert isinstance(window_id, str)
            if window_id in grouped.groups:
                # get the group for this window_id and remove the window_id column
                window_df = grouped.get_group(window_id).drop('window_id', axis=1).reset_index(drop=True)
                windows[window_id] = window_df
    else:
        raise ValueError("Windows file not found")

    return windows


def load_sample(samples_dir: str, window_id: str) -> List[np.ndarray]:
    # try to load from the new single parquet file format first
    samples_path = os.path.join(samples_dir, "all_samples.parquet")
    
    if os.path.exists(samples_path):
        # load only the specific sample using parquet filtering
        sample_df = pd.read_parquet(
            samples_path,
            filters=[('window_id', '==', window_id)]
        )
        
        # sort by array_index to maintain order
        sample_df = sample_df.sort_values('array_index')
        
        # reconstruct list of numpy arrays
        sample_arrays = []
        for _, row in sample_df.iterrows():
            # deserialize numpy array from bytes
            array_data = np.frombuffer(row['array_data'], dtype=row['array_dtype'])
            array = array_data.reshape(row['array_shape'])
            sample_arrays.append(array)
        
        return sample_arrays
    else:
        # fallback to old format for backward compatibility
        raise ValueError("Samples file not found")


def load_window(windows_dir: str, window_id: str) -> pd.DataFrame:
    # try to load from the new single parquet file format first
    windows_path = os.path.join(windows_dir, "all_windows.parquet")
    
    if os.path.exists(windows_path):
        # load only the specific window using parquet filtering (much more efficient)
        window_df = pd.read_parquet(
            windows_path,
            filters=[('window_id', '==', window_id)]
        )
        # remove the window_id column since it's not part of the original window data
        window_df = window_df.drop('window_id', axis=1).reset_index(drop=True)
        return window_df
    else:
        raise ValueError("Windows file not found")


def load_window_metadata(cache_dir: str) -> pd.DataFrame:
    window_metadata_path = os.path.join(cache_dir, "window_metadata.parquet")
    return pd.read_parquet(window_metadata_path)


def load_session_metadata(cache_dir: str) -> pd.DataFrame:
    session_metadata_path = os.path.join(cache_dir, "session_metadata.parquet")
    return pd.read_parquet(session_metadata_path)


def load_activity_metadata(cache_dir: str) -> pd.DataFrame:
    activity_metadata_path = os.path.join(cache_dir, "activity_metadata.parquet")
    return pd.read_parquet(activity_metadata_path)
