import os
import pandas as pd
from dask.delayed import delayed
from dask.base import compute

from whar_datasets.core.config import WHARConfig
from whar_datasets.core.utils.hashing import load_cfg_hash


def check_download(dataset_dir: str) -> bool:
    print("Checking download...")

    if not os.path.exists(dataset_dir):
        print(f"Error: Dataset directory not found at '{dataset_dir}'.")
        return False

    print("Download is up-to-date.")
    return True


def check_windowing(cache_dir: str, windows_dir: str, cfg_hash: str) -> bool:
    print("Checking windowing...")

    if not os.path.exists(windows_dir):
        print(f"Error: Windows directory not found at '{windows_dir}'.")
        return False

    if len(os.listdir(windows_dir)) == 0:
        print(f"Error: Windows directory '{windows_dir}' is empty.")
        return False

    window_index_path = os.path.join(cache_dir, "window_index.parquet")

    if not os.path.exists(window_index_path):
        print(f"Error: Window index file not found at '{window_index_path}'.")
        return False

    current_hash = load_cfg_hash(cache_dir)

    if cfg_hash != current_hash:
        print(
            f"Error: Config hash mismatch. Expected: {cfg_hash}, Found: {current_hash}"
        )
        return False

    print("Windowing is up-to-date.")
    return True


def check_sessions(cache_dir: str, sessions_dir: str) -> bool:
    print("Checking sessions...")

    if not os.path.exists(sessions_dir):
        print(f"Error: Sessions directory not found at '{sessions_dir}'.")
        return False

    if len(os.listdir(sessions_dir)) == 0:
        print(f"Error: Sessions directory '{sessions_dir}' is empty.")
        return False

    session_index_path = os.path.join(cache_dir, "session_index.parquet")
    if not os.path.exists(session_index_path):
        print(f"Error: Session index file not found at '{session_index_path}'.")
        return False

    activity_index_path = os.path.join(cache_dir, "activity_index.parquet")
    if not os.path.exists(activity_index_path):
        print(f"Error: Activity index file not found at '{activity_index_path}'.")
        return False

    print("Sessions are up-to-date.")
    return True


def check_common_format(cfg: WHARConfig, cache_dir: str, sessions_dir: str) -> bool:
    print("Checking common format...")

    session_index_path = os.path.join(cache_dir, "session_index.parquet")
    activity_index_path = os.path.join(cache_dir, "activity_index.parquet")

    # Check paths
    if not os.path.exists(sessions_dir):
        print(f"Error: sessions directory does not exist at {sessions_dir}")
        return False

    if not os.path.exists(session_index_path):
        print(f"Error: session_index.parquet not found at {session_index_path}")
        return False

    if not os.path.exists(activity_index_path):
        print(f"Error: activity_index.parquet not found at {activity_index_path}")
        return False

    sessions_index = pd.read_parquet(session_index_path)

    # Check session_index columns
    if not pd.api.types.is_integer_dtype(sessions_index["session_id"]):
        print("Error: 'session_id' column in session_index is not integer type.")
        return False
    if not pd.api.types.is_integer_dtype(sessions_index["subject_id"]):
        print("Error: 'subject_id' column in session_index is not integer type.")
        return False
    if not pd.api.types.is_integer_dtype(sessions_index["activity_id"]):
        print("Error: 'activity_id' column in session_index is not integer type.")
        return False
    if sessions_index["session_id"].nunique() != len(os.listdir(sessions_dir)):
        print("Error: Number of session_ids does not match number of session files.")
        return False
    if sessions_index["session_id"].min() != 0:
        print("Error: Minimum session_id is not 0 in session_index.")
        return False
    if sessions_index["subject_id"].min() != 0:
        print("Error: Minimum subject_id is not 0 in session_index.")
        return False
    if sessions_index["activity_id"].min() != 0:
        print("Error: Minimum activity_id is not 0 in session_index.")
        return False
    if sessions_index["subject_id"].nunique() != cfg.dataset.info.num_of_subjects:
        # print(sessions_index["subject_id"].unique())
        print(
            "Error: Number of subject_ids does not match number of subjects in session_index."
        )
        return False
    if sessions_index["activity_id"].nunique() != cfg.dataset.info.num_of_activities:
        print(
            "Error: Number of activity_ids does not match number of activities in session_index."
        )
        return False

    activity_index = pd.read_parquet(activity_index_path)

    # Check activity_index columns
    if not pd.api.types.is_integer_dtype(activity_index["activity_id"]):
        print("Error: 'activity_id' column in activity_index is not integer type.")
        return False
    if not pd.api.types.is_string_dtype(activity_index["activity_name"]):
        print("Error: 'activity_name' column in activity_index is not string type.")
        return False
    if activity_index["activity_id"].min() != 0:
        print("Error: Minimum activity_id is not 0 in activity_index.")
        return False
    if activity_index["activity_id"].nunique() != cfg.dataset.info.num_of_activities:
        print(
            "Error: Number of activity_ids does not match number of activities in activity_index."
        )
        return False

    # Check session files
    @delayed
    def session_exists(session_id):
        session_path = os.path.join(sessions_dir, f"session_{session_id}.parquet")

        if not os.path.exists(session_path):
            print(f"Error: Session file not found: {session_path}")
            return False

        session_df = pd.read_parquet(session_path)

        if not pd.api.types.is_datetime64_dtype(session_df["timestamp"]):
            print(
                f"Error: 'timestamp' column in {session_path} is not datetime64 type."
            )
            return False

        if (
            len(session_df.columns.difference(["timestamp"]))
            != cfg.dataset.info.num_of_channels
        ):
            print(
                f"Error: Number of columns in {session_path} does not match number of channels."
            )
            return False

        for col in session_df.columns.difference(["timestamp"]):
            if not pd.api.types.is_float_dtype(session_df[col]):
                print(f"Error: Column '{col}' in {session_path} is not float type.")
                return False

        return True

    checks = [session_exists(session_id) for session_id in sessions_index["session_id"]]
    results = compute(*checks)

    if not all(results):
        return False

    print("Common format is up-to-date.")
    return True
