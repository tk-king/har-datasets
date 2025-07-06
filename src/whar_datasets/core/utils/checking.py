import os
import pandas as pd
from dask.delayed import delayed
from dask.base import compute

from whar_datasets.core.config import WHARConfig
from whar_datasets.core.utils.hashing import load_cfg_hash


def check_download(dataset_dir: str) -> bool:
    print("Checking download...")

    if not os.path.exists(dataset_dir):
        print(f"Dataset directory not found at '{dataset_dir}'.")
        return False

    print("Download is up-to-date.")
    return True


def check_windowing(cache_dir: str, windows_dir: str, cfg_hash: str) -> bool:
    print("Checking windowing...")

    if not os.path.exists(windows_dir):
        print(f"Windows directory not found at '{windows_dir}'.")
        return False

    if len(os.listdir(windows_dir)) == 0:
        print(f"Windows directory '{windows_dir}' is empty.")
        return False

    window_metadata_path = os.path.join(cache_dir, "window_metadata.parquet")

    if not os.path.exists(window_metadata_path):
        print(f"Window index file not found at '{window_metadata_path}'.")
        return False

    current_hash = load_cfg_hash(cache_dir)

    if cfg_hash != current_hash:
        print("Config hash mismatch.")
        return False

    print("Windowing is up-to-date.")
    return True


def check_sessions(cache_dir: str, sessions_dir: str) -> bool:
    print("Checking sessions...")

    if not os.path.exists(sessions_dir):
        print(f"Sessions directory not found at '{sessions_dir}'.")
        return False

    if len(os.listdir(sessions_dir)) == 0:
        print(f"Sessions directory '{sessions_dir}' is empty.")
        return False

    session_metadata_path = os.path.join(cache_dir, "session_metadata.parquet")
    if not os.path.exists(session_metadata_path):
        print(f"Session index file not found at '{session_metadata_path}'.")
        return False

    activity_metadata_path = os.path.join(cache_dir, "activity_metadata.parquet")
    if not os.path.exists(activity_metadata_path):
        print(f"Activity index file not found at '{activity_metadata_path}'.")
        return False

    print("Sessions are up-to-date.")
    return True


def check_common_format(cfg: WHARConfig, cache_dir: str, sessions_dir: str) -> bool:
    print("Checking common format...")

    # define paths
    session_metadata_path = os.path.join(cache_dir, "session_metadata.parquet")
    activity_metadata_path = os.path.join(cache_dir, "activity_metadata.parquet")

    # Check paths
    if not os.path.exists(sessions_dir):
        print(f"sessions directory does not exist at {sessions_dir}")
        return False

    if not os.path.exists(session_metadata_path):
        print(f"session_metadata.parquet not found at {session_metadata_path}")
        return False

    if not os.path.exists(activity_metadata_path):
        print(f"activity_metadata.parquet not found at {activity_metadata_path}")
        return False

    # load session and activity index
    sessions_index = pd.read_parquet(session_metadata_path)
    activity_metadata = pd.read_parquet(activity_metadata_path)

    # Check session_metadata
    if not pd.api.types.is_integer_dtype(sessions_index["session_id"]):
        print("'session_id' column is not integer type.")
        return False
    if not pd.api.types.is_integer_dtype(sessions_index["subject_id"]):
        print("'subject_id' column is not integer type.")
        return False
    if not pd.api.types.is_integer_dtype(sessions_index["activity_id"]):
        print("'activity_id' column is not integer type.")
        return False
    if sessions_index["session_id"].nunique() != len(os.listdir(sessions_dir)):
        print(
            f"Number of session_ids {sessions_index['session_id'].nunique()} does not match number of session files {len(os.listdir(sessions_dir))}."
        )
        return False
    if sessions_index["session_id"].min() != 0:
        print("Minimum session_id is not 0.")
        return False
    if sessions_index["subject_id"].min() != 0:
        print("Minimum subject_id is not 0.")
        return False
    if sessions_index["activity_id"].min() != 0:
        print("Minimum activity_id is not 0.")
        return False
    if sessions_index["subject_id"].nunique() != cfg.dataset.info.num_of_subjects:
        # print(sessions_index["subject_id"].unique())
        print(
            f"In session_metadata, num of subject_ids {sessions_index['subject_id'].nunique()} does not match num of subjects {cfg.dataset.info.num_of_subjects}."
        )
        return False
    if sessions_index["activity_id"].nunique() != cfg.dataset.info.num_of_activities:
        print(
            f"In session_metadata, number of activity_ids {sessions_index['activity_id'].nunique()} does not match number of activities {cfg.dataset.info.num_of_activities} ."
        )
        return False

    # Check activity_metadata
    if not pd.api.types.is_integer_dtype(activity_metadata["activity_id"]):
        print("'activity_id' column is not integer type.")
        return False
    if not pd.api.types.is_string_dtype(activity_metadata["activity_name"]):
        print("'activity_name' column is not string type.")
        return False
    if activity_metadata["activity_id"].min() != 0:
        print("Minimum activity_id is not 0.")
        return False
    if activity_metadata["activity_id"].nunique() != cfg.dataset.info.num_of_activities:
        print("Number of activity_ids does not match number of activities.")
        return False

    # Check session files
    checks = [
        check_session(cfg, sessions_dir, session_id)
        for session_id in sessions_index["session_id"]
    ]

    if not all(compute(*checks)):
        return False

    print("Common format is up-to-date.")
    return True


@delayed
def check_session(cfg: WHARConfig, sessions_dir, session_id: int) -> bool:
    session_path = os.path.join(sessions_dir, f"session_{session_id}.parquet")

    if not os.path.exists(session_path):
        print(f"Session file not found: {session_path}")
        return False

    session_df = pd.read_parquet(session_path)

    if not pd.api.types.is_datetime64_dtype(session_df["timestamp"]):
        print(f"'timestamp' column in {session_path} is not datetime64 type.")
        return False

    if (
        len(session_df.columns.difference(["timestamp"]))
        != cfg.dataset.info.num_of_channels
    ):
        print(session_df.columns.difference(["timestamp"]))
        print(
            f"Number of columns {len(session_df.columns.difference(['timestamp']))} in {session_path} does not match number of channel {cfg.dataset.info.num_of_channels}."
        )
        return False

    for col in session_df.columns.difference(["timestamp"]):
        if not pd.api.types.is_float_dtype(session_df[col]):
            print(f"Column '{col}' in {session_path} is not float type.")
            return False

    return True
