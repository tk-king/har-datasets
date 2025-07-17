import os
from dask.delayed import delayed
from dask.base import compute
from dask.diagnostics.progress import ProgressBar
import pandas as pd
from tqdm import tqdm

from whar_datasets.core.config import WHARConfig


def validate_common_format(cfg: WHARConfig, cache_dir: str, sessions_dir: str) -> bool:
    print("Validating common format...")

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
    session_metadata = pd.read_parquet(session_metadata_path)
    activity_metadata = pd.read_parquet(activity_metadata_path)

    # Check session_metadata
    if not pd.api.types.is_integer_dtype(session_metadata["session_id"]):
        print("'session_id' column is not integer type.")
        return False
    if not pd.api.types.is_integer_dtype(session_metadata["subject_id"]):
        print("'subject_id' column is not integer type.")
        return False
    if not pd.api.types.is_integer_dtype(session_metadata["activity_id"]):
        print("'activity_id' column is not integer type.")
        return False
    if session_metadata["session_id"].nunique() != len(os.listdir(sessions_dir)):
        print(
            f"Number of session_ids {session_metadata['session_id'].nunique()} does not match number of session files {len(os.listdir(sessions_dir))}."
        )
        return False
    if session_metadata["session_id"].min() != 0:
        print("Minimum session_id is not 0.")
        return False
    if session_metadata["subject_id"].min() != 0:
        print("Minimum subject_id is not 0.")
        return False
    if session_metadata["activity_id"].min() != 0:
        print("Minimum activity_id is not 0.")
        return False
    if session_metadata["subject_id"].nunique() != cfg.num_of_subjects:
        print(
            f"In session_metadata, num of subject_ids {session_metadata['subject_id'].nunique()} does not match num of subjects {cfg.num_of_subjects}."
        )
        return False
    if session_metadata["activity_id"].nunique() != cfg.num_of_activities:
        print(
            f"In session_metadata, number of activity_ids {session_metadata['activity_id'].nunique()} does not match number of activities {cfg.num_of_activities} ."
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
    if activity_metadata["activity_id"].nunique() != cfg.num_of_activities:
        print("Number of activity_ids does not match number of activities.")
        return False

    validated = (
        validate_sessions_parallely(cfg, sessions_dir, session_metadata)
        if cfg.in_parallel
        else validate_sessions_sequentially(cfg, sessions_dir, session_metadata)
    )

    if not validated:
        return False

    print("Common format validated.")
    return True


def validate_sessions_sequentially(
    cfg: WHARConfig, sessions_dir: str, session_metadata: pd.DataFrame
) -> bool:
    loop = tqdm(session_metadata["session_id"])
    loop.set_description("Validating sessions")

    for session_id in loop:
        if not validate_session(cfg, sessions_dir, session_id):
            return False

    return True


def validate_sessions_parallely(
    cfg: WHARConfig, sessions_dir: str, session_metadata: pd.DataFrame
) -> bool:
    @delayed
    def validate_session_delayed(session_id: int) -> bool:
        return validate_session(cfg, sessions_dir, session_id)

    # define processing tasks
    tasks = [
        validate_session_delayed(session_id)
        for session_id in session_metadata["session_id"]
    ]

    # execute tasks in parallel
    pbar = ProgressBar()
    pbar.register()
    results = list(compute(*tasks, scheduler="processes"))
    pbar.unregister()

    return all(results)


def validate_session(cfg: WHARConfig, sessions_dir, session_id: int) -> bool:
    session_path = os.path.join(sessions_dir, f"session_{session_id}.parquet")

    if not os.path.exists(session_path):
        print(f"Session file not found: {session_path}")
        return False

    session_df = pd.read_parquet(session_path)

    if not pd.api.types.is_datetime64_dtype(session_df["timestamp"]):
        print(f"'timestamp' column in {session_path} is not datetime64 type.")
        return False

    if len(session_df.columns.difference(["timestamp"])) != cfg.num_of_channels:
        print(session_df.columns.difference(["timestamp"]))
        print(
            f"Number of columns {len(session_df.columns.difference(['timestamp']))} in {session_path} does not match number of channel {cfg.num_of_channels}."
        )
        return False

    for col in session_df.columns.difference(["timestamp"]):
        if not pd.api.types.is_float_dtype(session_df[col]):
            print(f"Column '{col}' in {session_path} is not float type.")
            return False

    if session_df.isna().any().any():
        print(f"Session file {session_path} contains NaN values.")
        return False

    return True
