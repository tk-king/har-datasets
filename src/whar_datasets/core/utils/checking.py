import os
import pandas as pd
from dask.delayed import delayed
from dask.base import compute

from whar_datasets.core.config import WHARConfig
from whar_datasets.core.utils.hashing import create_cfg_hash, load_cfg_hash


def check_download(datasets_dir: str, cfg: WHARConfig) -> bool:
    print("Checking download...")

    dataset_dir = os.path.join(datasets_dir, cfg.dataset.info.id)

    return os.path.exists(dataset_dir)


def check_windowing(cache_dir: str, windows_dir: str, cfg: WHARConfig) -> bool:
    print("Checking windowing...")

    if not os.path.exists(windows_dir):
        return False

    if not len(os.listdir(windows_dir)) > 0:
        return False

    if not os.path.exists(os.path.join(cache_dir, "window_index.parquet")):
        return False

    if not create_cfg_hash(cfg) == load_cfg_hash(cache_dir):
        return False

    return True


def check_sessions(cache_dir: str, sessions_dir: str) -> bool:
    print("Checking sessions...")

    if not os.path.exists(sessions_dir):
        return False

    if not len(os.listdir(sessions_dir)) > 0:
        return False

    if not os.path.exists(os.path.join(cache_dir, "session_index.parquet")):
        return False

    return True


def check_common_format(cache_dir: str, sessions_dir: str) -> bool:
    print("Checking common format...")

    session_index_path = os.path.join(cache_dir, "session_index.parquet")
    activity_index_path = os.path.join(cache_dir, "activity_index.parquet")

    if not (
        os.path.exists(sessions_dir)
        and os.path.exists(session_index_path)
        and os.path.exists(activity_index_path)
    ):
        return False

    sessions_index = pd.read_parquet(session_index_path)

    # check types in session index
    if not (
        sessions_index["session_id"].dtype == "int32"
        and sessions_index["subject_id"].dtype == "int32"
        and sessions_index["activity_id"].dtype == "int32"
        and sessions_index["session_id"].nunique() == len(os.listdir(sessions_dir))
        and sessions_index["session_id"].min() == 0
    ):
        return False

    activity_index = pd.read_parquet(activity_index_path)

    # check types in activity index
    if not (
        activity_index["activity_id"].dtype == "int32"
        and activity_index["activity_name"].dtype == "string"
        and activity_index["activity_id"].min() == 0
    ):
        return False

    @delayed
    def session_exists(session_id):
        session_path = os.path.join(sessions_dir, f"session_{session_id}.parquet")

        if not os.path.exists(session_path):
            return False

        session_df = pd.read_parquet(session_path)

        # check that timestamp col is datetime
        if session_df["timestamp"].dtype != "datetime64[ms]":
            return False

        # check that all other cols are float
        for col in session_df.columns.difference(["timestamp"]):
            if session_df[col].dtype != "float32":
                return False

        return True

    checks = [session_exists(session_id) for session_id in sessions_index["session_id"]]
    results = compute(*checks)

    return all(results)
