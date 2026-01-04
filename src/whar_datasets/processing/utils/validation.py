from pathlib import Path

import pandas as pd
from dask.base import compute
from dask.delayed import delayed
from dask.diagnostics.progress import ProgressBar
from tqdm import tqdm

from whar_datasets.config.config import WHARConfig
from whar_datasets.utils.loading import load_session
from whar_datasets.utils.logging import logger


def validate_common_format(
    cfg: WHARConfig,
    sessions_dir: Path,
    activity_df: pd.DataFrame,
    session_df: pd.DataFrame,
) -> bool:
    logger.info("Validating common format")

    # Check session_df
    if not pd.api.types.is_integer_dtype(session_df["session_id"]):
        logger.error("'session_id' column is not integer type.")
        return False
    if not pd.api.types.is_integer_dtype(session_df["subject_id"]):
        logger.error("'subject_id' column is not integer type.")
        return False
    if not pd.api.types.is_integer_dtype(session_df["activity_id"]):
        logger.error("'activity_id' column is not integer type.")
        return False
    if session_df["session_id"].min() != 0:
        logger.error("Minimum session_id is not 0.")
        return False
    if session_df["subject_id"].min() != 0:
        logger.error("Minimum subject_id is not 0.")
        return False
    if session_df["activity_id"].min() != 0:
        logger.error("Minimum activity_id is not 0.")
        return False
    if session_df["subject_id"].nunique() != cfg.num_of_subjects:
        logger.error(
            f"In session_df, num of subject_ids {session_df['subject_id'].nunique()} does not match num of subjects {cfg.num_of_subjects}."
        )
        return False
    if session_df["activity_id"].nunique() != cfg.num_of_activities:
        logger.error(
            f"In session_df, number of activity_ids {session_df['activity_id'].nunique()} does not match number of activities {cfg.num_of_activities} ."
        )
        return False

    # Check activity_df
    if not pd.api.types.is_integer_dtype(activity_df["activity_id"]):
        logger.error("'activity_id' column is not integer type.")
        return False
    if not pd.api.types.is_string_dtype(activity_df["activity_name"]):
        logger.error("'activity_name' column is not string type.")
        return False
    if activity_df["activity_id"].min() != 0:
        logger.error("Minimum activity_id is not 0.")
        return False
    if activity_df["activity_id"].nunique() != cfg.num_of_activities:
        logger.error("Number of activity_ids does not match number of activities.")
        return False

    validated = (
        validate_sessions_parallely(cfg, sessions_dir, session_df)
        if cfg.parallelize
        else validate_sessions_sequentially(cfg, sessions_dir, session_df)
    )

    if not validated:
        return False

    logger.info("Common format validated.")
    return True


def validate_sessions_sequentially(
    cfg: WHARConfig, sessions_dir: Path, session_df: pd.DataFrame
) -> bool:
    loop = tqdm(session_df["session_id"])
    loop.set_description("Validating sessions")

    for session_id in loop:
        if not validate_session(cfg, sessions_dir, session_id):
            return False

    return True


def validate_sessions_parallely(
    cfg: WHARConfig, sessions_dir: Path, session_df: pd.DataFrame
) -> bool:
    @delayed
    def validate_session_delayed(session_id: int) -> bool:
        return validate_session(cfg, sessions_dir, session_id)

    # define processing tasks
    tasks = [
        validate_session_delayed(session_id) for session_id in session_df["session_id"]
    ]

    # execute tasks in parallel
    pbar = ProgressBar()
    pbar.register()
    results = list(compute(*tasks, scheduler="processes"))
    pbar.unregister()

    return all(results)


def validate_session(cfg: WHARConfig, sessions_dir: Path, session_id: int) -> bool:
    session = load_session(sessions_dir, session_id)

    if not pd.api.types.is_datetime64_dtype(session["timestamp"]):
        logger.error(session["timestamp"].dtype)
        logger.error(f"'timestamp' column in {session_id} is not datetime64 type.")
        return False

    if len(session.columns.difference(["timestamp"])) != cfg.num_of_channels:
        logger.error(session.columns.difference(["timestamp"]))
        logger.error(
            f"Number of columns {len(session.columns.difference(['timestamp']))} in {session_id} does not match number of channel {cfg.num_of_channels}."
        )
        return False

    for col in session.columns.difference(["timestamp"]):
        if not pd.api.types.is_float_dtype(session[col]):
            logger.error(f"Column '{col}' in {session_id} is not float type.")
            return False

    if session.isna().any().any():
        logger.error(f"Session file {session_id} contains NaN values.")
        return False

    return True
