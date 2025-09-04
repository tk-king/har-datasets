import os

from whar_datasets.core.utils.hashing import load_cfg_hash
from whar_datasets.core.utils.logging import logger


def check_download(dataset_dir: str) -> bool:
    logger.info("Checking download...")

    if not os.path.exists(dataset_dir):
        logger.warning(f"Dataset directory not found at '{dataset_dir}'.")
        return False

    logger.info("Download exists.")
    return True


def check_sessions(cache_dir: str, sessions_dir: str) -> bool:
    logger.info("Checking sessions...")

    if not os.path.exists(sessions_dir):
        logger.warning(f"Sessions directory not found at '{sessions_dir}'.")
        return False

    if len(os.listdir(sessions_dir)) == 0:
        logger.warning(f"Sessions directory '{sessions_dir}' is empty.")
        return False

    session_metadata_path = os.path.join(cache_dir, "session_metadata.parquet")
    if not os.path.exists(session_metadata_path):
        logger.warning(f"Session index file not found at '{session_metadata_path}'.")
        return False

    activity_metadata_path = os.path.join(cache_dir, "activity_metadata.parquet")
    if not os.path.exists(activity_metadata_path):
        logger.warning(f"Activity index file not found at '{activity_metadata_path}'.")
        return False

    logger.info("Sessions exist.")
    return True


def check_windowing(
    cache_dir: str, windows_dir: str, hashes_dir: str, cfg_hash: str
) -> bool:
    logger.info("Checking windowing...")

    if not os.path.exists(windows_dir):
        logger.warning(f"Windows directory not found at '{windows_dir}'.")
        return False

    if len(os.listdir(windows_dir)) == 0:
        logger.warning(f"Windows directory '{windows_dir}' is empty.")
        return False

    window_metadata_path = os.path.join(cache_dir, "window_metadata.parquet")

    if not os.path.exists(window_metadata_path):
        logger.warning(f"Window index file not found at '{window_metadata_path}'.")
        return False

    current_hash = load_cfg_hash(hashes_dir)

    if cfg_hash != current_hash:
        logger.warning("Config hash mismatch.")
        return False

    logger.info("Windowing exists.")
    return True
