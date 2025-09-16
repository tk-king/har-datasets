from pathlib import Path

from whar_datasets.core.utils.logging import logger


def check_windowing(cache_dir: Path, windows_dir: Path) -> bool:
    logger.info("Checking windowing")

    if not windows_dir.exists():
        logger.warning(f"Windows directory not found at '{windows_dir}'.")
        return False

    if len(list(windows_dir.iterdir())) == 0:
        logger.warning(f"Windows directory '{windows_dir}' is empty.")
        return False

    window_metadata_path = cache_dir / "window_metadata.parquet"

    if not window_metadata_path.exists():
        logger.warning(f"Window index file not found at '{window_metadata_path}'.")
        return False

    logger.info("Windowing exists.")
    return True
