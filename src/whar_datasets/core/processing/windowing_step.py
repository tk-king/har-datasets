from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import pandas as pd

from whar_datasets.core.config import WHARConfig
from whar_datasets.core.utils.sessions import (
    process_sessions_parallely,
    process_sessions_sequentially,
)
from whar_datasets.core.processing.processing_step import ProcessingStep
from whar_datasets.core.utils.caching import cache_window_metadata, cache_windows
from whar_datasets.core.utils.loading import (
    load_activity_metadata,
    load_session_metadata,
    load_sessions,
    load_window_metadata,
    load_windows,
)
from whar_datasets.core.utils.logging import logger
from whar_datasets.core.utils.selecting import select_activities
from whar_datasets.core.utils.validation import validate_common_format


class WindowingStep(ProcessingStep):
    def __init__(
        self,
        cfg: WHARConfig,
        hash_dir: Path,
        cache_dir: Path,
        sessions_dir: Path,
        windows_dir: Path,
        dependent_on: List[ProcessingStep],
        force_results: bool = True,
    ):
        super().__init__(cfg, hash_dir, dependent_on, force_results)

        self.cache_dir = cache_dir
        self.sessions_dir = sessions_dir
        self.windows_dir = windows_dir

        self.hash_name: str = "windowing_hash"
        self.relevant_cfg_keys: Set[str] = {
            "sampling_freq",
            "activity_names",
            "sensor_channels",
            "window_time",
            "window_overlap",
            "resampling_freq",
            "cache_preprocessing",
        }

    def check_initial_format(self, base: Any) -> bool:
        return validate_common_format(self.cfg, self.cache_dir, self.sessions_dir)

    def compute_results(
        self, base: Any | None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame]]:
        logger.info("Windowing...")

        if base is None:
            activity_metadata = load_activity_metadata(self.cache_dir)
            session_metadata = load_session_metadata(self.cache_dir)
            sessions = load_sessions(self.sessions_dir, session_metadata)
        else:
            activity_metadata, session_metadata, sessions = base

        # select activities
        session_metadata = select_activities(
            session_metadata,
            activity_metadata,
            self.cfg.activity_names,
        )

        # generate windowing
        window_metadata, windows = (
            process_sessions_parallely(self.cfg, session_metadata, sessions)
            if self.cfg.in_parallel
            else process_sessions_sequentially(self.cfg, session_metadata, sessions)
        )

        return session_metadata, window_metadata, windows

    def save_results(self, results: Any) -> None:
        logger.info("Saving windowing...")

        _, window_metadata, windows = results

        cache_window_metadata(self.cache_dir, window_metadata)
        cache_windows(self.windows_dir, window_metadata, windows)

    def load_results(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame]]:
        logger.info("Loading windowing...")

        activity_metadata = load_activity_metadata(self.cache_dir)
        session_metadata = load_session_metadata(self.cache_dir)
        window_metadata = load_window_metadata(self.cache_dir)
        windows = load_windows(window_metadata, self.windows_dir)

        df = activity_metadata["activity_id"]
        logger.info(f"activity_ids from {df.min()} to {df.max()}")
        df = session_metadata["subject_id"]
        logger.info(f"subject_ids from {df.min()} to {df.max()}")

        return activity_metadata, session_metadata, window_metadata, windows
