from pathlib import Path
from typing import Dict, List, Set, Tuple, TypeAlias
import pandas as pd

from whar_datasets.core.config import WHARConfig
from whar_datasets.core.utils.sessions import (
    process_sessions_para,
    process_sessions_seq,
)
from whar_datasets.core.processing.processing_step import ProcessingStep
from whar_datasets.core.utils.caching import cache_window_metadata, cache_windows
from whar_datasets.core.utils.loading import (
    load_activity_metadata,
    load_session_metadata,
    load_window_metadata,
)
from whar_datasets.core.utils.logging import logger
from whar_datasets.core.utils.selecting import select_activities
from whar_datasets.core.utils.validation import validate_common_format

base_type: TypeAlias = Tuple[pd.DataFrame, pd.DataFrame]
result_type: TypeAlias = Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    Dict[str, pd.DataFrame],
]


class WindowingStep(ProcessingStep):
    def __init__(
        self,
        cfg: WHARConfig,
        metadata_dir: Path,
        sessions_dir: Path,
        windows_dir: Path,
        dependent_on: List[ProcessingStep],
    ):
        super().__init__(cfg, windows_dir, dependent_on)

        self.metadata_dir = metadata_dir
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
        }

    def get_base(self) -> base_type:
        activity_metadata = load_activity_metadata(self.metadata_dir)
        session_metadata = load_session_metadata(self.metadata_dir)

        return activity_metadata, session_metadata

    def check_initial_format(self, base: base_type) -> bool:
        activity_metadata, session_metadata = base

        return validate_common_format(
            self.cfg, self.sessions_dir, activity_metadata, session_metadata
        )

    def compute_results(self, base: base_type) -> result_type:
        activity_metadata, session_metadata = base

        logger.info("Compute windowing")

        # select activities
        session_metadata = select_activities(
            session_metadata,
            activity_metadata,
            self.cfg.activity_names,
        )

        # generate windowing
        process_sessions = (
            process_sessions_para if self.cfg.parallelize else process_sessions_seq
        )

        window_metadata, windows = process_sessions(
            self.cfg, self.sessions_dir, session_metadata
        )

        return activity_metadata, session_metadata, window_metadata, windows

    def save_results(self, results: result_type) -> None:
        logger.info("Saving windowing")

        _, _, window_metadata, windows = results

        cache_window_metadata(self.metadata_dir, window_metadata)
        cache_windows(self.windows_dir, window_metadata, windows)

    def load_results(self) -> result_type:
        logger.info("Loading windowing")

        activity_metadata = load_activity_metadata(self.metadata_dir)
        session_metadata = load_session_metadata(self.metadata_dir)
        window_metadata = load_window_metadata(self.metadata_dir)

        df = activity_metadata["activity_id"]
        logger.info(f"activity_ids from {df.min()} to {df.max()}")

        df = session_metadata["subject_id"]
        logger.info(f"subject_ids from {df.min()} to {df.max()}")

        return activity_metadata, session_metadata, window_metadata, {}
