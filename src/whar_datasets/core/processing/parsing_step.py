import os
from pathlib import Path
import shutil
from typing import Any, Dict, List, Set, Tuple

import pandas as pd
from tqdm import tqdm

from whar_datasets.core.config import WHARConfig
from whar_datasets.core.processing.processing_step import ProcessingStep
from whar_datasets.core.utils.loading import (
    load_activity_metadata,
    load_session_metadata,
    load_sessions,
)
from whar_datasets.core.utils.logging import logger


class ParsingStep(ProcessingStep):
    def __init__(
        self,
        cfg: WHARConfig,
        hash_dir: Path,
        cache_dir: Path,
        dataset_dir: Path,
        sessions_dir: Path,
        dependent_on: List[ProcessingStep],
    ):
        super().__init__(cfg, hash_dir, dependent_on)

        self.cache_dir = cache_dir
        self.dataset_dir = dataset_dir
        self.sessions_dir = sessions_dir

        self.hash_name: str = "parsing_hash"
        self.relevant_cfg_keys: Set[str] = {
            "dataset_id",
            "activity_id_col",
        }

    def check_initial_format(self, base: None) -> bool:
        logger.info("Checking download...")

        if not self.dataset_dir.exists():
            logger.warning(f"Dataset directory not found at '{self.dataset_dir}'.")
            return False

        logger.info("Download exists.")
        return True

    def compute_results(
        self, base: None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, pd.DataFrame]]:
        logger.info("Parsing to common format...")

        activity_metadata, session_metadata, sessions = self.cfg.parse(
            str(self.dataset_dir), self.cfg.activity_id_col
        )

        return activity_metadata, session_metadata, sessions

    def save_results(self, results: Any) -> None:
        logger.info("Saving common format...")

        activity_metadata, session_metadata, sessions = results

        # delete sessions directory if it exists
        if self.sessions_dir.exists():
            shutil.rmtree(self.sessions_dir)

        # create directories if do not exist
        os.makedirs(self.sessions_dir, exist_ok=True)

        # define paths
        activity_metadata_path = self.cache_dir / "activity_metadata.parquet"
        session_metadata_path = self.cache_dir / "session_metadata.parquet"

        # save activity and session index
        activity_metadata.to_parquet(activity_metadata_path, index=True)
        session_metadata.to_parquet(session_metadata_path, index=True)

        # loop over sessions
        loop = tqdm(session_metadata["session_id"])
        loop.set_description("Caching sessions")

        # save sessions
        for session_id in loop:
            assert isinstance(session_id, int)
            session_path = self.sessions_dir / f"session_{session_id}.parquet"
            sessions[session_id].to_parquet(session_path, index=False)

    def load_results(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, pd.DataFrame]]:
        logger.info("Loading common format...")

        session_metadata = load_session_metadata(self.cache_dir)
        activity_metadata = load_activity_metadata(self.cache_dir)
        sessions = load_sessions(self.sessions_dir, session_metadata)

        return activity_metadata, session_metadata, sessions
