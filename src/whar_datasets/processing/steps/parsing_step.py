import shutil
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple, TypeAlias

import pandas as pd
from tqdm import tqdm

from whar_datasets.config.config import WHARConfig
from whar_datasets.processing.steps.processing_step import ProcessingStep
from whar_datasets.utils.loading import load_activity_df, load_session_df, load_sessions
from whar_datasets.utils.logging import logger

base_type: TypeAlias = Any
result_type: TypeAlias = Tuple[
    pd.DataFrame,
    pd.DataFrame,
    Dict[int, pd.DataFrame],
]


class ParsingStep(ProcessingStep):
    def __init__(
        self,
        cfg: WHARConfig,
        raw_dir: Path,
        metadata_dir: Path,
        sessions_dir: Path,
        dependent_on: List[ProcessingStep],
    ):
        super().__init__(cfg, sessions_dir, dependent_on)

        self.download_dir = raw_dir
        self.metadata_dir = metadata_dir
        self.sessions_dir = sessions_dir

        self.hash_name: str = "parsing_hash"
        self.relevant_cfg_keys: Set[str] = {"dataset_id", "activity_id_col"}

    def get_base(self) -> base_type:
        return None

    def check_initial_format(self, base: base_type) -> bool:
        logger.info("Checking download")

        if not self.download_dir.exists():
            logger.warning(f"Download directory not found at '{self.download_dir}'.")
            return False

        logger.info("Download exists")
        return True

    def compute_results(self, base: base_type) -> result_type:
        logger.info("Parsing to common format")

        activity_df, session_df, sessions = self.cfg.parse(
            str(self.download_dir), self.cfg.activity_id_col
        )

        return activity_df, session_df, sessions

    def save_results(self, results: result_type) -> None:
        activity_df, session_df, sessions = results

        logger.info("Saving common format")

        # delete sessions directory if it exists
        if self.sessions_dir.exists():
            shutil.rmtree(self.sessions_dir)

        # create directories if do not exist
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

        # define paths
        activity_df_path = self.metadata_dir / "activity_df.csv"
        session_df_path = self.metadata_dir / "session_df.csv"

        # save activity and session index
        activity_df.to_csv(activity_df_path, index=False)
        session_df.to_csv(session_df_path, index=False)

        # loop over sessions
        loop = tqdm(session_df["session_id"])
        loop.set_description("Caching sessions")

        # save sessions
        for session_id in loop:
            assert isinstance(session_id, int)
            session_path = self.sessions_dir / f"session_{session_id}.csv"
            sessions[session_id].to_csv(session_path, index=False)

    def load_results(self) -> result_type:
        logger.info("Loading common format")

        session_df = load_session_df(self.metadata_dir)
        activity_df = load_activity_df(self.metadata_dir)
        sessions = load_sessions(self.sessions_dir, session_df)

        return activity_df, session_df, sessions
