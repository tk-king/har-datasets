from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from whar_datasets.core.config import WHARConfig
from whar_datasets.core.processing.downloading_step import DownloadingStep
from whar_datasets.core.processing.sampling_step import SamplingStep
from whar_datasets.core.processing.parsing_step import ParsingStep
from whar_datasets.core.processing.processing_step import ProcessingStep
from whar_datasets.core.processing.windowing_step import WindowingStep
from whar_datasets.core.utils.logging import logger


class ProcessingPipeline:
    def __init__(self, steps: List[ProcessingStep]):
        self.steps = steps

    def run(self, force_recompute: bool) -> Any:
        if force_recompute:
            logger.info("Forcing recompute...")

        results: Any = None
        for step in self.steps:
            results = step.run(results, force_recompute)

        return results


class PreProcessingPipeline(ProcessingPipeline):
    def __init__(self, cfg: WHARConfig):
        # define directories
        self.datasets_dir = Path(cfg.datasets_dir)
        self.dataset_dir = self.datasets_dir / cfg.dataset_id
        self.download_dir = self.dataset_dir / "download"
        self.metadata_dir = self.dataset_dir / "metadata"
        self.sessions_dir = self.dataset_dir / "sessions"
        self.windows_dir = self.dataset_dir / "windows"

        # create directories
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.windows_dir.mkdir(parents=True, exist_ok=True)

        # Create gitignore file if it doesn't exist
        gitignore_path = self.datasets_dir / ".gitignore"
        if not gitignore_path.exists():
            with open(gitignore_path, "w") as f:
                f.write("*")

        # define steos
        self.downloading_step = DownloadingStep(
            cfg, self.datasets_dir, self.dataset_dir, self.download_dir
        )
        self.parsing_step = ParsingStep(
            cfg,
            self.download_dir,
            self.metadata_dir,
            self.sessions_dir,
            [self.downloading_step],
        )
        self.windowing_step = WindowingStep(
            cfg,
            self.metadata_dir,
            self.sessions_dir,
            self.windows_dir,
            [self.parsing_step],
        )

        super().__init__(
            [self.downloading_step, self.parsing_step, self.windowing_step]
        )

    def run(
        self, force_recompute: bool
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        activity_metadata, session_metadata, window_metadata, _ = super().run(
            force_recompute
        )
        return activity_metadata, session_metadata, window_metadata


class PostProcessingPipeline(ProcessingPipeline):
    def __init__(
        self,
        cfg: WHARConfig,
        pre_processing_pipeline: PreProcessingPipeline,
        window_metadata: pd.DataFrame,
        indices: List[int],
    ):
        self.samples_dir = pre_processing_pipeline.dataset_dir / "samples"
        self.metadata_dir = pre_processing_pipeline.metadata_dir
        self.windows_dir = pre_processing_pipeline.windows_dir

        self.featuring_step = SamplingStep(
            cfg,
            self.metadata_dir,
            self.samples_dir,
            self.windows_dir,
            window_metadata,
            indices,
            [pre_processing_pipeline.windowing_step],
        )

        super().__init__([self.featuring_step])

    def run(self, force_recompute: bool) -> Dict[str, List[np.ndarray]] | None:
        samples = super().run(force_recompute)
        return samples
