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


class ProcessingPipeline:
    def __init__(self, steps: List[ProcessingStep]):
        self.steps = steps

    def run(self, force_recompute: bool | List[bool] | None = None) -> Any:
        if isinstance(force_recompute, list):
            assert len(self.steps) == len(force_recompute)
            for step, fr in zip(self.steps, force_recompute):
                step.run(fr)
        elif isinstance(force_recompute, bool):
            for step in self.steps:
                step.run(force_recompute)


class PreProcessingPipeline(ProcessingPipeline):
    def __init__(self, cfg: WHARConfig):
        # define directories
        self.datasets_dir = Path(cfg.datasets_dir)
        self.dataset_dir = self.datasets_dir / cfg.dataset_id
        self.raw_dir = self.dataset_dir / "data"
        self.metadata_dir = self.dataset_dir / "metadata"
        self.sessions_dir = self.dataset_dir / "sessions"
        self.windows_dir = self.dataset_dir / "windows"

        # create directories
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
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
            cfg=cfg,
            datasets_dir=self.datasets_dir,
            dataset_dir=self.dataset_dir,
            raw_dir=self.raw_dir,
        )

        self.parsing_step = ParsingStep(
            cfg=cfg,
            raw_dir=self.raw_dir,
            metadata_dir=self.metadata_dir,
            sessions_dir=self.sessions_dir,
            dependent_on=[self.downloading_step],
        )

        self.windowing_step = WindowingStep(
            cfg=cfg,
            metadata_dir=self.metadata_dir,
            sessions_dir=self.sessions_dir,
            windows_dir=self.windows_dir,
            dependent_on=[self.parsing_step],
        )

        super().__init__(
            steps=[self.downloading_step, self.parsing_step, self.windowing_step]
        )

    def run(
        self, force_recompute: bool | List[bool] | None = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        super().run(force_recompute)

        activity_metadata, session_metadata, window_metadata, _ = (
            self.windowing_step.load_results()
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
        self.cfg = cfg
        self.samples_dir = pre_processing_pipeline.dataset_dir / "samples"
        self.metadata_dir = pre_processing_pipeline.metadata_dir
        self.windows_dir = pre_processing_pipeline.windows_dir

        self.featuring_step = SamplingStep(
            cfg=cfg,
            metadata_dir=self.metadata_dir,
            samples_dir=self.samples_dir,
            windows_dir=self.windows_dir,
            window_metadata=window_metadata,
            indices=indices,
            dependent_on=[pre_processing_pipeline.windowing_step],
        )

        super().__init__(steps=[self.featuring_step])

    def run(
        self, force_recompute: bool | List[bool] | None = None
    ) -> Dict[str, List[np.ndarray]] | None:
        super().run(force_recompute)

        samples = self.featuring_step.load_results() if self.cfg.in_memory else None

        return samples
