from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from whar_datasets.core.config import WHARConfig
from whar_datasets.core.processing.downloading_step import DownloadingStep
from whar_datasets.core.processing.featuring_step import FeaturingStep
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
        self.datasets_dir = Path(cfg.datasets_dir)
        self.dataset_dir = self.datasets_dir / cfg.dataset_id
        self.cache_dir = (
            Path(cfg.cache_dir) if cfg.cache_dir else self.dataset_dir / "cache"
        )
        self.sessions_dir = self.cache_dir / "sessions"
        self.windows_dir = self.cache_dir / "windows"
        self.hashes_dir = self.cache_dir / "hashes"

        self.downloading_step = DownloadingStep(
            cfg, self.hashes_dir, self.datasets_dir, self.dataset_dir
        )
        self.parsing_step = ParsingStep(
            cfg,
            self.hashes_dir,
            self.cache_dir,
            self.dataset_dir,
            self.sessions_dir,
            [self.downloading_step],
        )
        self.windowing_step = WindowingStep(
            cfg,
            self.hashes_dir,
            self.cache_dir,
            self.sessions_dir,
            self.windows_dir,
            [self.parsing_step],
        )

        super().__init__(
            [self.downloading_step, self.parsing_step, self.windowing_step]
        )

    def run(
        self, force_recompute: bool
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame]]:
        activity_metadata, session_metadata, window_metadata, windows = super().run(
            force_recompute
        )
        return activity_metadata, session_metadata, window_metadata, windows


class PostProcessingPipeline(ProcessingPipeline):
    def __init__(
        self,
        cfg: WHARConfig,
        pre_processing_pipeline: PreProcessingPipeline,
        window_metadata: pd.DataFrame,
        indices: List[int],
        scv_group_index: int | None = None,
    ):
        self.samples_dir = pre_processing_pipeline.cache_dir / "samples"

        self.featuring_step = FeaturingStep(
            cfg,
            pre_processing_pipeline.hashes_dir,
            pre_processing_pipeline.cache_dir,
            self.samples_dir,
            pre_processing_pipeline.windows_dir,
            window_metadata,
            indices,
            scv_group_index,
            [pre_processing_pipeline.windowing_step],
        )

        super().__init__([self.featuring_step])

    def run(self, force_recompute: bool) -> Dict[str, List[np.ndarray]] | None:
        samples = super().run(force_recompute)
        return samples
