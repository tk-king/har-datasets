from pathlib import Path
from typing import Dict, List, Set, TypeAlias

import numpy as np
import pandas as pd

from whar_datasets.core.config import WHARConfig
from whar_datasets.core.processing.processing_step import ProcessingStep
from whar_datasets.core.utils.caching import cache_samples
from whar_datasets.core.utils.loading import (
    load_samples,
    load_windows,
)
from whar_datasets.core.utils.logging import logger
from whar_datasets.core.utils.normalization import (
    get_norm_params,
    normalize_windows_seq,
)
from whar_datasets.core.utils.transform import transform_windows_seq


base_type: TypeAlias = Dict[str, pd.DataFrame]
result_type: TypeAlias = Dict[str, List[np.ndarray]]


class SamplingStep(ProcessingStep):
    def __init__(
        self,
        cfg: WHARConfig,
        metadata_dir: Path,
        samples_dir: Path,
        windows_dir: Path,
        window_metadata: pd.DataFrame,
        indices: List[int],
        dependent_on: List[ProcessingStep],
    ):
        super().__init__(cfg, samples_dir, dependent_on)

        self.metadata_dir = metadata_dir
        self.samples_dir = samples_dir
        self.windows_dir = windows_dir
        self.window_metadata = window_metadata
        self.indices = indices

        self.hash_name: str = "sampling_hash"
        self.relevant_cfg_keys: Set[str] = {
            "given_train_test_subj_ids",
            "subj_cross_val_split_groups",
            "val_percentage",
            "normalization",
            "transform",
            "cache_postprocessing",
        }
        self.relevant_values = [str(i) for i in self.indices]

    def get_base(self, base: base_type | None) -> base_type:
        windows = (
            base
            if base is not None
            else load_windows(self.window_metadata, self.windows_dir)
        )
        return windows

    def check_initial_format(self, base: base_type) -> bool:
        return True

    def compute_results(self, base: base_type) -> result_type:
        windows = base

        logger.info("Computing samples...")

        norm_params = get_norm_params(
            self.cfg, self.indices, self.window_metadata, windows
        )

        normalized = normalize_windows_seq(
            self.cfg, norm_params, self.window_metadata, windows
        )

        transformed = transform_windows_seq(self.cfg, self.window_metadata, normalized)

        # combine normalized and transformed into samples
        assert windows.keys() == normalized.keys() == transformed.keys()
        samples: Dict[str, List[np.ndarray]] = {
            window_id: [normalized[window_id], *transformed[window_id]]
            for window_id in normalized.keys()
        }

        return samples  # if self.cfg.in_memory else None

    def save_results(self, results: result_type) -> None:
        samples = results
        logger.info("Saving samples")
        cache_samples(self.samples_dir, self.window_metadata, samples)

    def load_results(self) -> result_type:
        logger.info("Loading samples")
        return load_samples(self.window_metadata, self.samples_dir)
