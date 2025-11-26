import random
from typing import Tuple
import numpy as np
from torch import Tensor
import torch
from torch.utils.data import Dataset, Subset, DataLoader

from whar_datasets.core.postprocessing import postprocess
from whar_datasets.core.preprocessing import preprocess
from whar_datasets.core.sampling import get_label, get_sample
from whar_datasets.core.splitting import get_split_train_val_test
from whar_datasets.core.utils.loading import load_session_metadata, load_window_metadata
from whar_datasets.core.weighting import compute_class_weights
from whar_datasets.core.config import WHARConfig


class NumpyLocationAdapter:
    def __init__(self, cfg: WHARConfig, override_cache: bool = False):
        self.cfg = cfg
        self.override_cache = override_cache

        dirs = preprocess(cfg, override_cache)

        self.cache_dir, self.windows_dir, self.samples_dir, self.hashes_dir = dirs

        self.session_metadata = load_session_metadata(self.cache_dir)
        self.window_metadata = load_window_metadata(self.cache_dir)

        print(f"subject_ids: {np.sort(self.session_metadata['subject_id'].unique())}")
        print(f"activity_ids: {np.sort(self.session_metadata['activity_id'].unique())}")

        np.random.seed(cfg.seed)

        self.train_indices, self.val_indices, self.test_indices = (
            get_split_train_val_test(
                self.cfg,
                self.session_metadata,
                self.window_metadata,
                None,
            )
        )

        self.samples = postprocess(
            self.cfg,
            self.train_indices,
            self.hashes_dir,
            self.samples_dir,
            self.windows_dir,
            self.window_metadata,
            override_cache,
        )

        # Load all the data in memory
        labels = []
        samples = []
        for idx in range(len(self.window_metadata)):
            label = get_label(
                idx,
                self.window_metadata,
                self.session_metadata,
            )
            sample = get_sample(
                idx,
                self.cfg,
                self.samples_dir,
                self.window_metadata,
                self.samples
            )
            labels.append(label)
            samples.append(sample)

        self.labels = np.array(labels)
        self.samples = np.array(samples)
