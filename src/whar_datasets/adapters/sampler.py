import random
from typing import List, Tuple
import numpy as np
from torch import Tensor
import torch

from whar_datasets.core.postprocessing import postprocess
from whar_datasets.core.preprocessing import preprocess
from whar_datasets.core.sampling import get_label, get_sample
from whar_datasets.core.splitting import get_split_train_test
from whar_datasets.core.utils.loading import load_session_metadata, load_window_metadata
from whar_datasets.core.weighting import compute_class_weights
from whar_datasets.core.config import WHARConfig


class WHARSampler:
    def __init__(self, cfg: WHARConfig, override_cache: bool = False) -> None:
        self.cfg = cfg

        dirs = preprocess(cfg, override_cache)
        self.cache_dir, self.windows_dir, self.samples_dir, self.hashes_dir = dirs

        self.session_metadata = load_session_metadata(self.cache_dir)
        self.window_metadata = load_window_metadata(self.cache_dir)

        print(f"subject_ids: {np.sort(self.session_metadata['subject_id'].unique())}")
        print(f"activity_ids: {np.sort(self.session_metadata['activity_id'].unique())}")

        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

    def prepare(self, scv_group_index: int, override_cache: bool = False) -> None:
        # get split indices from config
        self.train_indices, self.test_indices = get_split_train_test(
            self.cfg,
            self.session_metadata,
            self.window_metadata,
            scv_group_index,
        )

        # normalize and transform windows
        self.samples = postprocess(
            self.cfg,
            self.train_indices,
            self.hashes_dir,
            self.samples_dir,
            self.windows_dir,
            self.window_metadata,
            override_cache,
        )

    def get_class_weights(self, indices: List[int]) -> dict:
        return compute_class_weights(
            self.session_metadata,
            self.window_metadata.iloc[indices],
        )

    def sample(
        self,
        num_samples: int,
        indices: List[int],
        activity_id: int | None = None,
        seed: int | None = None,
    ) -> Tuple[Tensor, ...]:
        assert indices is not None
        assert num_samples > 0

        if activity_id is not None:
            subset = self.window_metadata.iloc[indices]

            # Merge with session_metadata to get activity_id info
            merged = subset.merge(
                self.session_metadata[["session_id", "activity_id"]],
                on="session_id",
                how="left",
            )

            # Filter by activity_id
            filtered = merged[merged["activity_id"] == activity_id]
            print(filtered["activity_id"].value_counts())
            indices = filtered.index.to_list()

        # if seed is set make reproducable
        # else seeding with None will be random
        random.seed(seed)
        random.shuffle(indices)

        assert len(indices) >= num_samples
        indices = indices[:num_samples]

        labels = [
            get_label(i, self.window_metadata, self.session_metadata) for i in indices
        ]  # (num_samples)

        samples = [
            get_sample(
                i,
                self.cfg,
                self.samples_dir,
                self.window_metadata,
                self.samples,
            )
            for i in indices
        ]  # (num_samples, num_features)

        samples = list(zip(*samples))
        # (num_features, num_samples)

        y = torch.stack([torch.tensor(l, dtype=torch.long) for l in labels])
        x = [
            torch.stack([torch.tensor(s, dtype=torch.float32) for s in samples[i]])
            for i in range(len(samples))
        ]

        return (y, *x)
