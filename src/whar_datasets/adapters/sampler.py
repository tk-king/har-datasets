import random
from typing import List, Tuple
from matplotlib import pyplot as plt
import numpy as np
from torch import Tensor
import torch

from whar_datasets.core.pipeline import PostProcessingPipeline, PreProcessingPipeline
from whar_datasets.core.sampling import get_label, get_sample
from whar_datasets.core.splitting import get_split_train_test
from whar_datasets.core.weighting import compute_class_weights
from whar_datasets.core.config import WHARConfig


class Sampler:
    def __init__(
        self, cfg: WHARConfig, force_recompute: bool | List[bool] | None = False
    ) -> None:
        self.cfg = cfg

        # ensure correct seeding
        torch.manual_seed(cfg.seed)
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        self.generator = torch.Generator()
        self.generator.manual_seed(cfg.seed)

        # preform preprocessing
        self.pre_processing_pipeline = PreProcessingPipeline(cfg)
        results = self.pre_processing_pipeline.run(force_recompute)
        self.activity_metadata, self.session_metadata, self.window_metadata = results

    def prepare(self, scv_group_index: int, force_recompute: bool = False) -> None:
        # compute splitting
        self.train_indices, self.test_indices = get_split_train_test(
            self.cfg,
            self.session_metadata,
            self.window_metadata,
            scv_group_index,
        )

        # define postprocessing pipeline
        self.post_processing_pipeline = PostProcessingPipeline(
            self.cfg,
            self.pre_processing_pipeline,
            self.window_metadata,
            self.train_indices,
        )

        # perform postprocessing
        self.samples = self.post_processing_pipeline.run(force_recompute)

    def get_class_weights(self, indices: List[int]) -> dict:
        return compute_class_weights(
            self.session_metadata,
            self.window_metadata.iloc[indices],
        )

    def plot_indices_statistics(self, indices: List[int]) -> None:
        subset = self.window_metadata.iloc[indices]
        merged = subset.merge(
            self.session_metadata[["session_id", "subject_id", "activity_id"]],
            on="session_id",
            how="left",
        )
        counts = (
            merged.groupby(["subject_id", "activity_id"])
            .size()
            .reset_index(name="num_samples")
        )

        # pivot for easier plotting (subjects on x, activities as groups)
        pivot_table = counts.pivot(
            index="subject_id", columns="activity_id", values="num_samples"
        ).fillna(0)

        # plot
        pivot_table.plot(kind="bar", stacked=False, figsize=(12, 4))

        plt.title("number of samples per subject and activity")
        plt.xlabel("subject_id")
        plt.ylabel("number of samples")
        plt.legend(title="activity_id")
        plt.tight_layout()
        plt.show()

    def filter_indices(
        self,
        indices: List[int],
        subject_id: int | None = None,
        activity_id: int | None = None,
    ):
        assert indices is not None

        if subject_id is not None:
            subset = self.window_metadata.iloc[indices]

            # Merge with session_metadata to get subject_id info
            merged = subset.merge(
                self.session_metadata[["session_id", "subject_id"]],
                on="session_id",
                how="left",
            )

            # Filter by subject_id
            filtered = merged[merged["subject_id"] == subject_id]
            indices = filtered.index.to_list()

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
            indices = filtered.index.to_list()

        return indices

    def sample(
        self,
        num_samples: int,
        indices: List[int],
        subject_id: int | None = None,
        activity_id: int | None = None,
        seed: int | None = None,
    ) -> Tuple[Tensor, ...]:
        assert indices is not None
        assert num_samples > 0

        indices = self.filter_indices(indices, subject_id, activity_id)

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
                self.post_processing_pipeline.samples_dir,
                self.window_metadata,
                self.samples,
            )
            for i in indices
        ]  # (num_samples, num_features)

        samples = list(zip(*samples))
        # (num_features, num_samples)

        y = torch.stack([torch.tensor(l, dtype=torch.long) for l in labels])  # noqa: E741
        x = [
            torch.stack([torch.tensor(s, dtype=torch.float32) for s in samples[i]])
            for i in range(len(samples))
        ]

        return (y, *x)
