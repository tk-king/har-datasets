import random
from typing import List, Tuple
import numpy as np
from torch import Tensor
import torch
from torch.utils.data import Dataset, Subset, DataLoader

from whar_datasets.core.pipeline import PostProcessingPipeline, PreProcessingPipeline
from whar_datasets.core.sampling import get_label, get_sample
from whar_datasets.core.splitting import get_split_train_val_test
from whar_datasets.core.weighting import compute_class_weights
from whar_datasets.core.config import WHARConfig


class TorchAdapter(Dataset[Tuple[Tensor, ...]]):
    def __init__(
        self,
        cfg: WHARConfig,
        force_recompute: bool | List[bool] | None = False,
    ):
        super().__init__()

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

    def get_dataloaders(
        self,
        batch_size: int,
        scv_group_index: int | None = None,
        force_recompute: bool | List[bool] | None = False,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        # compute splitting
        self.train_indices, self.val_indices, self.test_indices = (
            get_split_train_val_test(
                self.cfg,
                self.session_metadata,
                self.window_metadata,
                scv_group_index,
            )
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

        # specify split subsets
        train_set = Subset(self, self.train_indices)
        test_set = Subset(self, self.test_indices)
        val_set = Subset(self, self.val_indices)

        # create dataloaders from split
        train_loader = DataLoader(train_set, batch_size, True, generator=self.generator)
        val_loader = DataLoader(val_set, len(val_set), False, generator=self.generator)
        test_loader = DataLoader(test_set, 1, False, generator=self.generator)

        return train_loader, val_loader, test_loader

    def get_class_weights(self, dataloader: DataLoader) -> dict:
        return compute_class_weights(
            self.session_metadata,
            self.window_metadata.iloc[dataloader.dataset.indices],  # type: ignore
        )

    def __len__(self) -> int:
        return len(self.window_metadata)

    def __getitem__(self, index: int) -> Tuple[Tensor, ...]:
        # get label
        label = get_label(index, self.window_metadata, self.session_metadata)

        # get sample
        sample = get_sample(
            index,
            self.cfg,
            self.post_processing_pipeline.samples_dir,
            self.window_metadata,
            self.samples,
        )

        # convert to tensors
        y = torch.tensor(label, dtype=torch.long)
        x = [torch.tensor(s, dtype=torch.float32) for s in sample]

        return (y, *x)
