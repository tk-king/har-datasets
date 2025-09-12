import random
from typing import Tuple
import numpy as np
from torch import Tensor
import torch
from torch.utils.data import Dataset, Subset, DataLoader

from whar_datasets.core.pipeline import PostProcessingPipeline, PreProcessingPipeline
from whar_datasets.core.sampling import get_label, get_sample
from whar_datasets.core.splitting import get_split_train_val_test
from whar_datasets.core.weighting import compute_class_weights
from whar_datasets.core.config import WHARConfig
from whar_datasets.core.utils.logging import logger


class PytorchAdapter(Dataset[Tuple[Tensor, ...]]):
    def __init__(self, cfg: WHARConfig, force_recompute: bool = False):
        super().__init__()

        torch.manual_seed(cfg.seed)
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        self.generator = torch.Generator()
        self.generator.manual_seed(cfg.seed)

        self.cfg = cfg

        self.pre_processing_pipeline = PreProcessingPipeline(cfg)
        self.session_metadata, self.window_metadata, self.windows = (
            self.pre_processing_pipeline.run(force_recompute)
        )

        logger.info(
            f"subject_ids: {np.sort(self.session_metadata['subject_id'].unique())}"
        )
        logger.info(
            f"activity_ids: {np.sort(self.session_metadata['activity_id'].unique())}"
        )

    def get_dataloaders(
        self,
        train_batch_size: int,
        train_shuffle: bool = True,
        scv_group_index: int | None = None,
        force_recompute: bool = False,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        # get split indices from config
        self.train_indices, self.val_indices, self.test_indices = (
            get_split_train_val_test(
                self.cfg,
                self.session_metadata,
                self.window_metadata,
                scv_group_index,
            )
        )

        # specify split subsets
        train_set = Subset(self, self.train_indices)
        test_set = Subset(self, self.test_indices)
        val_set = Subset(self, self.val_indices)
        logger.info(
            f"train: {len(train_set)} | val: {len(val_set)} | test: {len(test_set)}"
        )

        # create dataloaders from split
        train_loader = DataLoader(
            dataset=train_set,
            batch_size=train_batch_size,
            shuffle=train_shuffle,
            generator=self.generator,
        )

        val_loader = DataLoader(
            dataset=val_set,
            batch_size=len(val_set),
            shuffle=False,
            generator=self.generator,
        )

        test_loader = DataLoader(
            dataset=test_set,
            batch_size=1,
            shuffle=False,
            generator=self.generator,
        )

        self.post_processing_pipeline = PostProcessingPipeline(
            self.cfg,
            self.pre_processing_pipeline,
            self.window_metadata,
            self.train_indices,
            scv_group_index,
        )
        self.samples = self.post_processing_pipeline.run(force_recompute)

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
