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


class TorchAdapter(Dataset):
    def __init__(self, cfg: WHARConfig):
        self.cfg = cfg
        self._set_seed()
        self.pre_pipeline = PreProcessingPipeline(cfg)

    def __len__(self) -> int:
        return len(self.window_meta)

    def __getitem__(self, index: int) -> Tuple[Tensor, ...]:
        label = get_label(index, self.window_meta, self.session_meta)
        sample = get_sample(
            index,
            self.cfg,
            self.post_pipeline.samples_dir,
            self.window_meta,
            self.samples,
        )

        y = torch.tensor(label, dtype=torch.long)
        x = [torch.tensor(s, dtype=torch.float32) for s in sample]

        return (y, *x)

    def _set_seed(self):
        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        random.seed(self.cfg.seed)
        self.generator = torch.Generator()
        self.generator.manual_seed(self.cfg.seed)

    def preprocess(self, force_recompute: bool | List[bool] | None = False):
        self.activity_meta, self.session_meta, self.window_meta = self.pre_pipeline.run(
            force_recompute
        )

    def postprocess(
        self,
        fold_index: int | None = None,
        force_recompute: bool | List[bool] | None = False,
    ):
        self.train_indices, self.val_indices, self.indices = get_split_train_val_test(
            self.cfg, self.session_meta, self.window_meta, fold_index
        )
        self.post_pipeline = PostProcessingPipeline(
            self.cfg, self.pre_pipeline, self.window_meta, self.train_indices
        )
        self.samples = self.post_pipeline.run(force_recompute)

    def get_dataloaders(self, batch_size: int) -> dict[str, DataLoader]:
        train_set = Subset(self, self.train_indices)
        test_set = Subset(self, self.val_indices)
        val_set = Subset(self, self.indices)

        train_loader = DataLoader(train_set, batch_size, True, generator=self.generator)
        val_loader = DataLoader(val_set, len(val_set), False, generator=self.generator)
        test_loader = DataLoader(test_set, 1, False, generator=self.generator)

        return {"train": train_loader, "val": val_loader, "test": test_loader}

    def get_class_weights(self, dataloader: DataLoader) -> dict:
        return compute_class_weights(
            self.session_meta,
            self.window_meta.iloc[dataloader.dataset.indices],  # type: ignore
        )
