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
        self.seed = cfg.seed

        self._set_seed()
        self.pre_pipeline = PreProcessingPipeline(cfg)

    def prepare_data(self, force_recompute: bool | List[bool] | None = False):
        metadata = self.pre_pipeline.run(force_recompute)
        self.activity_meta, self.session_meta, self.window_meta = metadata

    def setup_splits(self, scv_group_index=None, recompute=False):
        self.train_idx, self.val_idx, self.test_idx = get_split_train_val_test(
            self.cfg, self.session_meta, self.window_meta, scv_group_index
        )
        self.post_pipeline = PostProcessingPipeline(
            self.cfg, self.pre_pipeline, self.window_meta, self.train_idx
        )
        self.samples = self.post_pipeline.run(recompute)

    def dataloaders(self, batch_size=32) -> dict[str, DataLoader]:
        # specify split subsets
        train_set = Subset(self, self.train_idx)
        test_set = Subset(self, self.val_idx)
        val_set = Subset(self, self.test_idx)

        # create dataloaders from split
        train_loader = DataLoader(train_set, batch_size, True, generator=self.generator)
        val_loader = DataLoader(val_set, len(val_set), False, generator=self.generator)
        test_loader = DataLoader(test_set, 1, False, generator=self.generator)

        return {"train": train_loader, "val": val_loader, "test": test_loader}

    def get_class_weights(self, indices: List[int]) -> dict:
        return compute_class_weights(self.session_meta, self.window_meta.iloc[indices])

    def __getitem__(self, idx: int):
        label = get_label(idx, self.window_meta, self.session_meta)
        sample = get_sample(
            idx,
            self.cfg,
            self.post_pipeline.samples_dir,
            self.window_meta,
            self.samples,
        )
        x = torch.stack([torch.as_tensor(s, dtype=torch.float32) for s in sample])
        y = torch.as_tensor(label, dtype=torch.long)
        return x, y

    def __len__(self):
        return len(self.window_meta)

    def _set_seed(self):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        self.generator = torch.Generator()
        self.generator.manual_seed(self.cfg.seed)
