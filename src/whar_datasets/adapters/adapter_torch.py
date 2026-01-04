import random
from typing import Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset

from whar_datasets.config.config import WHARConfig
from whar_datasets.loading.loader import Loader
from whar_datasets.splitting.split import Split


class TorchAdapter(Dataset):
    def __init__(self, cfg: WHARConfig, loader: Loader, split: Split):
        self.cfg = cfg

        self.loader = loader
        self.split = split

        self._set_seed()

    def _set_seed(self):
        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        random.seed(self.cfg.seed)
        self.generator = torch.Generator()
        self.generator.manual_seed(self.cfg.seed)

    def __len__(self) -> int:
        return len(self.loader)

    def __getitem__(self, index: int) -> Tuple[Tensor, ...]:
        activity_label, subject_label, sample = self.loader.get_triple(index)

        y = torch.tensor(activity_label, dtype=torch.long)
        x = [torch.tensor(s, dtype=torch.float32) for s in sample]

        return (y, *x)

    def get_dataloaders(self, batch_size: int) -> dict[str, DataLoader]:
        train_set = Subset(self, self.split.train_indices)
        test_set = Subset(self, self.split.val_indices)
        val_set = Subset(self, self.split.test_indices)

        train_loader = DataLoader(train_set, batch_size, True, generator=self.generator)
        val_loader = DataLoader(val_set, len(val_set), False, generator=self.generator)
        test_loader = DataLoader(test_set, 1, False, generator=self.generator)

        return {"train": train_loader, "val": val_loader, "test": test_loader}
