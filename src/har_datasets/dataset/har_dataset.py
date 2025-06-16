from typing import Callable, Tuple
import numpy as np
import pandas as pd
from torch import Tensor
import torch
from torch.utils.data import Dataset, Subset, DataLoader

from har_datasets.dataset.pipeline import pipeline, split
from har_datasets.pipeline.weighting import compute_class_weights
from har_datasets.config.config import HARConfig


class HARDataset(Dataset[Tuple[Tensor, Tensor]]):
    def __init__(
        self,
        cfg: HARConfig,
        parse: Callable[[str], pd.DataFrame],
        override_csv: bool = False,
    ):
        super().__init__()

        self.cfg = cfg

        _, self.window_index, self.windows = pipeline(
            cfg=cfg, parse=parse, override_csv=override_csv
        )

        self.train_indices, self.test_indices, self.val_indices = split(
            cfg=cfg, window_index=self.window_index
        )

    def get_dataloaders(
        self, train_batch_size: int | None = None, train_shuffle: bool | None = None
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        # specify split
        train_set = Subset(self, self.train_indices)
        test_set = Subset(self, self.test_indices)
        val_set = Subset(self, self.val_indices)

        # override default batch size and shuffle
        batch_size = train_batch_size or self.cfg.dataset.training.batch_size
        shuffle = train_shuffle or self.cfg.dataset.training.shuffle

        # create dataloaders from split
        train_loader = DataLoader(train_set, batch_size, shuffle)
        test_loader = DataLoader(test_set, 1, False)
        val_loader = DataLoader(val_set, 1, False)

        return train_loader, test_loader, val_loader

    def get_class_weights(self) -> dict:
        return compute_class_weights(self.window_index)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        # get class label of window
        label = self.window_index.loc[index]["activity_id"]
        assert isinstance(label, np.integer)

        # get window as sample
        window_id = self.window_index.loc[index]["window_id"]
        assert isinstance(window_id, np.integer)
        window = self.windows[window_id]

        # drop index since not a feature
        window = window.reset_index(drop=True)

        x = torch.tensor(window.values, dtype=torch.float32)
        y = torch.tensor([label], dtype=torch.long)

        return x, y
