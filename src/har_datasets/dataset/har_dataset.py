from typing import Callable, Tuple
import numpy as np
import pandas as pd
from torch import Tensor
import torch
from torch.utils.data import Dataset, Subset, DataLoader

from har_datasets.dataset.pipeline import pipeline, split
from har_datasets.pipeline.weighting import compute_class_weights
from har_datasets.config.config import HARConfig


class HARDataset(Dataset[Tuple[Tensor, Tensor | None, Tensor | None]]):
    def __init__(
        self,
        cfg: HARConfig,
        parse: Callable[[str], pd.DataFrame],
        override_csv: bool = False,
    ):
        super().__init__()

        self.cfg = cfg

        _, self.window_index, self.windows, self.spectograms = pipeline(
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
        train_loader = DataLoader(train_set, batch_size, shuffle, collate_fn=collate_fn)
        test_loader = DataLoader(test_set, 1, False, collate_fn=collate_fn)
        val_loader = DataLoader(val_set, len(val_set), False, collate_fn=collate_fn)

        return train_loader, test_loader, val_loader

    def get_class_weights(self) -> dict:
        return compute_class_weights(self.window_index)

    def __len__(self) -> int:
        return len(self.window_index)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor | None, Tensor | None]:
        # get class label of window
        label = self.window_index.loc[index]["activity_id"]
        assert isinstance(label, np.integer)

        # get window_id
        window_id = self.window_index.loc[index]["window_id"]
        assert isinstance(window_id, np.integer)

        # get window and spectogram as samples
        window = self.windows[window_id].values if self.windows is not None else None
        spect = self.spectograms[window_id] if self.spectograms is not None else None

        # convert to tensors
        y = torch.tensor([label], dtype=torch.long)
        x1 = torch.tensor(window, dtype=torch.float32) if window is not None else None
        x2 = torch.tensor(spect, dtype=torch.float32) if spect is not None else None

        return y, x1, x2


def collate_fn(data: list[Tuple[Tensor, Tensor | None, Tensor | None]]) -> tuple:
    y, x1, x2 = zip(*data)

    tensor_y = torch.stack(y)
    tensor_x1 = None if None in x1 else torch.stack(x1)
    tensor_x2 = None if None in x2 else torch.stack(x2)

    return tensor_y, tensor_x1, tensor_x2
