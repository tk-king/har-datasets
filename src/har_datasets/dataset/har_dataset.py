from typing import Callable, Tuple
import pandas as pd
from torch import Tensor
import torch
from torch.utils.data import Dataset, Subset, DataLoader

from har_datasets.dataset.pipeline import get_sample, pipeline, get_split
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

        self.dataset_dir, self.window_index, self.windows, self.spectograms = pipeline(
            cfg=cfg, parse=parse, override_csv=override_csv
        )

    def get_dataloaders(
        self,
        subj_cross_val_group_index: int | None = None,
        train_batch_size: int | None = None,
        train_shuffle: bool | None = None,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        # get split indices from config
        self.train_indices, self.val_indices, self.test_indices = get_split(
            cfg=self.cfg,
            window_index=self.window_index,
            subj_cross_val_group_index=subj_cross_val_group_index,
        )

        # specify split subsets
        train_set = Subset(self, self.train_indices)
        test_set = Subset(self, self.test_indices)
        val_set = Subset(self, self.val_indices)

        # override default batch size and shuffle
        batch_size = train_batch_size or self.cfg.dataset.training.batch_size
        shuffle = train_shuffle or self.cfg.dataset.training.shuffle

        # create dataloaders from split
        train_loader = DataLoader(train_set, batch_size, shuffle, collate_fn=collate_fn)
        val_loader = DataLoader(val_set, len(val_set), False, collate_fn=collate_fn)
        test_loader = DataLoader(test_set, 1, False, collate_fn=collate_fn)

        return train_loader, val_loader, test_loader

    def get_class_weights(self) -> dict:
        return compute_class_weights(self.window_index)

    def __len__(self) -> int:
        return len(self.window_index)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor | None]:
        # get label, window and spectogram
        label, window, spect = get_sample(
            cfg=self.cfg,
            index=index,
            dataset_dir=self.dataset_dir,
            window_index=self.window_index,
            windows=self.windows,
            spectograms=self.spectograms,
        )

        # convert to tensors
        y = torch.tensor([label], dtype=torch.long)
        x1 = torch.tensor(window, dtype=torch.float32)
        x2 = torch.tensor(spect, dtype=torch.float32) if spect is not None else None

        return y, x1, x2


def collate_fn(data: list[Tuple[Tensor, Tensor, Tensor | None]]) -> tuple:
    y, x1, x2 = zip(*data)

    tensor_y = torch.stack(y)
    tensor_x1 = torch.stack(x1)
    tensor_x2 = None if None in x2 else torch.stack(x2)

    return tensor_y, tensor_x1, tensor_x2
