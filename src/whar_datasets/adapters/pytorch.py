import random
from typing import Callable, List, Tuple
import numpy as np
import pandas as pd
from torch import Tensor
import torch
from torch.utils.data import Dataset, Subset, DataLoader

from whar_datasets.core.process import process
from whar_datasets.core.sample import get_label, get_window
from whar_datasets.core.split import get_split
from whar_datasets.core.utils.loading import load_session_index, load_windowing
from whar_datasets.core.weighting import compute_class_weights
from whar_datasets.core.config import WHARConfig


class PytorchAdapter(Dataset[Tuple[Tensor, Tensor]]):
    def __init__(
        self,
        cfg: WHARConfig,
        parse: Callable[
            [str, str], Tuple[pd.DataFrame, pd.DataFrame, List[pd.DataFrame]]
        ],
        override_cache: bool = False,
    ):
        super().__init__()

        self.cfg = cfg

        self.cache_dir, self.windows_dir = process(cfg, parse, override_cache)
        self.session_index = load_session_index(self.cache_dir)
        self.window_index, self.windows = load_windowing(
            self.cache_dir, self.windows_dir, self.cfg
        )

        self.seed = cfg.dataset.training.seed

        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)

        self.generator = torch.Generator()
        self.generator.manual_seed(self.seed)

    def get_dataloaders(
        self,
        subj_cross_val_group_index: int | None = None,
        train_batch_size: int | None = None,
        train_shuffle: bool | None = None,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        # get split indices from config
        train_indices, val_indices, test_indices = get_split(
            self.cfg, self.session_index, self.window_index, subj_cross_val_group_index
        )

        # specify split subsets
        train_set = Subset(self, train_indices)
        test_set = Subset(self, test_indices)
        val_set = Subset(self, val_indices)

        print(self.session_index["subject_id"].value_counts())
        print(self.session_index["activity_id"].value_counts())
        print(f"train: {len(train_set)} | val: {len(val_set)} | test: {len(test_set)}")

        # override default batch size and shuffle
        batch_size = train_batch_size or self.cfg.dataset.training.batch_size
        shuffle = train_shuffle or self.cfg.dataset.training.shuffle

        # create dataloaders from split
        train_loader = DataLoader(
            dataset=train_set,
            batch_size=batch_size,
            shuffle=shuffle,
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

        return train_loader, val_loader, test_loader

    def get_class_weights(self, dataloader: DataLoader) -> dict:
        indices = dataloader.dataset.indices  # type: ignore
        assert indices is not None

        return compute_class_weights(
            self.session_index, self.window_index.iloc[indices]
        )

    def __len__(self) -> int:
        return len(self.window_index)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        # get label, window and window
        label = get_label(index, self.window_index, self.session_index)
        window = get_window(
            index, self.cfg, self.windows_dir, self.window_index, self.windows
        )

        # convert to tensors
        y = torch.tensor([label], dtype=torch.long)
        x = torch.tensor(window, dtype=torch.float32)

        return y, x


# def collate_fn(data: list[Tuple[Tensor, Tensor, Tensor | None]]) -> tuple:
#     y, x1, x2 = zip(*data)

#     tensor_y = torch.stack(y)
#     tensor_x1 = torch.stack(x1)
#     tensor_x2 = None if None in x2 else torch.stack(x2)

#     return tensor_y, tensor_x1, tensor_x2
