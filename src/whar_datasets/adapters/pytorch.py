import random
from typing import Tuple
import numpy as np
from torch import Tensor
import torch
from torch.utils.data import Dataset, Subset, DataLoader

from whar_datasets.core.process import process
from whar_datasets.core.sample import get_label, get_window
from whar_datasets.core.split import get_split
from whar_datasets.core.utils.loading import load_session_metadata, load_windowing
from whar_datasets.core.weighting import compute_class_weights
from whar_datasets.core.config import WHARConfig


class PytorchAdapter(Dataset[Tuple[Tensor, Tensor]]):
    def __init__(self, cfg: WHARConfig, override_cache: bool = False):
        super().__init__()

        self.cfg = cfg

        self.cache_dir, self.windows_dir = process(cfg, override_cache)
        self.session_metadata = load_session_metadata(self.cache_dir)
        self.window_metadata, self.windows = load_windowing(
            self.cache_dir, self.windows_dir, self.cfg
        )

        print(f"subject_ids: {np.sort(self.session_metadata['subject_id'].unique())}")
        print(f"activity_ids: {np.sort(self.session_metadata['activity_id'].unique())}")

        self.seed = cfg.dataset.training.seed

        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)

        self.generator = torch.Generator()
        self.generator.manual_seed(self.seed)

    def get_dataloaders(
        self,
        train_batch_size: int,
        train_shuffle: bool = True,
        subj_cross_val_group_index: int | None = None,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        # get split indices from config
        train_indices, val_indices, test_indices = get_split(
            self.cfg,
            self.session_metadata,
            self.window_metadata,
            subj_cross_val_group_index,
        )

        # specify split subsets
        train_set = Subset(self, train_indices)
        test_set = Subset(self, test_indices)
        val_set = Subset(self, val_indices)

        print(f"train: {len(train_set)} | val: {len(val_set)} | test: {len(test_set)}")

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

        return train_loader, val_loader, test_loader

    def get_class_weights(self, dataloader: DataLoader) -> dict:
        indices = dataloader.dataset.indices  # type: ignore
        assert indices is not None

        return compute_class_weights(
            self.session_metadata, self.window_metadata.iloc[indices]
        )

    def __len__(self) -> int:
        return len(self.window_metadata)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        # get label, window and window
        label = get_label(index, self.window_metadata, self.session_metadata)
        window = get_window(
            index, self.cfg, self.windows_dir, self.window_metadata, self.windows
        )

        # convert to tensors
        y = torch.tensor(label, dtype=torch.long)
        x = torch.tensor(window, dtype=torch.float32)

        return y, x


# def collate_fn(data: list[Tuple[Tensor, Tensor, Tensor | None]]) -> tuple:
#     y, x1, x2 = zip(*data)

#     tensor_y = torch.stack(y)
#     tensor_x1 = torch.stack(x1)
#     tensor_x2 = None if None in x2 else torch.stack(x2)

#     return tensor_y, tensor_x1, tensor_x2
