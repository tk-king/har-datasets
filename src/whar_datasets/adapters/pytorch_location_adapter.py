from typing import Dict, List, Tuple
import torch
from torch.utils.data import DataLoader, Subset

from whar_datasets.adapters.pytorch import PytorchAdapter
from whar_datasets.core.config import WHARConfig
from whar_datasets.core.postprocessing import postprocess
from whar_datasets.core.splitting import get_split_train_val_test
from whar_datasets.core.weighting import compute_class_weights
from whar_datasets.support.sensor_types import get_sensor_types


class PyTorchLocationAdapter(PytorchAdapter):
    def __init__(self, cfg: WHARConfig, override_cache: bool = False, only_imu: bool = False):
        super().__init__(cfg, override_cache)
        sensor_locations, sensor_types = get_sensor_types(cfg.dataset_id)
        self.sensor_locations = sensor_locations
        self.sensor_types = sensor_types

        self.num_channels = len(sensor_locations)
        self.unique_locations = list(dict.fromkeys(sensor_locations))
        self.location_channel_indices: Dict[int, List[int]] = {
            loc: [idx for idx, loc_value in enumerate(sensor_locations) if loc_value == loc]
            for loc in self.unique_locations
        }
        self.location_sensor_types: Dict[int, List[int]] = {
            loc: [sensor_types[idx] for idx in indices]
            for loc, indices in self.location_channel_indices.items()
        }

        self.index_map: List[Tuple[int, int]] = [
            (window_idx, loc)
            for window_idx in range(len(self.window_metadata))
            for loc in self.unique_locations
        ]

        self.samples = None

    def __len__(self) -> int:
        return len(self.index_map)

    def _window_to_dataset_indices(self, window_indices: List[int]) -> List[int]:
        lookup = set(window_indices)
        return [
            dataset_idx
            for dataset_idx, (window_idx, _location) in enumerate(self.index_map)
            if window_idx in lookup
        ]

    def get_dataloaders(
        self,
        train_batch_size: int,
        train_shuffle: bool = True,
        scv_group_index: int | None = None,
        override_cache: bool = False,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        (
            train_window_indices,
            val_window_indices,
            test_window_indices,
        ) = get_split_train_val_test(
            self.cfg,
            self.session_metadata,
            self.window_metadata,
            scv_group_index,
        )

        self.train_indices = self._window_to_dataset_indices(train_window_indices)
        self.val_indices = self._window_to_dataset_indices(val_window_indices)
        self.test_indices = self._window_to_dataset_indices(test_window_indices)

        train_set = Subset(self, self.train_indices)
        test_set = Subset(self, self.test_indices)
        val_set = Subset(self, self.val_indices)
        print(f"train: {len(train_set)} | val: {len(val_set)} | test: {len(test_set)}")

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

        self.samples = postprocess(
            self.cfg,
            train_window_indices,
            self.hashes_dir,
            self.samples_dir,
            self.windows_dir,
            self.window_metadata,
            override_cache,
        )

        return train_loader, val_loader, test_loader

    def get_class_weights(self, dataloader: DataLoader) -> dict:
        window_indices = [self.index_map[idx][0] for idx in dataloader.dataset.indices]  # type: ignore
        return compute_class_weights(
            self.session_metadata,
            self.window_metadata.iloc[window_indices],
        )

    def __getitem__(self, index: int):
        window_idx, location = self.index_map[index]
        super_item = super().__getitem__(window_idx)

        channel_indices = self.location_channel_indices[location]
        grouped_tensors = [
            (
                tensor[..., channel_indices]
                if isinstance(tensor, torch.Tensor)
                and tensor.ndim > 0
                and tensor.shape[-1] == self.num_channels
                else tensor
            )
            for tensor in super_item[1:]
        ]

        return (super_item[0], *grouped_tensors, location, self.location_sensor_types[location])
