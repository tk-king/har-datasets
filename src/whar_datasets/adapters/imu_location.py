import random
from collections import OrderedDict, defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset

from whar_datasets.core.config import WHARConfig
from whar_datasets.core.postprocessing import postprocess
from whar_datasets.core.preprocessing import preprocess
from whar_datasets.core.sampling import get_sample
from whar_datasets.core.splitting import get_split_train_val_test
from whar_datasets.core.utils.loading import (
    load_session_metadata,
    load_window_metadata,
)


class IMULocationAdapter(Dataset[Tuple[Tensor, Tensor]]):
    """Dataset adapter that exposes per-window 9-axis IMU readings and their body location."""

    _REQUIRED_MEASUREMENTS = ("acc", "gyro", "mag")
    _AXES = ("x", "y", "z")
    _MULTI_TOKEN_ALIASES: Dict[Tuple[str, ...], str] = {
        ("body", "acc"): "acc",
        ("linear", "acc"): "acc",
        ("angvel", "body"): "gyro",
        ("body", "gyro"): "gyro",
    }
    _SINGLE_TOKEN_ALIASES: Dict[str, str] = {
        "acc": "acc",
        "accel": "acc",
        "acceleration": "acc",
        "gyro": "gyro",
        "gyroscope": "gyro",
        "angvel": "gyro",
        "angularvelocity": "gyro",
        "mag": "mag",
        "magn": "mag",
        "magnet": "mag",
        "magnetic": "mag",
        "magnetometer": "mag",
    }

    def __init__(self, cfg: WHARConfig, override_cache: bool = False):
        super().__init__()

        self.cfg = cfg

        dirs = preprocess(cfg, override_cache)
        self.cache_dir, self.windows_dir, self.samples_dir, self.hashes_dir = dirs

        self.session_metadata = load_session_metadata(self.cache_dir)
        self.window_metadata = load_window_metadata(self.cache_dir)

        torch.manual_seed(cfg.seed)
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        self.generator = torch.Generator()
        self.generator.manual_seed(cfg.seed)

        self.location_channel_indices = self._infer_location_channels(cfg.sensor_channels)
        print(f"Detected IMU locations: {list(self.location_channel_indices.keys())}")
        if not self.location_channel_indices:
            raise ValueError(
                "No 9-axis IMU groups could be derived from the configured sensor channels."
            )

        self.location_names = list(self.location_channel_indices.keys())
        self.location_to_idx = {name: idx for idx, name in enumerate(self.location_names)}
        self.location_channels = {
            name: [cfg.sensor_channels[i] for i in indices]
            for name, indices in self.location_channel_indices.items()
        }

        self.index_map = [
            (window_idx, location_name)
            for window_idx in range(len(self.window_metadata))
            for location_name in self.location_names
        ]

        self.samples: Dict[str, List[np.ndarray]] | None = None

    def _infer_location_channels(self, channels: List[str]) -> OrderedDict[str, List[int]]:
        location_axes: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(
            lambda: defaultdict(dict)
        )
        print(location_axes.keys())
        location_canonical: Dict[str, Dict[str, str]] = defaultdict(dict)
        location_order: Dict[str, Dict[str, int]] = defaultdict(dict)
        location_first_seen: Dict[str, int] = {}

        for idx, channel in enumerate(channels):
            parsed = self._parse_channel(channel)
            if parsed is None:
                continue

            location, measurement_key, canonical, axis = parsed

            if location not in location_first_seen:
                location_first_seen[location] = idx

            axes = location_axes[location][measurement_key]
            axes[axis] = idx

            location_canonical[location][measurement_key] = canonical

            if measurement_key not in location_order[location]:
                location_order[location][measurement_key] = idx

        ordered_locations = sorted(location_first_seen.keys(), key=location_first_seen.get)

        location_indices: OrderedDict[str, List[int]] = OrderedDict()
        for location in ordered_locations:
            measurement_axes = location_axes[location]
            canonical_map = location_canonical[location]
            order_map = location_order[location]

            indices: List[int] = []
            valid_location = True

            for canonical in self._REQUIRED_MEASUREMENTS:
                candidates = [
                    (order_map[key], key)
                    for key, axes in measurement_axes.items()
                    if canonical_map[key] == canonical and set(axes.keys()) >= set(self._AXES)
                ]
                if not candidates:
                    valid_location = False
                    break

                selected_key = self._select_measurement_key(candidates, canonical)
                axes = measurement_axes[selected_key]
                indices.extend(axes[axis] for axis in self._AXES)

            if valid_location:
                location_indices[location] = indices

        return location_indices

    def _select_measurement_key(
        self, candidates: List[Tuple[int, str]], canonical: str
    ) -> str:
        canonical_matches = [c for c in candidates if c[1] == canonical]
        if canonical_matches:
            return sorted(canonical_matches)[0][1]

        prefix_matches = [c for c in candidates if c[1].startswith(canonical)]
        if prefix_matches:
            prefix_matches.sort(key=lambda item: (len(item[1]), item[0]))
            return prefix_matches[0][1]

        candidates.sort(key=lambda item: (item[0], len(item[1])))
        return candidates[0][1]

    def _parse_channel(self, channel: str) -> Tuple[str, str, str, str] | None:
        tokens = channel.lower().split("_")
        if len(tokens) < 2:
            return None

        axis = tokens[-1]
        if axis not in self._AXES:
            return None

        body_tokens = tokens[:-1]

        parsed = self._match_multi_token_alias(body_tokens)
        if parsed is None:
            parsed = self._match_single_token_alias(body_tokens)

        if parsed is None:
            return None

        alias_idx, alias_len, canonical = parsed
        location_tokens = body_tokens[:alias_idx]
        measurement_tokens = body_tokens[alias_idx : alias_idx + alias_len]
        suffix_tokens = body_tokens[alias_idx + alias_len :]

        location = "_".join(location_tokens) if location_tokens else "global"
        measurement_key_tokens = [*measurement_tokens, *suffix_tokens]
        measurement_key = "_".join(measurement_key_tokens)

        return location, measurement_key, canonical, axis

    def _match_multi_token_alias(
        self, tokens: List[str]
    ) -> Tuple[int, int, str] | None:
        for alias_tokens, canonical in self._MULTI_TOKEN_ALIASES.items():
            alias_len = len(alias_tokens)
            if alias_len > len(tokens):
                continue
            for idx in range(len(tokens) - alias_len + 1):
                if tuple(tokens[idx : idx + alias_len]) == alias_tokens:
                    return idx, alias_len, canonical
        return None

    def _match_single_token_alias(self, tokens: List[str]) -> Tuple[int, int, str] | None:
        for idx, token in enumerate(tokens):
            canonical = self._SINGLE_TOKEN_ALIASES.get(token)
            if canonical is not None:
                return idx, 1, canonical
        return None

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

        self._train_dataset_indices = self._window_to_dataset_indices(train_window_indices)
        self._val_dataset_indices = self._window_to_dataset_indices(val_window_indices)
        self._test_dataset_indices = self._window_to_dataset_indices(test_window_indices)

        train_subset = Subset(self, self._train_dataset_indices)
        val_subset = Subset(self, self._val_dataset_indices)
        test_subset = Subset(self, self._test_dataset_indices)

        train_loader = DataLoader(
            dataset=train_subset,
            batch_size=train_batch_size,
            shuffle=train_shuffle,
            generator=self.generator,
        )

        val_loader = DataLoader(
            dataset=val_subset,
            batch_size=len(val_subset),
            shuffle=False,
            generator=self.generator,
        )

        test_loader = DataLoader(
            dataset=test_subset,
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

    def _window_to_dataset_indices(self, window_indices: List[int]) -> List[int]:
        lookup = set(window_indices)
        return [
            dataset_idx
            for dataset_idx, (window_idx, _location) in enumerate(self.index_map)
            if window_idx in lookup
        ]

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        window_idx, location_name = self.index_map[index]
        window_row = self.window_metadata.iloc[window_idx]
        window_id = window_row["window_id"]
        sample = get_sample(
            window_idx,
            self.cfg,
            self.samples_dir,
            self.window_metadata,
            self.samples,
        )

        if len(sample) == 0:
            raise ValueError("Post-processed sample is empty for window_id %s" % window_id)

        normalized = sample[0]
        indices = self.location_channel_indices[location_name]
        imu_data = normalized[:, indices]

        x = torch.tensor(imu_data, dtype=torch.float32)
        location_idx = self.location_to_idx[location_name]
        location_tensor = torch.tensor(location_idx, dtype=torch.long)

        return x, location_tensor

