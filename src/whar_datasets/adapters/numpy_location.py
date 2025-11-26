from typing import Dict, List, Tuple

import numpy as np

from whar_datasets.core.config import WHARConfig
from whar_datasets.core.postprocessing import postprocess
from whar_datasets.core.preprocessing import preprocess
from whar_datasets.core.sampling import get_label, get_sample
from whar_datasets.core.splitting import get_split_train_val_test
from whar_datasets.core.utils.loading import load_session_metadata, load_window_metadata
from whar_datasets.support.sensor_types import get_sensor_types


class NumpyLocationAdapter:
    def __init__(self, cfg: WHARConfig, override_cache: bool = False):
        self.cfg = cfg
        self.override_cache = override_cache

        dirs = preprocess(cfg, override_cache)

        self.cache_dir, self.windows_dir, self.samples_dir, self.hashes_dir = dirs

        self.session_metadata = load_session_metadata(self.cache_dir)
        self.window_metadata = load_window_metadata(self.cache_dir)

        self.num_classes = len(self.session_metadata["activity_id"].unique())

        print(f"subject_ids: {np.sort(self.session_metadata['subject_id'].unique())}")
        print(f"activity_ids: {np.sort(self.session_metadata['activity_id'].unique())}")

        np.random.seed(cfg.seed)

        sensor_locations, sensor_types = get_sensor_types(cfg.dataset_id)
        self.sensor_locations = sensor_locations
        self.sensor_types = sensor_types
        self.num_channels = len(sensor_locations)
        self.unique_locations = list(dict.fromkeys(sensor_locations))
        self.location_channel_indices: Dict[int, List[int]] = {
            loc: [idx for idx, loc_value in enumerate(sensor_locations) if loc_value == loc]
            for loc in self.unique_locations
        }

        (
            train_window_indices,
            val_window_indices,
            test_window_indices,
        ) = get_split_train_val_test(
            self.cfg,
            self.session_metadata,
            self.window_metadata,
            None,
        )

        cached_samples = postprocess(
            self.cfg,
            train_window_indices,
            self.hashes_dir,
            self.samples_dir,
            self.windows_dir,
            self.window_metadata,
            override_cache,
        )

        # Load all the data in memory, splitting per-location and skipping incomplete IMUs.
        labels: List[int] = []
        samples: List[List[np.ndarray]] = []
        locations: List[int] = []
        participants: List[int] = []
        index_map: List[Tuple[int, int]] = []

        for idx in range(len(self.window_metadata)):
            label = get_label(
                idx,
                self.window_metadata,
                self.session_metadata,
            )
            sample = get_sample(
                idx,
                self.cfg,
                self.samples_dir,
                self.window_metadata,
                cached_samples,
            )
            session_id = int(self.window_metadata.at[idx, "session_id"])
            participant_id = int(
                self.session_metadata.loc[
                    self.session_metadata["session_id"] == session_id, "subject_id"
                ].item()
            )

            for location in self.unique_locations:
                channel_indices = self.location_channel_indices[location]
                sliced_sample: List[np.ndarray] = []
                valid_location = True

                for component in sample:
                    if (
                        isinstance(component, np.ndarray)
                        and component.ndim > 0
                        and component.shape[-1] == self.num_channels
                    ):
                        if not self._has_full_location(component, channel_indices):
                            valid_location = False
                            break
                        sliced_sample.append(component[..., channel_indices])
                    else:
                        sliced_sample.append(component)

                if not valid_location:
                    continue

                labels.append(label)
                samples.append(sliced_sample)
                locations.append(location)
                participants.append(participant_id)
                index_map.append((idx, location))

        self.labels = np.array(labels)
        self.samples = np.array(samples)
        self.samples = self.samples.transpose(0,2,3,1)
        self.locations = np.array(locations)
        self.participants = np.array(participants)
        self.index_map = index_map

        self.train_indices = self._window_to_dataset_indices(train_window_indices)
        self.val_indices = self._window_to_dataset_indices(val_window_indices)
        self.test_indices = self._window_to_dataset_indices(test_window_indices)

    def _window_to_dataset_indices(self, window_indices: List[int]) -> List[int]:
        lookup = set(window_indices)
        return [
            dataset_idx
            for dataset_idx, (window_idx, _location) in enumerate(self.index_map)
            if window_idx in lookup
        ]

    def _has_full_location(self, component: np.ndarray, channel_indices: List[int]) -> bool:
        if not channel_indices or component.shape[-1] <= max(channel_indices):
            return False
        sliced = component[..., channel_indices]
        return not np.isnan(sliced).any()

    def get_keras_dataloaders(self):
        """Return train/val/test tuples ready for Keras.

        Each split is a tuple of (x, y, locations, participants). If multiple feature
        components exist, x is a list of arrays stacked per component; otherwise x is
        a single array.
        """

        def _gather(indices: List[int]):
            # Directly index into the already-stacked samples to keep the
            # sample axis leading: (num_samples, timesteps, channels, 1).
            x = self.samples[indices]
            y = self.labels[indices]
            locs = self.locations[indices]
            participants = self.participants[indices]
            return x, y, locs, participants

        train = _gather(self.train_indices)
        val = _gather(self.val_indices)
        test = _gather(self.test_indices)

        return train, val, test
