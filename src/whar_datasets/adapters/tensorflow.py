import random
from typing import Tuple
import numpy as np
import tensorflow as tf

from whar_datasets.core.preprocessing import preprocess
from whar_datasets.core.sampling import get_label, get_window
from whar_datasets.core.splitting import get_split
from whar_datasets.core.normalization import normalize_windows
from whar_datasets.core.utils.loading import load_session_metadata, load_window_metadata
from whar_datasets.core.weighting import compute_class_weights
from whar_datasets.core.config import WHARConfig


class TensorflowAdapter:
    def __init__(self, cfg: WHARConfig, override_cache: bool = False):
        self.cfg = cfg

        dirs = preprocess(cfg, override_cache)
        self.cache_dir, self.windows_dir, self.hashes_dir = dirs

        self.session_metadata = load_session_metadata(self.cache_dir)
        self.window_metadata = load_window_metadata(self.cache_dir)

        print(f"subject_ids: {np.sort(self.session_metadata['subject_id'].unique())}")
        print(f"activity_ids: {np.sort(self.session_metadata['activity_id'].unique())}")

        self.seed = cfg.dataset.training.seed

        random.seed(self.seed)
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)

    def get_datasets(
        self,
        train_batch_size: int,
        train_shuffle: bool = True,
        subj_cross_val_group_index: int | None = None,
        override_cache: bool = False,
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        self.train_indices, self.val_indices, self.test_indices = get_split(
            self.cfg,
            self.session_metadata,
            self.window_metadata,
            subj_cross_val_group_index,
        )

        # normalize windows
        self.window_metadata, self.windows = normalize_windows(
            self.cfg,
            self.train_indices,
            self.hashes_dir,
            self.windows_dir,
            self.window_metadata,
            override_cache,
        )

        # Prepare full dataset
        full_dataset = [self._get_item(i) for i in range(len(self.window_metadata))]

        # Split
        def subset(indices):
            subset_data = [full_dataset[i] for i in indices]
            labels, features = zip(*subset_data)
            ds = tf.data.Dataset.from_tensor_slices(
                (tf.stack(features), tf.stack(labels))
            )
            return ds

        train_ds = subset(self.train_indices)
        val_ds = subset(self.val_indices)
        test_ds = subset(self.test_indices)

        if train_shuffle:
            train_ds = train_ds.shuffle(
                buffer_size=len(self.train_indices), seed=self.seed
            )

        train_ds = train_ds.batch(train_batch_size)
        val_ds = val_ds.batch(len(self.val_indices))
        test_ds = test_ds.batch(1)

        print(
            f"train: {len(self.train_indices)} | val: {len(self.val_indices)} | test: {len(self.test_indices)}"
        )

        return train_ds, val_ds, test_ds

    def get_class_weights(self, indices: list[int]) -> dict:
        return compute_class_weights(
            self.session_metadata, self.window_metadata.iloc[indices]
        )

    def _get_item(self, index: int) -> Tuple[tf.Tensor, tf.Tensor]:
        label = get_label(index, self.window_metadata, self.session_metadata)
        window = get_window(
            index, self.cfg, self.windows_dir, self.window_metadata, self.windows
        )

        y = tf.convert_to_tensor(label, dtype=tf.int64)
        x = tf.convert_to_tensor(window.values, dtype=tf.float32)

        return y, x
