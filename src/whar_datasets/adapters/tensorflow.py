import random
from typing import List, Tuple

import numpy as np
import tensorflow as tf

from whar_datasets.core.normalizing import get_norm_params, normalize_window
from whar_datasets.core.preprocessing import preprocess
from whar_datasets.core.sampling import get_label, get_window
from whar_datasets.core.splitting import get_split
from whar_datasets.core.utils.loading import load_session_metadata, load_windowing
from whar_datasets.core.weighting import compute_class_weights
from whar_datasets.core.config import WHARConfig


class TensorflowAdapter:
    def __init__(self, cfg: WHARConfig, override_cache: bool = False):
        self.cfg = cfg

        self.cache_dir, self.windows_dir = preprocess(cfg, override_cache)
        self.session_metadata = load_session_metadata(self.cache_dir)
        self.window_metadata, self.windows = load_windowing(
            self.cache_dir, self.windows_dir, self.cfg
        )

        print(f"subject_ids: {np.sort(self.session_metadata['subject_id'].unique())}")
        print(f"activity_ids: {np.sort(self.session_metadata['activity_id'].unique())}")

        self.seed = cfg.dataset.training.seed

        random.seed(self.seed)
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)

        # For tracking indices used in each dataset
        self.train_indices: List[int] = []
        self.val_indices: List[int] = []
        self.test_indices: List[int] = []

    def get_tf_dataset(
        self, indices: List[int], batch_size: int, shuffle: bool = False
    ) -> tf.data.Dataset:
        def generator():
            for idx in indices:
                label = get_label(idx, self.window_metadata, self.session_metadata)
                window = get_window(
                    idx, self.cfg, self.windows_dir, self.window_metadata, self.windows
                )
                window = normalize_window(self.cfg, self.norm_params, window)

                yield (
                    tf.convert_to_tensor(window.values, dtype=tf.float32),
                    tf.convert_to_tensor(label, dtype=tf.int32),
                )

        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                tf.TensorSpec(shape=(None, None), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.int32),
            ),
        )

        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(indices), seed=self.seed)

        return dataset.batch(batch_size)

    def get_datasets(
        self,
        train_batch_size: int,
        subj_cross_val_group_index: int | None = None,
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        self.train_indices, self.val_indices, self.test_indices = get_split(
            self.cfg,
            self.session_metadata,
            self.window_metadata,
            subj_cross_val_group_index,
        )

        # get normalization parameters from train indices
        self.norm_params = get_norm_params(
            self.cfg,
            self.train_indices,
            self.windows_dir,
            self.window_metadata,
            self.windows,
        )

        print(
            f"train: {len(self.train_indices)} | val: {len(self.val_indices)} | test: {len(self.test_indices)}"
        )

        train_dataset = self.get_tf_dataset(
            self.train_indices, batch_size=train_batch_size, shuffle=True
        )
        val_dataset = self.get_tf_dataset(
            self.val_indices, batch_size=len(self.val_indices), shuffle=False
        )
        test_dataset = self.get_tf_dataset(
            self.test_indices, batch_size=1, shuffle=False
        )

        return train_dataset, val_dataset, test_dataset

    def get_class_weights(self, dataset: str = "train") -> dict:
        if dataset == "train":
            indices = self.train_indices
        elif dataset == "val":
            indices = self.val_indices
        elif dataset == "test":
            indices = self.test_indices
        else:
            raise ValueError(
                f"Invalid dataset name: {dataset}. Use 'train', 'val', or 'test'."
            )

        return compute_class_weights(
            self.session_metadata, self.window_metadata.iloc[indices]
        )
