import random
from typing import Dict, List

import numpy as np
import tensorflow as tf

from whar_datasets.config.config import WHARConfig
from whar_datasets.loading.loader import Loader
from whar_datasets.splitting.split import Split


class TFAdapter:
    def __init__(self, cfg: WHARConfig, loader: Loader, split: Split):
        self.cfg = cfg
        self.loader = loader
        self.split = split

        # Detect shapes automatically from the first sample
        self._input_shape = self._infer_shapes()
        self._set_seed()

    def _infer_shapes(self) -> tf.TensorShape:
        _, _, sample = self.loader.get_item(0)
        # Create a list of shapes for each sensor in the sample
        return tf.TensorShape(sample[0].shape)

    def _set_seed(self) -> None:
        tf.random.set_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        random.seed(self.cfg.seed)

    def _generator(self, indices: List[int]) -> tf.Iterator:  # type: ignore
        for idx in indices:
            activity_label, _, sample = self.loader.get_item(idx)

            y = np.array(activity_label, dtype=np.int64)
            # Ensure samples are converted to float32 numpy arrays
            x = np.array(sample[0], dtype=np.float32)

            yield y, x

    def _create_dataset(self, indices: List[int]) -> tf.data.Dataset:
        # Define the explicit signature
        output_signature = (
            tf.TensorSpec(shape=(), dtype=tf.int64),  # Label
            tf.TensorSpec(shape=self._input_shape, dtype=tf.float32),  # Sample
        )

        return tf.data.Dataset.from_generator(
            lambda: self._generator(indices), output_signature=output_signature
        )

    def get_datasets(self, batch_size: int) -> Dict[str, tf.data.Dataset]:
        train_ds = self._create_dataset(self.split.train_indices)
        val_ds = self._create_dataset(self.split.val_indices)
        test_ds = self._create_dataset(self.split.test_indices)

        # Buffer size for shuffle should ideally be the size of the set
        train_ds = (
            train_ds.shuffle(len(self.split.train_indices))
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )

        val_ds = val_ds.batch(len(self.split.val_indices)).prefetch(tf.data.AUTOTUNE)
        test_ds = test_ds.batch(1).prefetch(tf.data.AUTOTUNE)

        return {"train": train_ds, "val": val_ds, "test": test_ds}
