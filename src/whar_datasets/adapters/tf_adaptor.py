import random
from typing import Generator, List, Tuple
import numpy as np
import tensorflow as tf

from whar_datasets.core.pipeline import PostProcessingPipeline, PreProcessingPipeline
from whar_datasets.core.sampling import get_label, get_sample
from whar_datasets.core.splitting import get_split_train_val_test
from whar_datasets.core.weighting import compute_class_weights
from whar_datasets.core.config import WHARConfig


class TensorflowAdapter:
    def __init__(
        self,
        cfg: WHARConfig,
        force_recompute: bool | List[bool] | None = False,
    ) -> None:
        self.cfg: WHARConfig = cfg

        # Set seeds
        tf.random.set_seed(cfg.seed)
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)

        # Preprocessing
        self.pre_processing_pipeline: PreProcessingPipeline = PreProcessingPipeline(cfg)
        results = self.pre_processing_pipeline.run(force_recompute)
        self.activity_metadata, self.session_metadata, self.window_metadata = results

    def get_datasets(
        self,
        batch_size: int,
        scv_group_index: int | None = None,
        force_recompute: bool | List[bool] | None = False,
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        # Compute splits
        self.train_indices, self.val_indices, self.test_indices = (
            get_split_train_val_test(
                self.cfg,
                self.session_metadata,
                self.window_metadata,
                scv_group_index,
            )
        )

        # Postprocessing
        self.post_processing_pipeline = PostProcessingPipeline(
            self.cfg,
            self.pre_processing_pipeline,
            self.window_metadata,
            self.train_indices,
        )
        self.samples = self.post_processing_pipeline.run(force_recompute)

        # Build datasets
        train_ds = self._build_tf_dataset(self.train_indices, batch_size, shuffle=True)
        val_ds = self._build_tf_dataset(
            self.val_indices, len(self.val_indices), shuffle=False
        )
        test_ds = self._build_tf_dataset(self.test_indices, 1, shuffle=False)

        return train_ds, val_ds, test_ds

    def _build_tf_dataset(
        self, indices: List[int], batch_size: int, shuffle: bool
    ) -> tf.data.Dataset:
        def generator() -> Generator[Tuple[np.ndarray, ...], None, None]:
            for idx in indices:
                yield self._get_item(idx)

        # Infer sample shapes using one sample
        y, *x = self._get_item(indices[0])
        output_signature: Tuple[tf.TensorSpec, ...] = (
            tf.TensorSpec(shape=(), dtype=tf.int64),
        )
        for arr in x:
            output_signature += (tf.TensorSpec(shape=arr.shape, dtype=tf.float32),)

        ds = tf.data.Dataset.from_generator(
            generator, output_signature=output_signature
        )

        if shuffle:
            ds = ds.shuffle(len(indices), seed=self.cfg.seed)
        ds = ds.batch(batch_size)
        return ds.prefetch(tf.data.AUTOTUNE)

    def get_class_weights(self, dataset: tf.data.Dataset) -> dict:
        # Note: TensorFlow datasets don't expose indices directly
        # You'll need to provide indices manually if you want exact class weights
        return compute_class_weights(
            self.session_metadata,
            self.window_metadata.iloc[
                self.train_indices
            ],  # safest: compute on training split
        )

    def _get_item(self, index: int) -> Tuple[np.ndarray, ...]:
        label: int = get_label(index, self.window_metadata, self.session_metadata)
        sample: list = get_sample(
            index,
            self.cfg,
            self.post_processing_pipeline.samples_dir,
            self.window_metadata,
            self.samples,
        )

        y: np.ndarray = np.array(label, dtype=np.int64)
        x: List[np.ndarray] = [np.array(s, dtype=np.float32) for s in sample]

        return (y, *x)
