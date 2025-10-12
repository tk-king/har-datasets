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
    def __init__(self, cfg: WHARConfig):
        self.cfg = cfg
        self._set_seed()
        self.pre_pipeline = PreProcessingPipeline(cfg)

    def _set_seed(self):
        tf.random.set_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        random.seed(self.cfg.seed)

    def preprocess(self, force_recompute: bool | List[bool] | None = False):
        self.activity_meta, self.session_meta, self.window_meta = self.pre_pipeline.run(
            force_recompute
        )

    def postprocess(
        self,
        scv_group_index: int | None = None,
        force_recompute: bool | List[bool] | None = False,
    ):
        self.train_indices, self.val_indices, self.test_indices = (
            get_split_train_val_test(
                self.cfg, self.session_meta, self.window_meta, scv_group_index
            )
        )
        self.post_pipeline = PostProcessingPipeline(
            self.cfg,
            self.pre_pipeline,
            self.window_meta,
            self.train_indices,
        )
        self.samples = self.post_pipeline.run(force_recompute)

    def _build_tf_dataset(
        self, indices: List[int], batch_size: int, shuffle: bool
    ) -> tf.data.Dataset:
        """Internal helper to build a tf.data.Dataset from index list."""

        def generator() -> Generator[Tuple[np.ndarray, ...], None, None]:
            for idx in indices:
                yield self._get_item(idx)

        # Infer output signature from one sample
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

    def _get_item(self, index: int) -> Tuple[np.ndarray, ...]:
        label = get_label(index, self.window_meta, self.session_meta)
        sample = get_sample(
            index,
            self.cfg,
            self.post_pipeline.samples_dir,
            self.window_meta,
            self.samples,
        )
        y = np.array(label, dtype=np.int64)
        x = [np.array(s, dtype=np.float32) for s in sample]
        return (y, *x)

    def get_datasets(self, batch_size: int) -> dict[str, tf.data.Dataset]:
        train_ds = self._build_tf_dataset(self.train_indices, batch_size, shuffle=True)
        val_ds = self._build_tf_dataset(
            self.val_indices, len(self.val_indices), shuffle=False
        )
        test_ds = self._build_tf_dataset(self.test_indices, 1, shuffle=False)

        return {"train": train_ds, "val": val_ds, "test": test_ds}

    def get_class_weights(self, indices: List[int]) -> dict:
        return compute_class_weights(
            self.session_meta,
            self.window_meta.iloc[indices],
        )
