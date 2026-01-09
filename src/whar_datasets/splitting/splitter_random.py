import math
from typing import List, Tuple

import numpy as np
import pandas as pd

from whar_datasets.config.config import WHARConfig
from whar_datasets.splitting.split import Split
from whar_datasets.splitting.splitter import Splitter


class RandomSplitter(Splitter):
    def __init__(
        self,
        cfg: WHARConfig,
        *,
        train_percentage: float | None = None,
        val_percentage: float | None = None,
        test_percentage: float | None = None,
    ):
        super().__init__(cfg)

        self.val_percentage = val_percentage if val_percentage is not None else cfg.val_percentage
        self.test_percentage = (
            test_percentage if test_percentage is not None else cfg.test_percentage
        )

        if train_percentage is None:
            self.train_percentage = 1.0 - self.val_percentage - self.test_percentage
        else:
            self.train_percentage = train_percentage

        assert self.train_percentage >= 0.0, "train_percentage must be >= 0"
        assert self.val_percentage >= 0.0, "val_percentage must be >= 0"
        assert self.test_percentage >= 0.0, "test_percentage must be >= 0"
        assert math.isclose(
            self.train_percentage + self.val_percentage + self.test_percentage, 1.0
        ), "train/val/test percentages must sum to 1.0"

    def _counts(self, n_total: int) -> Tuple[int, int, int]:
        ratios = (self.train_percentage, self.val_percentage, self.test_percentage)
        raw = [r * n_total for r in ratios]
        floors = [int(math.floor(v)) for v in raw]
        remainder = n_total - sum(floors)

        fractional = [v - math.floor(v) for v in raw]
        order = np.argsort(fractional)[::-1]
        for i in order[:remainder]:
            floors[int(i)] += 1

        return floors[0], floors[1], floors[2]

    def get_splits(
        self,
        session_df: pd.DataFrame,
        window_df: pd.DataFrame,
    ) -> List[Split]:
        indices = list(window_df.index)
        self.rng.shuffle(indices)

        n_train, n_val, n_test = self._counts(len(indices))

        train_indices = indices[:n_train]
        val_indices = indices[n_train : n_train + n_val]
        test_indices = indices[n_train + n_val : n_train + n_val + n_test]

        split = Split(
            identifier="random",
            train_indices=train_indices,
            val_indices=val_indices,
            test_indices=test_indices,
        )

        assert not self.check_indices_overlap(
            split.train_indices, split.val_indices, split.test_indices
        ), "Overlap detected in indices!"

        return [split]
